"""Batched teacher forcing gives batch one's results (CPU).

One unpadded task must equal ``teacher_force_task`` exactly, for every gene the
search uses, including zero-commit steps. Several left-padded tasks must equal each
task forced alone up to rounding, with identical commits, flips, FLOPs and rows.
The 7B bf16 comparison is experiments/batched_teacher_forcing_check.py.
"""

import importlib
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest

import torch
import yaml

from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.batched_teacher_forcing import NEVER, commit_steps, record_trajectory, teacher_force_batch
from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline, DenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.teacher_forcing import POSITION_TIE_MARGIN, counts_as_position_change, teacher_force_task
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome
from experiments.masked_row_reuse_teacher_forcing import MaskedRowReuseBackend
from test_backend_cpu import FakeModel
from utils.compute_cost import ModelShape, dense_forward_flops
from test_teacher_forcing_compounding import greedy_sample_tokens

MASK = 31
STOPS = {0}
# Every gene, on the fake model's two layers (depth 2 resumes at the second).
GENOMES = [
    KVRefreshPolicyGenome(1, 8, 1, 0),
    KVRefreshPolicyGenome(0, 4, 2, 0),
    KVRefreshPolicyGenome(2, 3, 1, 3),
    KVRefreshPolicyGenome(1, 4, 1, 0, prompt_refresh_interval=3),
    KVRefreshPolicyGenome(1, 4, 2, 0, local_refresh_radius=2),
    KVRefreshPolicyGenome(1, 8, 2, 0, masked_window=3, protected_size=2),
    KVRefreshPolicyGenome(1, 8, 1, 0, masked_window=2, protected_size=3, protected_rule="split"),
    KVRefreshPolicyGenome(1, 8, 1, 0, masked_window=2, protected_size=2, padding_window=4),
    KVRefreshPolicyGenome(1, 8, 1, 0, masked_window=2, protected_size=2, entropy_threshold=0.5),
    KVRefreshPolicyGenome(1, 8, 2, 0, masked_window=4, protected_size=2, entropy_threshold=0.5, entropy_scope="full"),
    # Run 4's edges, scaled to 12 steps: refresh, prompt and padding intervals longer than the
    # generation, one protected row, and a low entropy threshold.
    KVRefreshPolicyGenome(2, 16, 1, 0, masked_window=4, protected_size=1, protected_rule="split",
                          prompt_refresh_interval=16, local_refresh_radius=3, padding_window=16,
                          entropy_threshold=0.25, entropy_scope="full"),
]
DISCRETE = ("step", "committed_positions", "outside_positions", "top1_flip", "position_flip")


def settings(steps=12, gen_length=12):
    return SimpleNamespace(steps=steps, gen_length=gen_length, temperature=0.0, top_p=1.0, top_k=None,
                           algorithm="entropy", algorithm_temperature=0.0)


def fake_sampler(backend):
    sampler = object.__new__(DiffuCoderDenoisingSampler)
    sampler.backend, sampler.model, sampler.sample_tokens = backend, backend.model, greedy_sample_tokens
    return sampler


def batch_one_backend(model, genome):
    policy = genome.policy(STOPS)
    return (MaskedRowReuseBackend if genome.wrapped else DiffuCoderActiveTokenKVBackend)(model, policy)


def with_rows(backends):
    """Record rows per forward as the search's evaluate does."""

    rows = {name: [] for name in backends}
    for name, instance in backends.items():
        forward = instance.forward_mask_logits

        def record(_forward=forward, _name=name, **inputs):
            result = _forward(**inputs)
            decision = result.decision
            rows[_name].append({"active": int(decision.active.sum()), "masked": result.mask_logits.shape[0],
                                "full_depth": int((decision.active & decision.refresh_depth.le(1)).sum())})
            return result

        instance.forward_mask_logits = record
    return rows


def left_pad(prompts, pad=0):
    width = max(len(prompt) for prompt in prompts)
    ids = torch.tensor([[pad] * (width - len(prompt)) + prompt for prompt in prompts])
    mask = torch.tensor([[0] * (width - len(prompt)) + [1] * len(prompt) for prompt in prompts])
    return ids, mask


class CommitStepTests(unittest.TestCase):
    def test_commit_steps_read_the_first_unmasked_step(self):
        history = [torch.tensor([[5, MASK, 7, MASK, MASK]]), torch.tensor([[5, MASK, 7, 8, MASK]]),
                   torch.tensor([[5, 6, 7, 8, MASK]])]
        self.assertEqual(commit_steps(history, 1, MASK), [2, 0, 1, NEVER])


class FakeModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(5)
        self.model = FakeModel().eval()
        for parameter in self.model.parameters():
            parameter.data.mul_(3)  # sharper logits: stale caches visibly drift

    def dense_sampler(self):
        return fake_sampler(DiffuCoderActiveTokenKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1)))

    def batch_one(self, prompt, generation):
        """Batch-one teacher forcing of every genome, and the dense trajectory it followed."""

        prompt = torch.tensor([prompt])
        mask = torch.ones_like(prompt)
        trajectory = record_trajectory(self.dense_sampler(), prompt, mask, generation)
        backends = {genome.name: batch_one_backend(self.model, genome) for genome in GENOMES}
        rows = with_rows(backends)
        result = teacher_force_task(self.model, self.dense_sampler(), backends, prompt, mask, generation, None)
        self.assertEqual(result["dense_token_ids"], trajectory["dense_token_ids"])
        return trajectory, result, rows

    def batched(self, prompts, trajectories, generation, **options):
        ids, mask = left_pad(prompts)
        dense = BatchedDiffuCoderKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        backends = {genome.name: BatchedDiffuCoderKVBackend(self.model, genome.policy(STOPS)) for genome in GENOMES}
        return teacher_force_batch(self.model, dense, backends, ids, mask, trajectories, generation, None, **options)

    def test_vectorized_scoring_equals_per_task_scoring(self):
        # Padded tasks, zero-commit steps and answer ends: the two scoring paths agree exactly.
        prompts = [[4, 5, 6], [7], [8, 9, 10, 11, 12]]
        for generation in (settings(), settings(steps=16, gen_length=8)):
            with self.subTest(steps=generation.steps):
                trajectories = [self.batch_one(prompt, generation)[0] for prompt in prompts]
                ends = [3, None, 0]
                timings = {}
                vectorized = self.batched(prompts, trajectories, generation, answer_ends=ends, timings=timings)
                per_task = self.batched(prompts, trajectories, generation, answer_ends=ends, scoring="per_task")
                self.assertEqual(vectorized, per_task)
                self.assertEqual(set(timings), {"dense_forward", "policy_forward", "scoring", "read_back"})

    def test_one_task_equals_batch_one_exactly(self):
        # 16 steps over 8 positions includes steps that commit nothing.
        for generation in (settings(), settings(steps=16, gen_length=8)):
            with self.subTest(steps=generation.steps):
                trajectory, expected, rows = self.batch_one([4, 5, 6], generation)
                (actual,) = self.batched([[4, 5, 6]], [trajectory], generation)
                self.assertEqual(actual["dense_token_ids"], expected["dense_token_ids"])
                self.assertEqual(actual["flops"], expected["flops"])
                self.assertEqual(actual["rows"], rows)
                self.assertEqual(actual["policies"], expected["policies"])
                if generation.steps > generation.gen_length:
                    self.assertLess(len(actual["policies"][GENOMES[0].name]), generation.steps)
                # Reuse policies commit outside their candidates somewhere, so that path is exercised.
                self.assertTrue(any(row.get("outside_positions") for name in actual["policies"]
                                    for row in actual["policies"][name]))

    def test_padded_tasks_match_each_task_alone(self):
        prompts = [[4, 5, 6], [7], [8, 9, 10, 11, 12]]
        generation = settings()
        alone = [self.batch_one(prompt, generation) for prompt in prompts]
        together = self.batched(prompts, [trajectory for trajectory, _, _ in alone], generation)
        for (trajectory, expected, rows), actual, prompt in zip(alone, together, prompts):
            with self.subTest(prompt=prompt):
                self.assertEqual(actual["dense_token_ids"], expected["dense_token_ids"])
                self.assertEqual(actual["flops"], expected["flops"])
                self.assertEqual(actual["rows"], rows)
                self.assertEqual(actual["dense_flops"], generation.steps * dense_forward_flops(
                    ModelShape.from_model(self.model), len(prompt) + generation.gen_length))
                for name in expected["policies"]:
                    self.assertEqual(len(actual["policies"][name]), len(expected["policies"][name]))
                    for got, want in zip(actual["policies"][name], expected["policies"][name]):
                        self.assert_same_step(got, want, name)

    def assert_same_step(self, got, want, name):
        """What the search reads must agree; tied positions and KL may round differently."""

        where = (name, want["step"])
        for key in DISCRETE:
            if key != "position_flip":
                self.assertEqual(got.get(key), want.get(key), (*where, key))
        # A flip where dense's positions tie is a tie-break; the search does not count it.
        self.assertEqual(counts_as_position_change(got), counts_as_position_change(want), where)
        if want["dense_position_margin"] is None or want["dense_position_margin"] >= POSITION_TIE_MARGIN:
            self.assertEqual(got["position_flip"], want["position_flip"], where)
        torch.testing.assert_close(torch.tensor(got["ear"]), torch.tensor(want["ear"]), rtol=0, atol=1e-5)
        for key in ("ear_masked_answer", "token_tv", "dense_margin", "dense_position_margin"):
            if want[key] is None:
                self.assertIsNone(got[key], (*where, key))
            else:
                self.assertAlmostEqual(got[key], want[key], delta=1e-5, msg=(*where, key))
        for key in ("kl", "kl_masked_mean", "kl_masked_max"):
            self.assertAlmostEqual(got[key], want[key], delta=1e-5 + 1e-3 * abs(want[key]), msg=(*where, key))

    def test_answer_rows_follow_each_tasks_answer_end(self):
        prompts = [[4, 5, 6], [7]]
        generation = settings()
        trajectories = [self.batch_one(prompt, generation)[0] for prompt in prompts]
        ids, mask = left_pad(prompts)
        dense = BatchedDiffuCoderKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        backends = {"r4": BatchedDiffuCoderKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=4))}
        results = teacher_force_batch(self.model, dense, backends, ids, mask, trajectories, generation, None,
                                      answer_ends=[0, 5])
        self.assertTrue(all(row["ear_masked_answer"] is None for row in results[0]["policies"]["r4"]))
        # Task 1 has masked answer rows until its first five positions are committed.
        self.assertIsNotNone(results[1]["policies"]["r4"][0]["ear_masked_answer"])

    def test_wrong_trajectory_length_is_rejected(self):
        ids, mask = left_pad([[4, 5]])
        dense = BatchedDiffuCoderKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        with self.assertRaises(ValueError):
            teacher_force_batch(self.model, dense, {}, ids, mask, [{"dense_token_ids": [1], "commit_steps": [0]}],
                                settings(), None)


class TinyDreamTests(unittest.TestCase):
    """DiffuCoder's own modeling code with tiny random weights, and its native dense forward."""

    @classmethod
    def setUpClass(cls):
        config_path = Path(__file__).parents[1] / "experiments/configs/diffucoder7b_output_variance_greedy_64tasks.yaml"
        root = Path(os.environ.get("DIFFUCODER_SOURCE_DIR", yaml.safe_load(config_path.read_text())["model"]["path"]))
        if not (root / "modeling_dream.py").is_file():
            raise unittest.SkipTest("Local DiffuCoder source unavailable")
        package = ModuleType("ecad_test_dream")
        package.__path__ = [str(root)]
        sys.modules[package.__name__] = package
        previous = sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode = True
            cls.module = importlib.import_module("ecad_test_dream.modeling_dream")
            cls.top_p_logits = importlib.import_module("ecad_test_dream.generation_utils").top_p_logits
        finally:
            sys.dont_write_bytecode = previous

    def setUp(self):
        config = self.module.DreamConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=3,
                                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                                         mask_token_id=63, pad_token_id=0)
        config._attn_implementation = "sdpa"
        torch.manual_seed(3)
        self.model = self.module.DreamModel(config).eval()
        for parameter in self.model.parameters():
            parameter.data.mul_(4)
        self.genomes = [KVRefreshPolicyGenome(1, 4, 1, 0), KVRefreshPolicyGenome(1, 4, 3, 0),
                        KVRefreshPolicyGenome(1, 8, 3, 0, masked_window=3, protected_size=2)]

    def sampler(self):
        sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(self.model))
        sampler.sample_tokens = greedy_sample_tokens_for(63)
        return sampler

    def test_native_dense_padded_batch_matches_batch_one(self):
        prompts, generation = [[5, 6, 7], [9], [10, 11, 12, 13]], settings(steps=10, gen_length=10)
        expected, trajectories = [], []
        for prompt in prompts:
            ids = torch.tensor([prompt])
            mask = torch.ones_like(ids)
            trajectories.append(record_trajectory(self.sampler(), ids, mask, generation))
            backends = {genome.name: batch_one_backend(self.model, genome) for genome in self.genomes}
            expected.append(teacher_force_task(self.model, self.sampler(), backends, ids, mask, generation,
                                               self.top_p_logits))
        # Batch one through the batched path, with the batch-one dense forward: exact.
        ids = torch.tensor([prompts[0]])
        backends = {genome.name: BatchedDiffuCoderKVBackend(self.model, genome.policy(STOPS)) for genome in self.genomes}
        (single,) = teacher_force_batch(self.model, DenseDiffuCoderBaseline(self.model), backends, ids,
                                        torch.ones_like(ids), trajectories[:1], generation, self.top_p_logits)
        self.assertEqual(single["policies"], expected[0]["policies"])
        self.assertEqual(single["dense_flops"], expected[0]["dense_flops"])
        # All three padded, with the batched native dense forward: close.
        ids, mask = left_pad(prompts)
        backends = {genome.name: BatchedDiffuCoderKVBackend(self.model, genome.policy(STOPS)) for genome in self.genomes}
        together = teacher_force_batch(self.model, BatchedDenseDiffuCoderBaseline(self.model), backends, ids, mask,
                                       trajectories, generation, self.top_p_logits)
        for actual, want in zip(together, expected):
            self.assertEqual(actual["flops"], want["flops"])
            self.assertEqual(actual["dense_flops"], want["dense_flops"])
            for name, rows in want["policies"].items():
                for got, row in zip(actual["policies"][name], rows):
                    self.assertEqual(got["committed_positions"], row["committed_positions"])
                    torch.testing.assert_close(torch.tensor(got["ear"]), torch.tensor(row["ear"]), rtol=1e-4, atol=1e-5)
                    self.assertAlmostEqual(got["kl_masked_mean"], row["kl_masked_mean"], delta=1e-4)


def greedy_sample_tokens_for(mask_token_id):
    def sample(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
        logits = logits.clone()
        logits[:, mask_token_id] = -torch.inf
        probs = torch.softmax(logits, dim=-1)
        confidence, tokens = probs.max(dim=-1)
        if neg_entropy:
            confidence = torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
        return confidence, tokens
    return sample


if __name__ == "__main__":
    unittest.main()
