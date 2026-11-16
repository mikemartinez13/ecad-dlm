import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from experiments.commit_position_study import TaskLog, protected_set
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.forced_trajectory_replay import replay_task
from experiments.masked_row_reuse_replay import MaskedRowReuseBackend, MaskedRowReusePolicy
from experiments.stale_kv_error_study import same_steps
from test_backend_cpu import FakeModel
from test_replay_compounding import greedy_sample_tokens

# The sampler looks up sample_tokens in the module that defines the model class.
sample_tokens = greedy_sample_tokens

REFRESH_256 = DKVDecodeRefreshPolicy(cache_reloading_step=256)
STOP_IDS = {0}


def replay(model, backends, gen_length=16):
    reference = DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
    sampler = object.__new__(DiffuCoderDenoisingSampler)
    sampler.backend, sampler.model, sampler.sample_tokens = reference, model, greedy_sample_tokens
    generation = SimpleNamespace(steps=gen_length, gen_length=gen_length, temperature=0.0, top_p=1.0, top_k=None,
                                 algorithm="entropy", algorithm_temperature=0.0)
    prompt = torch.tensor([[4, 5, 6]])
    return replay_task(model, sampler, backends, prompt, torch.ones_like(prompt), generation, None)


class MaskedRowReuseTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)
        self.model = FakeModel().eval()
        for parameter in self.model.parameters():
            parameter.data.mul_(3)

    def reuse(self, window, m=2, rule="split+recent"):
        return MaskedRowReuseBackend(self.model, MaskedRowReusePolicy(REFRESH_256, window, m, rule, STOP_IDS))

    def test_window_one_is_refresh_256(self) -> None:
        backends = {"base": DiffuCoderActiveTokenKVBackend(self.model, REFRESH_256), "check": self.reuse(1)}
        steps = replay(self.model, backends)["policies"]
        self.assertTrue(same_steps(steps["check"], steps["base"]))
        self.assertTrue(all(row["outside_positions"] == [] for row in steps["check"]))

    def test_reuse_skips_rows_and_blocks_commits_outside_the_protected_set(self) -> None:
        backends = {"base": DiffuCoderActiveTokenKVBackend(self.model, REFRESH_256), "r4": self.reuse(4, m=1, rule="split")}
        result = replay(self.model, backends)
        self.assertLess(result["flops"]["r4"], result["flops"]["base"])
        rows = backends["r4"].active_rows
        self.assertGreater(rows[4], rows[5])  # step 4 refreshes every masked row, step 5 reuses most
        # The last step commits two positions here, which a one-candidate step cannot flag.
        outside = [row for row in result["policies"]["r4"] if row["outside_positions"] and row["step"] < 15]
        self.assertTrue(outside)
        self.assertTrue(all(row["position_flip"] for row in outside))
        self.assertTrue(all(row["step"] % 4 and row["step"] > 1 for row in outside))

    def test_reused_positions_keep_their_last_logits(self) -> None:
        backend = self.reuse(4, m=1, rule="split")
        seen = []
        forward = backend.forward_mask_logits

        def recording(**inputs):
            result = forward(**inputs)
            seen.append((inputs["step"], backend.policy.mask_positions.clone(), result.mask_logits.clone(),
                         backend.policy.protected.clone()))
            return result

        backend.forward_mask_logits = recording
        replay(self.model, {"r4": backend})
        (_, positions_4, logits_4, _), (_, positions_5, logits_5, protected_5) = seen[4], seen[5]
        # Position p is predicted from row p-1; when both are masked and neither is recomputed, nothing changes.
        computed = protected_5 | torch.cat([protected_5[1:], torch.zeros(1, dtype=torch.bool)])
        checked = 0
        for row, position in enumerate(positions_5.tolist()):
            if position - 1 in positions_5.tolist() and not computed[position - 1]:
                torch.testing.assert_close(logits_5[row], logits_4[positions_4.tolist().index(position)])
                checked += 1
        self.assertGreater(checked, 0)

    def test_depth_base_keeps_its_decoded_rows(self) -> None:
        # The fake has two layers, so refresh_depth 2 recomputes decoded rows in the last layer only.
        depth = DKVDecodeRefreshPolicy(cache_reloading_step=8, refresh_depth=2)
        check = MaskedRowReuseBackend(self.model, MaskedRowReusePolicy(depth, 1, 2, "split+recent", STOP_IDS))
        reuse = MaskedRowReuseBackend(self.model, MaskedRowReusePolicy(depth, 4, 1, "split", STOP_IDS))
        backends = {"base": DiffuCoderActiveTokenKVBackend(self.model, depth), "check": check, "r4": reuse}
        result = replay(self.model, backends)
        steps = result["policies"]
        self.assertTrue(same_steps(steps["check"], steps["base"]))
        self.assertLess(result["flops"]["r4"], result["flops"]["base"])
        # Step 5 is cheap for both: decoded rows run in the last layer only, and r4 also reuses masked rows.
        self.assertLess(check.full_depth_rows[5], check.active_rows[5])
        self.assertEqual(check.active_rows[5] - check.full_depth_rows[5], reuse.active_rows[5] - reuse.full_depth_rows[5])
        self.assertLess(reuse.full_depth_rows[5], check.full_depth_rows[5])

    def test_generation_commits_only_candidates(self) -> None:
        def generate(backend):
            sampler = object.__new__(DiffuCoderDenoisingSampler)
            sampler.backend, sampler.model, sampler.sample_tokens = backend, self.model, greedy_sample_tokens
            prompt = torch.tensor([[4, 5, 6]])
            with mock.patch("torch.cuda.synchronize"):  # generate times itself on a GPU
                return sampler.generate(prompt, attention_mask=torch.ones_like(prompt), steps=16, gen_length=16,
                                        temperature=0.0, top_p=1.0, top_k=None, algorithm="entropy",
                                        algorithm_temperature=0.0, collect_history=True)

        base = generate(DiffuCoderActiveTokenKVBackend(self.model, REFRESH_256))
        self.assertEqual(generate(self.reuse(1)).sequences.tolist(), base.sequences.tolist())
        backend = self.reuse(4, m=1, rule="split")
        allowed = []
        forward = backend.forward_mask_logits

        def recording(**inputs):
            result = forward(**inputs)
            allowed.append(set(backend.policy.mask_positions[result.candidates].tolist()))
            return result

        backend.forward_mask_logits = recording
        history = generate(backend).history
        previous = torch.nn.functional.pad(torch.tensor([[4, 5, 6]]), (0, 16), value=31)
        for step, tokens in enumerate(history[:-1]):  # the last step commits every remaining position
            committed = set((previous[0] != tokens[0]).nonzero()[:, 0].tolist())
            self.assertLessEqual(committed, allowed[step])
            previous = tokens

    def test_protected_set_matches_step_0_rule(self) -> None:
        rng = np.random.default_rng(0)
        steps = length = 40
        order = rng.permutation(length)
        committed = np.zeros((steps, length), bool)
        committed[np.arange(steps), order] = True
        tokens = rng.integers(1, 6, length)
        tokens[25:] = 0
        log = TaskLog({"confidence": rng.normal(size=(steps, length)).astype(np.float32),
                       "top1": rng.integers(0, 3, (steps, length)).astype(np.int32), "committed": committed,
                       "dense_token_ids": tokens, "prompt_length": 3, "stop_ids": np.array([0])}, 0)
        for rule in ("split", "split+recent"):
            for step in (5, 12, 30):
                lag = 4
                policy = MaskedRowReusePolicy(REFRESH_256, lag, 8, rule, STOP_IDS)
                policy.confidence = torch.from_numpy(log.confidence[step - lag].copy())
                policy.top1 = torch.from_numpy(log.top1[step - lag].astype(np.int64))
                decided_at = np.where(log.done_before[step], np.argmax(committed, axis=0), -1)
                decoded_age = torch.from_numpy(np.where(decided_at >= 0, step - decided_at, -1))
                live = policy.protected_positions(torch.from_numpy(log.masked[step]), decoded_age)
                np.testing.assert_array_equal(live.numpy(), protected_set(log, step, lag, rule, 8))


if __name__ == "__main__":
    unittest.main()
