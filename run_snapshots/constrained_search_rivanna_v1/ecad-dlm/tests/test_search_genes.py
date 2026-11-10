import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.denoising_token_state import DenoisingTokenCategory as C
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.kv_policy_experiment_config import KVPolicyNSGA2SearchConfig
from dlm_kv_policy.masked_row_reuse_backend import MaskedRowReuseBackend
from dlm_kv_policy.masked_row_reuse_policy import MaskedRowReusePolicy
from evolution import constrained_kv_policy_search as search
from evolution.nsga2_kv_policy_search import (
    EvaluatedKVPolicy,
    KVRefreshPolicyGenome as Genome,
    _dominates,
    environmental_selection,
    initial_population,
    offspring_population,
)
from experiments.batched_cost_model import predicted_seconds
from experiments.task_sets import deal
from experiments.tau_confirmation import decide as tau_decide, feasibility, ranking
from test_backend_cpu import FakeModel
from test_batched_backend import generate
from test_replay_compounding import greedy_sample_tokens

# The sampler looks up sample_tokens in the module that defines the model class.
sample_tokens = greedy_sample_tokens

STOP = {0}


def decide(policy, states, step, decoded_age=None):
    states = torch.tensor([states])
    age = decoded_age if decoded_age is not None else torch.full_like(states, -1)
    return policy.decide(states=states, decoded_age=age, token_cache_age=None, cache_ready=True, step=step,
                         total_steps=64, num_layers=4)


def reuse(base=None, window=1, m=1, rule="split", **kwargs):
    policy = MaskedRowReusePolicy(base or DKVDecodeRefreshPolicy(cache_reloading_step=8), window, m, rule, STOP, **kwargs)
    policy.reset(8, torch.device("cpu"))
    return policy


class PromptIntervalTests(unittest.TestCase):
    STATES = [C.PROMPT, C.PROMPT, C.STABLE_DECODED, C.MASK]

    def test_periodic_refresh_skips_the_prompt_off_its_own_schedule(self) -> None:
        policy = DKVDecodeRefreshPolicy(cache_reloading_step=8, prompt_refresh_interval=32)
        self.assertEqual(decide(policy, self.STATES, 8).active[0].tolist(), [False, False, True, True])
        self.assertTrue(decide(policy, self.STATES, 32).active.all())

    def test_unset_and_dividing_intervals_keep_every_full_refresh(self) -> None:
        for interval in (None, 4):
            policy = DKVDecodeRefreshPolicy(cache_reloading_step=8, prompt_refresh_interval=interval)
            self.assertTrue(decide(policy, self.STATES, 8).active.all())

    def test_prompt_still_refreshes_on_its_step_between_full_refreshes(self) -> None:
        policy = DKVDecodeRefreshPolicy(cache_reloading_step=32, prompt_refresh_interval=8)
        self.assertTrue(decide(policy, self.STATES, 8).active[0, :2].all())
        self.assertFalse(decide(policy, self.STATES, 9).active[0, :2].any())

    def test_opening_steps_stay_full(self) -> None:
        policy = DKVDecodeRefreshPolicy(cache_reloading_step=8, prompt_refresh_interval=32)
        self.assertTrue(decide(policy, self.STATES, 1).active.all())


class PaddingWindowTests(unittest.TestCase):
    STATES = [C.PROMPT, C.STABLE_DECODED, C.MASK, C.MASK, C.MASK, C.MASK, C.MASK, C.MASK]

    def policy(self, **kwargs):
        policy = reuse(**kwargs)
        policy.top1 = torch.tensor([9, 9, 5, 5, 0, 0, 0, 0])  # the last four predict a stop token
        policy.confidence = torch.tensor([0.0, 0.0, -1.0, -2.0, -0.1, -0.2, -0.3, -0.4])
        return policy

    def test_padding_rows_skip_a_full_refresh_off_their_window(self) -> None:
        policy = self.policy(window=1, m=1, padding_window=16)
        policy.stop_ids = [0]
        active = decide(policy, self.STATES, 8).active[0]
        self.assertTrue(active[:4].all())  # prompt, decoded and answer-like rows
        self.assertEqual(policy.protected.tolist()[2:4], [True, True])

    def test_padding_rows_refresh_on_their_window(self) -> None:
        policy = self.policy(window=2, padding_window=16)
        self.assertTrue(decide(policy, self.STATES, 16).active.all())
        self.assertTrue(policy.protected[2:].all())

    def test_only_protected_padding_rows_can_commit_between_windows(self) -> None:
        import dlm_kv_policy.masked_row_reuse_policy as module
        with mock.patch.object(module, "PADDING_TOP", 1):
            policy = self.policy(window=1, padding_window=16)
            active = decide(policy, self.STATES, 8).active[0]
        self.assertEqual(policy.protected.tolist(), [False, False, True, True, True, False, False, False])
        self.assertEqual(active.tolist(), [True, True, True, True, True, False, False, False])

    def test_padding_window_must_be_a_multiple_of_the_window(self) -> None:
        with self.assertRaises(ValueError):
            reuse(window=4, padding_window=6).validate()


class EntropyTriggerTests(unittest.TestCase):
    STATES = [C.PROMPT, C.STABLE_DECODED, C.STABLE_DECODED, C.NEWLY_DECODED, C.STABLE_DECODED, C.MASK, C.MASK, C.MASK]

    def policy(self, entropy, scope):
        base = DKVDecodeRefreshPolicy(cache_reloading_step=64, refresh_depth=3, additional_decoded_forwards=0)
        policy = reuse(base, window=4, m=1, entropy_threshold=1.0, entropy_scope=scope)
        policy.confidence = torch.full((8,), -0.1)
        policy.confidence[3] = -entropy
        policy.top1 = torch.full((8,), 5)
        return policy

    def test_low_entropy_commit_does_nothing(self) -> None:
        quiet = self.policy(0.5, "full")
        plain = reuse(quiet.base, window=4, m=1)
        plain.confidence, plain.top1 = quiet.confidence, quiet.top1
        self.assertTrue(torch.equal(decide(quiet, self.STATES, 5).active, decide(plain, self.STATES, 5).active))

    def test_local_scope_refreshes_committed_neighbors_from_layer_one(self) -> None:
        decision = decide(self.policy(2.0, "local"), self.STATES, 5)
        self.assertEqual(decision.refresh_depth[0, 1:5].tolist(), [1, 1, 1, 1])
        self.assertFalse(decision.active[0, 0])  # the prompt is not a committed row

    def test_full_scope_refreshes_the_whole_request_and_frees_its_commits(self) -> None:
        policy = self.policy(2.0, "full")
        decision = decide(policy, self.STATES, 5)
        self.assertTrue(decision.active.all())
        self.assertEqual(decision.refresh_depth.max().item(), 1)
        self.assertTrue(policy.protected[5:].all())


class BatchedBackendTests(unittest.TestCase):
    def test_batch_one_matches_the_reference_for_every_new_gene(self) -> None:
        torch.manual_seed(12)
        model = FakeModel().eval()
        genomes = [Genome(1, 4, 2, 0, 1, 32, "split", 8), Genome(1, 4, 1, 0, 1, 32, "split", 0, 2),
                   Genome(1, 4, 2, 0, 2, 4, "split+recent", 0, 0, 8), Genome(1, 8, 2, 0, 2, 4, "split", 8, 2, 8),
                   Genome(1, 8, 1, 0, 1, 32, "split", 0, 0, 0, 0.001, "local"),
                   Genome(1, 8, 2, 0, 2, 4, "split", 0, 0, 0, 0.001, "full")]
        prompt = torch.tensor([[4, 5, 6]])
        for genome in genomes:
            policy = genome.policy(STOP)
            with mock.patch("torch.cuda.synchronize"):
                expected = generate(policy.make_backend(model) if genome.wrapped else
                                    DiffuCoderActiveTokenKVBackend(model, policy), prompt)
                actual = generate(BatchedDiffuCoderKVBackend(model, genome.policy(STOP)), prompt)
            self.assertEqual([h.tolist() for h in actual.history], [h.tolist() for h in expected.history], genome.name)


class GenomeTests(unittest.TestCase):
    def test_round_trip_and_old_checkpoints(self) -> None:
        genome = Genome(2, 16, 22, 8, 4, 48, "split", 32, 2, 16, 1.5, "full")
        self.assertEqual(Genome.from_dict(genome.to_dict()), genome)
        self.assertEqual(Genome.from_dict({"delay": 1, "refresh_interval": 8, "refresh_depth": 22, "terminal_refresh": 0}),
                         Genome(1, 8, 22, 0))

    def test_inert_genes_are_canonicalized(self) -> None:
        self.assertEqual(Genome(1, 8, 22, 0, 1, 64, "split").canonical(), Genome(1, 8, 22, 0))
        self.assertEqual(Genome(1, 8, 22, 0, entropy_scope="full").canonical(), Genome(1, 8, 22, 0))
        kept = Genome(1, 8, 22, 0, 1, 64, "split", padding_window=8)
        self.assertEqual(kept.canonical(), kept)

    def test_policy_type_and_genes(self) -> None:
        plain = Genome(1, 8, 22, 0, prompt_refresh_interval=32, local_refresh_radius=2).policy()
        self.assertIsInstance(plain, DKVDecodeRefreshPolicy)
        self.assertEqual((plain.prompt_refresh_interval, plain.local_refresh_radius), (32, 2))
        wrapped = Genome(1, 8, 22, 0, 4, 48, "split", padding_window=16, entropy_threshold=0.5, entropy_scope="full").policy(STOP)
        self.assertEqual((wrapped.window, wrapped.m, wrapped.rule, wrapped.padding_window, wrapped.entropy_threshold,
                          wrapped.entropy_scope), (4, 48, "split", 16, 0.5, "full"))
        wrapped.validate()
        self.assertIsInstance(wrapped.make_backend(FakeModel()), MaskedRowReuseBackend)
        with self.assertRaises(ValueError):
            Genome(1, 8, 22, 0, 4).policy()

    def test_offspring_stay_in_the_space_and_are_canonical(self) -> None:
        config = settings()
        population = [individual(genome, seconds=10.0 + i) for i, genome in enumerate(initial_population(config))]
        children = offspring_population(population, config, 1)
        allowed = config.choices()
        for child in children:
            self.assertEqual(child, child.canonical())
            self.assertTrue(all(value in options for value, options in zip(child.to_dict().values(), allowed)))

    def test_config_rejects_a_padding_window_the_masked_window_does_not_divide(self) -> None:
        with self.assertRaises(ValueError):
            KVPolicyNSGA2SearchConfig(masked_windows=[1, 4], padding_windows=[0, 6]).validate()


def settings(**kwargs):
    values = dict(population_size=6, generations=2, quality_objective="constrained_batched_time", delays=[1, 2],
                  refresh_intervals=[8, 16], refresh_depths=[22, 24], terminal_refresh_steps=[0, 8], masked_windows=[1, 2, 4],
                  protected_sizes=[32, 64], protected_rules=["split", "split+recent"], prompt_refresh_intervals=[0, 32],
                  local_refresh_radii=[0, 2], padding_windows=[0, 8], entropy_thresholds=[0.0, 1.5], entropy_scopes=["local", "full"],
                  cost_table="table.json", dense_reference="dense")
    return KVPolicyNSGA2SearchConfig(**{**values, **kwargs})


def individual(genome=Genome(1, 8, 22, 0), seconds=10.0, violation=0.0):
    return EvaluatedKVPolicy(genome=genome, quality_loss=0.0, compute_fraction=0.5, predicted_seconds=seconds,
                             violation=violation, constraints={"predicted_speedup": 2.0, "ear": 0.99})


class ConstrainedDominationTests(unittest.TestCase):
    def test_feasible_beats_infeasible_however_slow(self) -> None:
        self.assertTrue(_dominates(individual(seconds=99.0), individual(seconds=1.0, violation=0.001)))
        self.assertFalse(_dominates(individual(seconds=1.0, violation=0.001), individual(seconds=99.0)))

    def test_smaller_violation_wins_between_infeasible(self) -> None:
        self.assertTrue(_dominates(individual(seconds=99.0, violation=0.01), individual(seconds=1.0, violation=0.02)))

    def test_feasible_policies_rank_on_time_alone(self) -> None:
        self.assertTrue(_dominates(individual(seconds=1.0), individual(seconds=2.0)))
        self.assertFalse(_dominates(individual(seconds=2.0), individual(seconds=2.0)))

    def test_selection_keeps_fast_feasible_policies(self) -> None:
        items = [individual(Genome(1, 8, 22, 0, entropy_threshold=0.0), 3.0), individual(Genome(2, 8, 22, 0), 1.0),
                 individual(Genome(1, 16, 22, 0), 0.5, violation=0.1), individual(Genome(2, 16, 22, 0), 2.0)]
        kept = environmental_selection(items, 2)
        self.assertEqual(sorted(item.predicted_seconds for item in kept), [1.0, 2.0])

    def test_legacy_two_objective_ranking_is_unchanged(self) -> None:
        left = EvaluatedKVPolicy(Genome(1, 8, 22, 0), 0.1, 0.5)
        right = EvaluatedKVPolicy(Genome(1, 16, 22, 0), 0.2, 0.5)
        self.assertTrue(_dominates(left, right))


def table():
    curve = {"0": 0.0, "1": 1.0, "400": 400.0}
    blocks = {f"{a}:{b}": dict(curve) for a, b in ((0, 17), (17, 21), (21, 23), (23, 28))}
    return {"lengths": [400], "curves": {"2:400:1": {**blocks, "head": {"0": 0.0, "400": 0.0}}}}


def task_record(index, ear=0.995, outside=0, rows=None, name=None):
    rows = rows or [{"active": 10, "full_depth": 4, "masked": 3}]
    metrics = {"ear": ear, "kl": 0.0, "position_flip_rate": 0.1, "position_change_rate": 0.0, "masked_ear": 0.9,
               "answer_outside": outside, "rows": rows, "flops": 1}
    return {"task_id": f"t{index}", "index": index, "prompt_tokens": 100, "answer_end": 50, "dense_flops": 4,
            "policies": {name: metrics}}


class ScoringTests(unittest.TestCase):
    def test_predicted_time_uses_full_rows_below_depth_and_batch_maxima(self) -> None:
        tasks = [{"task_id": "a", "prompt_tokens": 100, "rows": [{"active": 10, "full_depth": 4, "masked": 3}]},
                 {"task_id": "b", "prompt_tokens": 90, "rows": [{"active": 20, "full_depth": 2, "masked": 3}]}]
        seconds, dense = predicted_seconds(table(), tasks, 2, 22)
        self.assertEqual(seconds, 4 + 4 + 20 + 20)  # two blocks below depth 22, two above
        self.assertEqual(dense, 4 * 356)

    def test_feasible_policy_has_no_violation(self) -> None:
        genome = Genome(1, 8, 22, 0)
        records = [task_record(i, name=genome.name) for i in range(4)]
        item = search.score(genome, records, settings(objective_batch_size=2), table())
        self.assertEqual(item.violation, 0.0)
        self.assertEqual(item.constraints["answer_coverage"], 1.0)
        self.assertEqual(item.compute_fraction, 0.25)

    def test_each_constraint_adds_its_shortfall(self) -> None:
        genome = Genome(1, 8, 22, 0)
        low = [task_record(i, ear=0.95, name=genome.name) for i in range(4)]
        item = search.score(genome, low, settings(objective_batch_size=2), table())
        self.assertAlmostEqual(item.violation, (0.9870 - 0.95) + (0.9758 - 0.95))
        uncovered = [task_record(i, outside=2, name=genome.name) for i in range(4)]
        item = search.score(genome, uncovered, settings(objective_batch_size=2), table())
        self.assertAlmostEqual(item.violation, 0.99 - 0.96)

    def test_audit_prefers_unvalidated_genes_and_adds_the_fastest(self) -> None:
        fast = individual(Genome(1, 8, 22, 0, 4), 1.0)
        padded = individual(Genome(1, 8, 22, 0, 4, padding_window=8), 2.0)
        prompt = individual(Genome(1, 8, 22, 0, prompt_refresh_interval=32), 3.0)
        bad = individual(Genome(1, 8, 22, 0, local_refresh_radius=2), 0.1, violation=0.1)
        self.assertEqual(search.audit_choices([prompt, fast, bad, padded], 1), [padded, fast])


class DecisionTests(unittest.TestCase):
    def test_tau_verdicts(self) -> None:
        names = ["a", "b", "c", "x"]
        within = {"a": True, "b": True, "c": True, "x": False}
        self.assertEqual(tau_decide(names, {"a": True, "b": True, "c": False, "x": False}, within)["verdict"], "CONFIRMED")
        self.assertEqual(tau_decide(names, {"a": True, "b": True, "c": True, "x": True}, within)["verdict"], "UNSAFE")
        self.assertEqual(tau_decide(names, {"a": True, "b": False, "c": False, "x": False}, within)["verdict"], "TOO_STRICT")

    def test_feasibility_uses_the_lower_bound_tail_and_coverage(self) -> None:
        limits = SimpleNamespace(min_ear=0.987, min_tail_ear=0.97, min_answer_coverage=0.99)
        self.assertTrue(feasibility(np.full(96, 0.99), 1.0, limits)["feasible"])
        self.assertFalse(feasibility(np.full(96, 0.99), 0.98, limits)["feasible"])
        tail = np.full(96, 0.995)
        tail[:20] = 0.96
        self.assertFalse(feasibility(tail, 1.0, limits)["feasible"])

    def test_ranking_compares_speedups_over_dense(self) -> None:
        predicted = {"a": 2.0, "b": 3.0, "c": 4.0}
        self.assertTrue(ranking(predicted, {"dense": 12.0, "a": 8.0, "b": 6.0, "c": 3.0})["passed"])
        self.assertFalse(ranking(predicted, {"dense": 12.0, "a": 3.0, "b": 6.0, "c": 8.0})["passed"])

    def row(self, name, speed, within=True, seed=False):
        return {"policy": name, "measured_speedup": speed, "within_margin": within, "seed": seed}

    def test_success_needs_faithful_fast_and_ahead_of_the_best_seed(self) -> None:
        seed = self.row("seed", 2.0, seed=True)
        self.assertEqual(search.success([seed, self.row("good", 2.3)])["winners"], ["good"])
        self.assertFalse(search.success([seed, self.row("near", 2.1)])["success"])  # under 10% ahead
        self.assertFalse(search.success([seed, self.row("lossy", 3.0, within=False)])["success"])
        self.assertFalse(search.success([self.row("slow", 1.8)])["success"])
        self.assertFalse(search.success([seed])["success"])  # a seed alone is not a searched policy


class TaskSetTests(unittest.TestCase):
    def test_sets_are_disjoint_equal_and_balanced_by_length(self) -> None:
        items = [f"t{index:03d}" for index in range(100)]
        lengths = list(range(100))
        sets = deal(items, lengths, 28, random.Random(0))
        self.assertEqual([len(chosen) for chosen in sets], [28, 28, 28])
        self.assertEqual(len({item for chosen in sets for item in chosen}), 84)
        means = [np.mean([lengths[items.index(item)] for item in chosen]) for chosen in sets]
        self.assertLess(max(means) - min(means), 6)

    def test_too_few_tasks_is_an_error(self) -> None:
        with self.assertRaises(ValueError):
            deal(["a", "b"], [1, 2], 2, random.Random(0))


class AdvanceTests(unittest.TestCase):
    def test_generations_advance_and_rerunning_is_safe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "search"
            cost = Path(directory) / "table.json"
            cost.write_text(json.dumps(table()))
            evolution = settings(objective_batch_size=2, cost_table=str(cost))
            config = SimpleNamespace(evolution=evolution, evaluation=SimpleNamespace(max_prompts=2))
            with mock.patch.object(search, "load_kv_policy_experiment_config", return_value=config):
                search.advance(Path("c"), output)
                self.assertEqual(len(search.read_genomes(search.generation_dir(output, 0))), 6)
                with self.assertRaises(SystemExit):
                    search.advance(Path("c"), output)  # no tasks yet
                for generation in range(2):
                    folder = search.generation_dir(output, generation)
                    names = [genome.name for genome in search.read_genomes(folder)]
                    (folder / "tasks").mkdir()
                    for index in range(2):
                        record = task_record(index, name=names[0])
                        record["policies"] = {name: record["policies"][names[0]] for name in names}
                        (folder / "tasks" / f"task_{index:03d}.json").write_text(json.dumps(record))
                    search.advance(Path("c"), output)
                search.advance(Path("c"), output)
            first = {tuple(g.to_dict().items()) for g in search.read_genomes(search.generation_dir(output, 0))}
            second = {tuple(g.to_dict().items()) for g in search.read_genomes(search.generation_dir(output, 1))}
            self.assertFalse(first & second)
            summary = json.loads((search.generation_dir(output, 1) / "summary.json").read_text())
            self.assertEqual(len(summary["population"]), 6)
            self.assertFalse(search.generation_dir(output, 2).exists())


if __name__ == "__main__":
    unittest.main()
