import unittest
from unittest import mock
from pathlib import Path
from tempfile import TemporaryDirectory

from dlm_kv_policy.kv_policy_experiment_config import KVPolicyNSGA2SearchConfig, load_kv_policy_experiment_config
from dlm_kv_policy.nsga2_kv_policy_search import (
    EvaluatedKVPolicy,
    KVRefreshPolicyGenome,
    environmental_selection,
    initial_population,
    non_dominated_fronts,
    offspring_population,
    run_kv_policy_nsga2_search,
)


def individual(genome: KVRefreshPolicyGenome, quality_loss: float, compute_fraction: float) -> EvaluatedKVPolicy:
    return EvaluatedKVPolicy(genome, quality_loss=quality_loss, compute_fraction=compute_fraction)


class EvolutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = KVPolicyNSGA2SearchConfig(population_size=4, seed=7)

    def test_initial_population_is_unique_and_contains_fixed_controls(self) -> None:
        population = initial_population(self.config)
        self.assertEqual(len(population), 4)
        self.assertEqual(len(set(population)), 4)
        self.assertIn(KVRefreshPolicyGenome(1, 8, 1, 0), population)
        self.assertIn(KVRefreshPolicyGenome(1, 16, 1, 0), population)
        self.assertIn(KVRefreshPolicyGenome(1, 8, 22, 0), population)

    def test_nsga_selection_keeps_non_dominated_extremes(self) -> None:
        items = [
            individual(KVRefreshPolicyGenome(1, 8, 1, 0), 0.0, 100.0),
            individual(KVRefreshPolicyGenome(1, 8, 18, 0), 0.1, 80.0),
            individual(KVRefreshPolicyGenome(1, 8, 20, 0), 0.2, 120.0),
        ]
        self.assertEqual(non_dominated_fronts(items)[0], [0, 1])
        selected = environmental_selection(items, 2)
        self.assertEqual(set(selected), {items[0], items[1]})

    def test_offspring_are_new_valid_genomes(self) -> None:
        parents = [
            individual(genome, 0.0, 100.0 + index)
            for index, genome in enumerate(initial_population(self.config))
        ]
        children = offspring_population(parents, self.config, generation=1)
        self.assertEqual(len(children), self.config.population_size)
        self.assertFalse(set(children) & {parent.genome for parent in parents})
        for child in children:
            self.assertIn(child.delay, self.config.delays)
            self.assertIn(child.refresh_interval, self.config.refresh_intervals)
            self.assertIn(child.refresh_depth, self.config.refresh_depths)
            self.assertIn(child.terminal_refresh, self.config.terminal_refresh_steps)

    def test_runner_checkpoints_and_resumes_one_population_at_a_time(self) -> None:
        class FakeEvaluator:
            signature = "cpu-test"

            def prepare_evolution_baseline(self, reuse_baseline):
                del reuse_baseline
                baseline = type("Baseline", (), {"record": {"median_elapsed_seconds": 1000.0}})()
                execution = {"aggregate": {"base_pass_at_1": 0.7}}
                return baseline, execution, {"median_elapsed_seconds": 1000.0}

            def evaluate_evolution_policies(self, policies, *, baseline, baseline_execution):
                del baseline, baseline_execution
                records = []
                for policy in policies:
                    records.append(
                        {
                            "quality_loss": 0.0 if policy.refresh_depth == 22 else 0.1,
                            "median_elapsed_seconds": 1000.0 - 10.0 * policy.cache_reloading_step,
                            "compute_fraction": 1.0 - 0.01 * policy.cache_reloading_step,
                            "candidate_base_pass_at_1": 0.7,
                            "syntax_invalid_count": 0,
                            "output_hash": policy.name,
                        }
                    )
                return records

        root = Path(__file__).resolve().parents[1]
        config = load_kv_policy_experiment_config(root / "configs" / "diffucoder7b_dkv_decode_nsga2_search_64tasks_8populations.yaml")
        config.evolution.population_size = 4
        config.evolution.generations = 2
        with TemporaryDirectory() as directory:
            config.run.output_dir = directory
            config.run.name = "resume"
            first = run_kv_policy_nsga2_search(
                config, FakeEvaluator(), reuse_baseline=None, generations_this_job=1
            )
            second = run_kv_policy_nsga2_search(
                config, FakeEvaluator(), reuse_baseline=None, generations_this_job=1
            )
            self.assertEqual(first["completed_generations"], 1)
            self.assertTrue(second["complete"])
            self.assertTrue((Path(directory) / "resume" / "evolution" / "generations" / "generation_001.json").is_file())

    def test_replay_objective_scores_a_generation_without_evalplus(self) -> None:
        def fake_replay(model, tokenizer, tasks, policies, generation, device, task_dir):
            # Longer refresh intervals: cheaper but more divergent.
            rows = {
                name: [{"token_tv": policy.cache_reloading_step / 4, "kl": 0.0, "kl_masked_mean": 0.0,
                        "top1_flip": False, "position_flip": False, "dense_margin": 1.0,
                        "dense_position_margin": 0.0}]
                for name, policy in policies.items()
            }
            flops = {name: 100 - policy.cache_reloading_step for name, policy in policies.items()}
            return [{"dense_flops": 100, "flops": flops, "policies": rows}]

        class FakeEvaluator:
            signature = "cpu-test"
            model = tokenizer = None
            tasks: list = []

            def prepare_evolution_baseline(self, reuse_baseline):
                raise AssertionError("The replay objective needs no dense baseline")

        root = Path(__file__).resolve().parents[1]
        config = load_kv_policy_experiment_config(root / "configs" / "diffucoder7b_dkv_decode_nsga2_search_64tasks_8populations.yaml")
        config.evolution.quality_objective = "replay_divergence"
        config.evolution.population_size = 4
        config.evolution.generations = 2
        evaluator = FakeEvaluator()
        evaluator.config = config
        with TemporaryDirectory() as directory, mock.patch(
            "dlm_kv_policy.nsga2_kv_policy_search.replay_policies", fake_replay
        ):
            config.run.output_dir = directory
            config.run.name = "replay"
            result = run_kv_policy_nsga2_search(config, evaluator, reuse_baseline=None, generations_this_job=2)
        self.assertTrue(result["complete"])
        pareto = result["last_generation"]["pareto"]
        self.assertEqual(result["last_generation"]["quality_objective"], "replay_divergence")
        self.assertTrue(all(item["objectives"]["quality_loss"] > 0 for item in pareto))


if __name__ == "__main__":
    unittest.main()
