import unittest
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


def individual(genome: KVRefreshPolicyGenome, regression: float, wall_time: float) -> EvaluatedKVPolicy:
    return EvaluatedKVPolicy(genome, regression, wall_time, 0.0, 0)


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
                            "candidate_base_pass_at_1": 0.7,
                            "syntax_invalid_count": 0,
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


if __name__ == "__main__":
    unittest.main()
