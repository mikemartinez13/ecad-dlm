import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

import torch

from dlm_kv_policy.code_correctness_and_latency_evaluator import (
    DiffuCoderKVPolicyEvaluator,
    output_hash,
)
from dlm_kv_policy.forced_trajectory_replay import (
    large_margin_decision_changes,
    step_divergence,
    step_metrics,
    summarize,
)
from experiments.output_variance_study import compare_to_reference


def execution(statuses: dict[str, str]) -> dict:
    """A minimal EvalPlus base result with one dataset."""

    tasks = [{"task_id": task, "base_status": status} for task, status in statuses.items()]
    passed = sum(status == "pass" for status in statuses.values())
    return {
        "scope": "base",
        "aggregate": {"base_pass_at_1": passed / len(statuses)},
        "datasets": {
            "humaneval": {
                "dataset_hash": "d",
                "subset_key": "s",
                "groundtruth_cache_key": "g",
                "tasks": tasks,
            }
        },
    }


def generation_record(code: dict[str, str], statuses: dict[str, str]) -> dict:
    return {
        "output_hash": output_hash([[len(text)] for text in code.values()]) + "".join(code.values()),
        "task_outputs": [{"task_id": task, "executable_code": text} for task, text in code.items()],
        "code_execution": execution(statuses),
    }


class ReplayMetricTests(unittest.TestCase):
    def test_identical_logits_have_zero_kl_and_no_flips(self) -> None:
        logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 3.0, 0.0]])
        confidence = torch.tensor([-0.5, -0.1])
        metrics = step_metrics(logits, logits.clone(), torch.tensor([1]), confidence)
        self.assertAlmostEqual(metrics["kl"], 0.0, places=6)
        self.assertFalse(metrics["top1_flip"])
        self.assertFalse(metrics["position_flip"])

    def test_flip_records_the_dense_margin(self) -> None:
        dense = torch.tensor([[1.0, 0.9, -5.0]])
        cached = torch.tensor([[0.9, 1.0, -5.0]])
        metrics = step_metrics(dense, cached, torch.tensor([0]), torch.tensor([0.0]))
        self.assertTrue(metrics["top1_flip"])
        self.assertGreater(metrics["kl"], 0.0)
        self.assertLess(metrics["dense_margin"], 0.1)

    def test_position_flip_when_cached_prefers_another_row(self) -> None:
        logits = torch.zeros(2, 3)
        metrics = step_metrics(logits, logits, torch.tensor([0]), torch.tensor([-1.0, -0.1]))
        self.assertTrue(metrics["position_flip"])

    def test_dense_position_margin_is_gap_to_best_other_position(self) -> None:
        logits = torch.zeros(3, 3)
        metrics = step_metrics(
            logits, logits, torch.tensor([1]), torch.tensor([-0.5, -0.1, -0.9]),
            dense_confidence=torch.tensor([-0.4, -0.1, -0.9]),
        )
        self.assertAlmostEqual(metrics["dense_position_margin"], 0.3, places=6)

    def test_tied_confidence_is_not_a_position_flip(self) -> None:
        logits = torch.zeros(2, 3)
        metrics = step_metrics(logits, logits, torch.tensor([1]), torch.tensor([-0.5, -0.5]))
        self.assertFalse(metrics["position_flip"])

    def test_greedy_divergence_counts_token_and_position_changes(self) -> None:
        dense = torch.tensor([[1.0, 0.9, -5.0], [0.0, 0.0, 0.0]])
        cached = torch.tensor([[0.9, 1.0, -5.0], [0.0, 0.0, 0.0]])
        # Dense clearly preferred row 0 (margin 0.4), so a position flip counts.
        dense_confidence = torch.tensor([-0.1, -0.5])
        token_only = step_metrics(
            dense, cached, torch.tensor([0]), torch.tensor([-0.1, -0.5]), dense_confidence=dense_confidence
        )
        both = step_metrics(
            dense, cached, torch.tensor([0]), torch.tensor([-0.5, -0.1]), dense_confidence=dense_confidence
        )
        self.assertAlmostEqual(step_divergence(token_only), 1.0)
        self.assertAlmostEqual(step_divergence(both), 2.0)

    def test_position_flip_at_a_dense_tie_is_not_counted(self) -> None:
        logits = torch.zeros(2, 3)
        metrics = step_metrics(
            logits, logits, torch.tensor([0]), torch.tensor([-0.5, -0.1]),
            dense_confidence=torch.tensor([-0.3, -0.3]),
        )
        self.assertTrue(metrics["position_flip"])
        self.assertAlmostEqual(step_divergence(metrics), 0.0)
        self.assertAlmostEqual(step_divergence(metrics, position_tie_margin=None), 1.0)

    def test_sampled_divergence_is_total_variation_of_decoding_distributions(self) -> None:
        dense = torch.tensor([[0.0, 0.0]])
        cached = torch.tensor([[0.0, 0.4 * torch.log(torch.tensor(3.0)).item()]])
        metrics = step_metrics(dense, cached, torch.tensor([0]), torch.tensor([0.0]), temperature=0.4)
        # Dense decodes (1/2, 1/2); cached decodes (1/4, 3/4) at temperature 0.4.
        self.assertAlmostEqual(step_divergence(metrics), 0.25, places=5)

    def test_large_margin_changes_ignore_near_tie_flips(self) -> None:
        rows = [
            {"top1_flip": True, "position_flip": False, "dense_margin": 0.01},
            {"top1_flip": True, "position_flip": False, "dense_margin": 0.5},
            {"top1_flip": False, "position_flip": True, "dense_margin": 0.9, "dense_position_margin": 0.5},
            {"top1_flip": False, "position_flip": True, "dense_margin": 0.9, "dense_position_margin": 0.0},
        ]
        self.assertEqual(large_margin_decision_changes(rows), 2)
        self.assertEqual(large_margin_decision_changes(rows, position_tie_margin=None), 3)

    def test_summary_counts_near_tie_flips(self) -> None:
        rows = [
            {"token_tv": 0.0, "kl": 0.0, "kl_masked_mean": 0.0, "top1_flip": False, "position_flip": False,
             "dense_margin": 0.9, "dense_position_margin": 0.0},
            {"token_tv": 1.0, "kl": 0.2, "kl_masked_mean": 0.1, "top1_flip": True, "position_flip": True,
             "dense_margin": 0.01, "dense_position_margin": 0.3},
            {"token_tv": 1.0, "kl": 0.4, "kl_masked_mean": 0.1, "top1_flip": True, "position_flip": False,
             "dense_margin": 0.5, "dense_position_margin": 0.0},
        ]
        task = {"dense_flops": 100, "flops": {"p": 40}, "policies": {"p": rows}}
        summary = summarize([task], ["p"])["p"]
        self.assertAlmostEqual(summary["compute_fraction"], 0.4)
        self.assertAlmostEqual(summary["divergence_events_per_task"], 3.0)
        self.assertAlmostEqual(summary["large_margin_decision_changes_per_task"], 2.0)
        self.assertEqual(summary["top1_flips"], 2)
        self.assertEqual(summary["top1_flips_at_near_ties"], 1)
        self.assertEqual(summary["position_flips"], 1)
        self.assertEqual(summary["position_flips_counted"], 1)
        self.assertEqual(summary["tasks_with_any_top1_flip"], 1)


class VarianceStudyTests(unittest.TestCase):
    def test_compare_counts_changed_code_and_both_flip_directions(self) -> None:
        reference = generation_record({"a": "x", "b": "y", "c": "z"}, {"a": "pass", "b": "fail", "c": "pass"})
        candidate = generation_record({"a": "x", "b": "Y", "c": "Z"}, {"a": "pass", "b": "pass", "c": "fail"})
        row = compare_to_reference(reference, candidate)
        self.assertFalse(row["identical_output"])
        self.assertEqual(row["tasks_with_changed_code"], 2)
        self.assertEqual(row["pass_to_fail"], 1)
        self.assertEqual(row["fail_to_pass"], 1)
        self.assertEqual(row["changed_task_ids"], ["b", "c"])


class OutputHashReuseTests(unittest.TestCase):
    def test_identical_output_reuses_scores_without_running_evalplus(self) -> None:
        with TemporaryDirectory() as directory:
            run_dir = Path(directory)
            evaluator = object.__new__(DiffuCoderKVPolicyEvaluator)
            evaluator.config = SimpleNamespace(run=SimpleNamespace(run_dir=run_dir))
            evaluator.signature = "sig"
            evaluator._evalplus = lambda *args, **kwargs: self.fail("EvalPlus should not run")
            scored = {
                "signature": "sig",
                "policy": {"name": "first"},
                "output_hash": output_hash([[1, 2, 3]]),
                "quality_loss": 0.25,
                "correctness": {},
                "candidate_base_pass_at_1": 0.5,
                "syntax_invalid_count": 0,
                "syntax_invalid_task_ids": [],
                "code_execution": {"aggregate": {}},
            }
            (run_dir / "policies").mkdir()
            (run_dir / "policies" / "first.json").write_text(json.dumps(scored))

            result = {"output_hash": output_hash([[1, 2, 3]]), "task_outputs": []}
            evaluator._score_code(result, "second", baseline_execution={})
            self.assertEqual(result["same_output_as"], "first")
            self.assertEqual(result["quality_loss"], 0.25)


if __name__ == "__main__":
    unittest.main()
