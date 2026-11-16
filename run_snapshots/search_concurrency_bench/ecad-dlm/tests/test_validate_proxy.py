import contextlib
import io
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest import mock

from dlm_kv_policy import validate_proxy
from dlm_kv_policy.validate_proxy import auroc, average_ranks, spearman

CONFIG = """
run: {{name: study, output_dir: {output_dir}}}
model: {{path: /unused}}
generation: {{temperature: 0.0, top_p: 1.0}}
evaluation: {{code_tasks_file: /unused/tasks.jsonl}}
fixed_policy_evaluation: {{policies: [dense]}}
policies: {{dense: {{force_dense: true}}}}
dkv_policies:
  good: {{cache_reloading_step: 2}}
  bad: {{additional_decoded_forwards: 0}}
"""


def variant(statuses: dict[str, str], tokens: list[list[int]]) -> dict:
    tasks = [{"task_id": task, "base_status": status} for task, status in statuses.items()]
    return {
        "output_hash": json.dumps(tokens),
        "generated_token_ids": tokens,
        "code_execution": {"datasets": {"humaneval": {"tasks": tasks}}},
    }


def replay_rows(changes: int) -> list[dict]:
    row = {"token_tv": 1.0, "kl": 0.1, "kl_masked_mean": 0.1, "top1_flip": True, "position_flip": False,
           "dense_margin": 0.5, "dense_position_margin": 0.0}
    quiet = {**row, "token_tv": 0.0, "kl": 0.0, "top1_flip": False}
    return [row] * changes + [quiet] * 3


class StatisticsTests(unittest.TestCase):
    def test_ties_share_the_average_rank(self) -> None:
        self.assertEqual(average_ranks([5, 1, 5, 3]), [3.5, 1.0, 3.5, 2.0])

    def test_spearman_of_monotone_values_is_one(self) -> None:
        self.assertAlmostEqual(spearman([1, 2, 3, 10], [0, 1, 1.5, 9]), 1.0)

    def test_auroc_counts_ties_as_half(self) -> None:
        self.assertEqual(auroc([3, 1, 1], [True, True, False]), 0.75)


class ValidateProxyEndToEndTests(unittest.TestCase):
    def test_a_proxy_that_tracks_failures_passes_criterion_a(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text(CONFIG.format(output_dir=root))
            folder = root / "study" / "GPU"
            (folder / "variants").mkdir(parents=True)
            (folder / "forced_trajectory_replay" / "tasks").mkdir(parents=True)
            tokens = [[1], [2]]
            all_pass = {"t0": "pass", "t1": "pass"}
            for name, record in {
                "dense": variant(all_pass, tokens),
                "dense_repeat": variant(all_pass, tokens),
                "good": variant(all_pass, tokens),
                "bad": variant({"t0": "fail", "t1": "pass"}, tokens),
            }.items():
                (folder / "variants" / f"{name}.json").write_text(json.dumps(record))
            for index, task in enumerate(("t0", "t1")):
                payload = {
                    "task_id": task,
                    "dense_token_ids": tokens[index],
                    "dense_flops": 10,
                    "flops": {"good": 8, "bad": 5},
                    "policies": {"good": replay_rows(0), "bad": replay_rows(2 if task == "t0" else 0)},
                }
                (folder / "forced_trajectory_replay" / "tasks" / f"task_{index:03d}.json").write_text(json.dumps(payload))

            argv = ["validate_proxy", "--config", str(config_path), "--gpu-folder", "GPU"]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                validate_proxy.main()
            report = json.loads((folder / "proxy_validation_report.json").read_text())

        self.assertTrue(all(report["preconditions"].values()))
        self.assertEqual(report["criterion_a"]["result"], "pass")
        self.assertEqual(report["criterion_b"]["auroc"], 1.0)
        # One failure is too few for criterion B, so the verdict cannot be PASS.
        self.assertEqual(report["criterion_b"]["result"], "inconclusive")
        self.assertEqual(report["verdict"], "INCONCLUSIVE")


if __name__ == "__main__":
    unittest.main()
