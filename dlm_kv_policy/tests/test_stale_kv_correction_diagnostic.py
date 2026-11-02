import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import torch

from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.stale_kv_correction_diagnostic import DiagnosedReplay, empty_sums, report
from dlm_kv_policy.stale_kv_error_study import COMBOS, POLICIES, row_types
from test_stale_kv_error_study import RotatingFakeModel, fake_replay
from test_replay_compounding import greedy_sample_tokens

# The sampler looks up sample_tokens in the module that defines the model class.
sample_tokens = greedy_sample_tokens


def identity_fit(layers=2, heads=1, head_dim=4):
    """e_hat = c."""

    return {
        "m4_matrix": torch.eye(head_dim).expand(layers, 2, heads, COMBOS, head_dim, head_dim).clone(),
        "m4_intercept": torch.zeros(layers, 2, heads, COMBOS, head_dim),
    }


class DiagnosedReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = RotatingFakeModel().eval()
        self.generation = fake_replay(self.model, {}).generation
        self.prompt = torch.tensor([[4, 5, 6]])

    def diagnose(self, policy):
        dense = DiffuCoderActiveTokenKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        replay = DiagnosedReplay(self.model, dense, {"p": policy}, {"p": identity_fit()}, {0}, self.generation, None)
        result = replay.run(self.prompt, torch.ones_like(self.prompt))
        return replay, result

    def test_uncorrected_kv_error_equals_the_study_measurement(self) -> None:
        policy = DKVDecodeRefreshPolicy(cache_reloading_step=4)
        replay, result = self.diagnose(policy)
        study = fake_replay(self.model, {"p": policy})
        expected = torch.zeros(2, 2, dtype=torch.float64)

        def sink(name, layer, kind, c, e, combo):
            expected[layer, kind] += e.square().sum()

        row_type = row_types(3, result["dense_token_ids"], {0}, "cpu")
        study.run(self.prompt, torch.ones_like(self.prompt), row_type, sink)
        uncorrected = replay.sums["p@0.0"]
        self.assertGreater(expected[1].sum().item(), 0)
        torch.testing.assert_close(uncorrected["kv_before"][1:], expected[1:])  # layer 0 is never corrected
        torch.testing.assert_close(uncorrected["kv_after"], uncorrected["kv_before"])
        self.assertFalse(torch.allclose(replay.sums["p@0.5"]["kv_after"][1:], replay.sums["p@0.5"]["kv_before"][1:]))
        self.assertTrue(bool((uncorrected["attention_error"][1:] > 0).all()))

    def test_no_stale_rows_means_no_error(self) -> None:
        replay, result = self.diagnose(DKVDecodeRefreshPolicy(cache_reloading_step=1))
        steps, length = self.generation.steps, 3 + self.generation.gen_length
        for sums in replay.sums.values():
            self.assertEqual(sums["kv_before"].abs().sum().item(), 0)
            self.assertEqual(sums["attention_error"].abs().sum().item(), 0)
            self.assertEqual(sums["attention_rows"].tolist(), [steps * length] * 2)


class ReportTests(unittest.TestCase):
    def test_ratios_and_explanation(self) -> None:
        def sums(kv, attention):
            values = {key: tensor.tolist() for key, tensor in empty_sums(3, "cpu").items()}
            values["kv_before"] = [[1.0, 1.0]] * 3
            values["kv_after"] = [[kv, kv]] * 3
            values["kv_dense"] = [[10.0, 10.0]] * 3
            values["attention_error"] = [attention] * 3
            values["attention_dense"] = [10.0] * 3
            return values

        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "tasks").mkdir()
            for index in range(4):
                record = {"matches_pilot": True, "sums": {}}
                for name in POLICIES:
                    # K/V closer to dense at both strengths; attention output only at 0.5.
                    record["sums"][f"{name}@0.0"] = sums(1.0 + 0.1 * index, 1.0 + 0.1 * index)
                    record["sums"][f"{name}@0.5"] = sums(0.5 * (1.0 + 0.1 * index), 0.5 * (1.0 + 0.1 * index))
                    record["sums"][f"{name}@1.0"] = sums(0.5 * (1.0 + 0.1 * index), 1.0 + 0.1 * index)
                (output_dir / "tasks" / f"task_{index:03d}.json").write_text(json.dumps(record))
            with redirect_stdout(io.StringIO()):
                summary = report(output_dir, 4)
        entry = summary["comparisons"]["refresh_256@0.5"]
        self.assertAlmostEqual(entry["kv"]["ratio"], 0.5)
        self.assertAlmostEqual(entry["fit"]["ratio"], 0.5 * 4.6 / 4)
        self.assertEqual(summary["explanation"], {0.5: "residual", 1.0: "not_decision_relevant"})
        self.assertTrue(summary["pipeline_passed"])


if __name__ == "__main__":
    unittest.main()
