import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.forced_trajectory_replay import expected_acceptance, replay_task
from experiments.ear_validity import ALL_POLICIES, REGISTERED, leave_one_out, pass_fraction, task_metrics, verdict
from test_backend_cpu import FakeModel
from test_replay_compounding import greedy_sample_tokens

# The sampler looks up sample_tokens in the module that defines the model class.
sample_tokens = greedy_sample_tokens


def log_probs(*rows):
    return torch.tensor(rows, dtype=torch.float64).log()


class ExpectedAcceptanceTests(unittest.TestCase):
    def test_is_one_minus_total_variation_when_dense_has_at_most_ten_tokens(self) -> None:
        torch.manual_seed(0)
        p = torch.zeros(4, 50, dtype=torch.float64)
        p[:, :10] = torch.softmax(torch.randn(4, 10, dtype=torch.float64), dim=-1)
        q = torch.softmax(torch.randn(4, 50, dtype=torch.float64), dim=-1)
        expected = 1 - 0.5 * (p - q).abs().sum(dim=-1)
        torch.testing.assert_close(expected_acceptance(p.log(), q.log()), expected)

    def test_identical_and_disjoint(self) -> None:
        p = torch.softmax(torch.randn(3, 40, dtype=torch.float64), dim=-1)
        top = p.topk(10, dim=-1).values.sum(dim=-1)
        torch.testing.assert_close(expected_acceptance(p.log(), p.log()), top)
        one_hot = F.one_hot(torch.tensor([0]), 12).double()
        other = F.one_hot(torch.tensor([1]), 12).double()
        self.assertEqual(expected_acceptance(one_hot.log(), other.log()).item(), 0.0)

    def test_tail_outside_the_top_ten_is_not_renormalized(self) -> None:
        p = torch.full((1, 20), 0.05, dtype=torch.float64)
        self.assertAlmostEqual(expected_acceptance(p.log(), p.log()).item(), 0.5)


class ReplayLoggingTests(unittest.TestCase):
    def test_replay_logs_ear_at_each_committed_position(self) -> None:
        torch.manual_seed(1)
        model = FakeModel().eval()
        for parameter in model.parameters():
            parameter.data.mul_(3)
        reference = DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        sampler = object.__new__(DiffuCoderDenoisingSampler)
        sampler.backend, sampler.model, sampler.sample_tokens = reference, model, greedy_sample_tokens
        generation = SimpleNamespace(steps=12, gen_length=12, temperature=0.0, top_p=1.0, top_k=None,
                                     algorithm="entropy", algorithm_temperature=0.0)
        prompt = torch.tensor([[4, 5, 6]])
        backends = {"same": DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=1)),
                    "stale": DiffuCoderActiveTokenKVBackend(model, DKVDecodeRefreshPolicy(cache_reloading_step=256))}
        result = replay_task(model, sampler, backends, prompt, torch.ones_like(prompt), generation, None, answer_end=6)
        same, stale = result["policies"]["same"], result["policies"]["stale"]
        self.assertEqual(sorted(p for row in same for p in row["committed_positions"]), list(range(12)))
        self.assertTrue(all(len(row["ear"]) == len(row["committed_positions"]) for row in same))
        self.assertGreater(min(e for row in same for e in row["ear"]), 0.0)
        self.assertLessEqual(max(e for row in same for e in row["ear"]), 1.0 + 1e-5)
        same_mean = task_metrics(same, 6)["masked_ear"]
        self.assertLess(task_metrics(stale, 6)["masked_ear"], same_mean)


def row(ear, positions, flip=False, margin=1.0, masked=None, kl=0.0):
    return {"ear": ear, "committed_positions": positions, "position_flip": flip, "dense_position_margin": margin,
            "ear_masked_answer": masked, "kl": kl}


class ReportTests(unittest.TestCase):
    def test_only_answer_commits_count(self) -> None:
        rows = [row([0.5], [1], flip=True, masked=0.4), row([0.9], [2]), row([0.1], [9], flip=True), row([0.2], [8], flip=True)]
        metrics = task_metrics(rows, 5)
        self.assertAlmostEqual(metrics["ear"], 0.7)
        self.assertAlmostEqual(metrics["position_flip_rate"], 0.5)
        self.assertAlmostEqual(metrics["masked_ear"], 0.4)

    def test_a_step_committing_answer_and_padding_counts_only_its_answer_position(self) -> None:
        self.assertAlmostEqual(task_metrics([row([0.6, 0.1], [1, 7])], 5)["ear"], 0.6)

    def test_task_with_no_answer_has_no_ear(self) -> None:
        self.assertTrue(task_metrics([row([0.9], [0])], 0)["ear"] != task_metrics([row([0.9], [0])], 0)["ear"])

    def test_separated_panel_is_valid_with_one_rejected_policy(self) -> None:
        names = ["a", "b", "c", "x", "y"]
        ear = {"a": 0.99, "b": 0.98, "c": 0.97, "x": 0.90, "y": 0.80}
        within = {"a": True, "b": True, "c": True, "x": False, "y": False}
        result = leave_one_out(names, ear, within)
        self.assertEqual(result["false_safe"], [])
        self.assertEqual(result["false_reject"], ["c"])  # the lowest within-margin policy, held out
        self.assertTrue(result["passed"])

    def test_outside_policy_above_the_threshold_is_false_safe(self) -> None:
        names = ["a", "b", "x"]
        ear = {"a": 0.99, "b": 0.95, "x": 0.97}
        within = {"a": True, "b": True, "x": False}
        result = leave_one_out(names, ear, within)
        self.assertEqual(result["false_safe"], ["x"])
        self.assertFalse(result["passed"])

    def test_position_flips_can_rescue_the_verdict(self) -> None:
        names = ["a", "b", "x"]
        ear = {"a": 0.99, "b": 0.95, "x": 0.97}
        within = {"a": True, "b": True, "x": False}
        self.assertEqual(verdict(names, ear, within, {"a": 0.1, "b": 0.1, "x": 0.5})["verdict"], "valid_with_position_flips")
        self.assertEqual(verdict(names, ear, within, {"a": 0.1, "b": 0.1, "x": 0.1})["verdict"], "not_valid")

    def test_pass_fraction_pools_base_and_plus_tests(self) -> None:
        self.assertAlmostEqual(pass_fraction({"base_passed": 5, "base_total": 10, "plus_passed": 15, "plus_total": 30}), 0.5)

    def test_registered_panel_is_inside_the_replayed_panel(self) -> None:
        self.assertLessEqual(set(REGISTERED), set(ALL_POLICIES))


if __name__ == "__main__":
    unittest.main()
