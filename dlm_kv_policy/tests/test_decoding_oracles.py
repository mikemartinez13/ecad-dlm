import unittest

import torch

from dlm_kv_policy.decoding_oracles import (
    MaskedRowsOnly,
    MinimalForward,
    answer_length,
    ceilings,
    interpolate,
    parallel_decoding_oracle,
)
from dlm_kv_policy.denoising_token_state import DenoisingTokenState
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from test_backend_cpu import FakeModel

MASK = 31


class ParallelDecodingOracleTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = FakeModel().eval()
        # Recomputing every row each step is exact dense on the fake model.
        self.dense = DiffuCoderActiveTokenKVBackend(self.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
        self.prompt = torch.tensor([[4, 5, 6]])

    def test_reproduces_any_target_and_commits_every_position_once(self) -> None:
        target = [7, 8, 9, 10, 11, 12]
        steps = parallel_decoding_oracle(self.dense, self.prompt, None, target, MASK)
        self.assertEqual(sum(step["committed"] for step in steps), len(target))
        self.assertEqual(steps[0]["masked"], len(target))

    def test_commits_everything_at_once_when_the_model_already_agrees(self) -> None:
        # The fake model's own first-step argmax is a target it agrees with.
        x = torch.cat([self.prompt, torch.full((1, 4), MASK)], dim=1)
        state = DenoisingTokenState.create(3, 7, None, MASK)
        self.dense.start_generation(7)
        logits = self.dense.forward_mask_logits(
            input_ids=x, attention_mask="full", position_ids=torch.arange(7)[None], state=state, step=0, total_steps=4
        ).mask_logits
        steps = parallel_decoding_oracle(self.dense, self.prompt, None, logits.argmax(dim=-1).tolist(), MASK)
        self.assertEqual(steps, [{"masked": 4, "committed": 4, "fallback": False}])


class CachingOracleTests(unittest.TestCase):
    def test_recomputes_only_masked_rows_after_the_first_step(self) -> None:
        states = torch.tensor([[1, 1, 4, 3, 2, 2]])
        first = MaskedRowsOnly().decide(states=states, cache_ready=False, step=0)
        later = MaskedRowsOnly().decide(states=states, cache_ready=True, step=3)
        self.assertEqual(first.active.tolist(), [[True] * 6])
        self.assertEqual(later.active.tolist(), [[False, False, False, False, True, True]])

    def test_minimal_forward_returns_logits_for_the_requested_rows(self) -> None:
        self.assertEqual(MinimalForward(FakeModel().eval(), 6)(2).shape, (1, 2, 32))


class CeilingArithmeticTests(unittest.TestCase):
    def test_interpolates_between_measured_points(self) -> None:
        self.assertAlmostEqual(interpolate({1: 1.0, 3: 3.0, 10: 10.0}, 2), 2.0)
        self.assertAlmostEqual(interpolate({1: 1.0, 3: 3.0}, 7), 3.0)

    def test_answer_length_stops_at_the_first_stop_token(self) -> None:
        self.assertEqual(answer_length([5, 6, 0, 7, 0], {0}), 2)
        self.assertEqual(answer_length([5, 6], {0}), 2)

    def test_ceilings_with_a_cost_proportional_to_rows(self) -> None:
        # gen_length 4, prompt 2, so L = 6; t(n) = n and native dense t = 6.
        curve = {rows: float(rows) for rows in (1, 2, 3, 4, 6)}
        timing = {"sequence_length": 6, "native_dense": 6.0, "curves": {"minimal": curve}}
        task = {
            "prompt_length": 2,
            "answer_tokens": 3,
            "parallel_oracle_steps": [
                {"masked": 4, "committed": 2, "fallback": False},
                {"masked": 2, "committed": 2, "fallback": False},
            ],
        }
        result = ceilings(timing, [task], gen_length=4)
        self.assertAlmostEqual(result["parallel_decoding"], 2.0)
        # Dense: 4 steps x 6. Caching: t(6) + t(3) + t(2) + t(1) = 12.
        self.assertAlmostEqual(result["minimal"]["caching_speed_only"], 24 / 12)
        # Combined: t(6) + t(2) = 8.
        self.assertAlmostEqual(result["minimal"]["combined_approximate"], 24 / 8)
        # Autoregressive: t(2) + 3 t(1) = 5.
        self.assertAlmostEqual(result["minimal"]["autoregressive_estimate"], 24 / 5)


if __name__ == "__main__":
    unittest.main()
