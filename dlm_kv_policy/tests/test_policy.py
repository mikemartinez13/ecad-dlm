import unittest

import torch

from dlm_kv_policy.fixed_token_refresh_policy import FixedTokenRefreshPolicy
from dlm_kv_policy.denoising_token_state import DenoisingTokenCategory


class PolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.states = torch.tensor(
            [[
                int(DenoisingTokenCategory.PROMPT),
                int(DenoisingTokenCategory.MASK),
                int(DenoisingTokenCategory.NEWLY_DECODED),
                int(DenoisingTokenCategory.STABLE_DECODED),
                int(DenoisingTokenCategory.PAD),
            ]]
        )
        self.decoded_age = torch.tensor([[-1, -1, 1, 5, -1]])
        self.cache_age = torch.tensor([[3, 3, 1, 5, 0]])

    def decide(self, policy: FixedTokenRefreshPolicy, *, step: int = 4, total: int = 20):
        return policy.decide(
            states=self.states,
            decoded_age=self.decoded_age,
            token_cache_age=self.cache_age,
            cache_ready=True,
            step=step,
            total_steps=total,
            num_layers=28,
        )

    def test_decoded_delay_has_unambiguous_zero_and_one_forward_behavior(self) -> None:
        immediate = self.decide(FixedTokenRefreshPolicy(name="immediate", additional_decoded_forwards=0))
        delayed = self.decide(FixedTokenRefreshPolicy(name="delay", additional_decoded_forwards=1))
        self.assertEqual(immediate.active.tolist(), [[False, True, False, False, False]])
        self.assertEqual(delayed.active.tolist(), [[False, True, True, False, False]])

    def test_masks_and_newly_decoded_rows_must_start_at_layer_one(self) -> None:
        policy = FixedTokenRefreshPolicy.from_dict(
            "depth",
            {"refresh_depth": {"default": {"prompt": 22, "stable_decoded": 15}}},
        )
        decision = self.decide(policy)
        self.assertEqual(decision.refresh_depth[0, 1].item(), 1)
        self.assertEqual(decision.refresh_depth[0, 2].item(), 1)
        self.assertEqual(decision.refresh_depth[0, 3].item(), 15)

    def test_local_window_and_terminal_refresh_are_explicit(self) -> None:
        local = self.decide(FixedTokenRefreshPolicy(name="local", local_refresh_radius=1))
        self.assertEqual(local.active.tolist(), [[False, True, True, True, False]])
        terminal = self.decide(
            FixedTokenRefreshPolicy(name="terminal", terminal_full_refresh_steps=3), step=18, total=20
        )
        self.assertEqual(terminal.active.tolist(), [[True, True, True, True, False]])
        self.assertTrue(torch.equal(terminal.refresh_depth, torch.ones_like(terminal.refresh_depth)))


if __name__ == "__main__":
    unittest.main()
