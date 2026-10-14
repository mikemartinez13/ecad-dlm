import unittest

import torch

from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.denoising_token_state import DenoisingTokenCategory


class DkvDecodePolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = DKVDecodeRefreshPolicy(cache_reloading_step=8)
        self.states = torch.tensor(
            [[DenoisingTokenCategory.PROMPT, DenoisingTokenCategory.STABLE_DECODED, DenoisingTokenCategory.NEWLY_DECODED, DenoisingTokenCategory.MASK]]
        )

    def decide(self, *, step: int, ready: bool):
        return self.policy.decide(
            states=self.states,
            decoded_age=torch.tensor([[-1, 2, 1, -1]]),
            token_cache_age=torch.zeros_like(self.states),
            cache_ready=ready,
            step=step,
            total_steps=256,
            num_layers=28,
        )

    def test_first_two_steps_and_periodic_refresh_are_full(self) -> None:
        self.assertTrue(self.decide(step=0, ready=False).active.all())
        self.assertTrue(self.decide(step=1, ready=True).active.all())
        self.assertTrue(self.decide(step=8, ready=True).active.all())

    def test_cached_step_only_recomputes_mask_and_newly_decoded(self) -> None:
        decision = self.decide(step=2, ready=True)
        self.assertEqual(decision.active.tolist(), [[False, False, True, True]])
        self.assertEqual(decision.reasons["cached_stable"], 2)

    def test_delay_and_depth_are_explicit_ablation_coordinates(self) -> None:
        states = torch.tensor(
            [[DenoisingTokenCategory.PROMPT, DenoisingTokenCategory.MASK, DenoisingTokenCategory.NEWLY_DECODED, DenoisingTokenCategory.STABLE_DECODED]]
        )
        ages = torch.tensor([[-1, -1, 1, 2]])
        immediate = DKVDecodeRefreshPolicy(additional_decoded_forwards=0)
        long_delay = DKVDecodeRefreshPolicy(additional_decoded_forwards=2, refresh_depth=15)
        immediate_decision = immediate.decide(
            states=states, decoded_age=ages, token_cache_age=ages, cache_ready=True,
            step=2, total_steps=256, num_layers=28,
        )
        long_decision = long_delay.decide(
            states=states, decoded_age=ages, token_cache_age=ages, cache_ready=True,
            step=2, total_steps=256, num_layers=28,
        )
        self.assertEqual(immediate_decision.active.tolist(), [[False, True, False, False]])
        self.assertEqual(long_decision.active.tolist(), [[False, True, True, True]])
        self.assertEqual(long_decision.refresh_depth.tolist(), [[15, 1, 1, 15]])

    def test_depth_boundary_recomputes_stable_rows(self) -> None:
        states = torch.tensor(
            [[DenoisingTokenCategory.PROMPT, DenoisingTokenCategory.STABLE_DECODED, DenoisingTokenCategory.NEWLY_DECODED, DenoisingTokenCategory.MASK]]
        )
        policy = DKVDecodeRefreshPolicy(refresh_depth=15)
        decision = policy.decide(
            states=states,
            decoded_age=torch.tensor([[-1, 3, 1, -1]]),
            token_cache_age=torch.zeros_like(states),
            cache_ready=True,
            step=2,
            total_steps=256,
            num_layers=28,
        )
        self.assertEqual(decision.active.tolist(), [[False, True, True, True]])
        self.assertEqual(decision.refresh_depth.tolist(), [[15, 15, 1, 1]])
        self.assertEqual(decision.reasons["depth_stable"], 1)

    def test_policy_record_marks_the_depth_boundary_implementation(self) -> None:
        self.assertEqual(
            DKVDecodeRefreshPolicy().to_dict()["implementation_revision"], "depth_boundary_v2"
        )


if __name__ == "__main__":
    unittest.main()
