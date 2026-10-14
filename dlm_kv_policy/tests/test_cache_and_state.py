import unittest

import torch

from dlm_kv_policy.active_token_kv_cache import ActiveTokenKVCache
from dlm_kv_policy.denoising_token_state import DenoisingTokenState, DenoisingTokenCategory


class CacheAndStateTests(unittest.TestCase):
    def test_sparse_updates_keep_other_cache_rows(self) -> None:
        cache = ActiveTokenKVCache(2, 6, 2, 3, torch.device("cpu"), torch.float32)
        cache.reset(4)
        positions = torch.tensor([1, 3])
        keys = torch.arange(12, dtype=torch.float32).view(1, 2, 2, 3)
        values = keys + 100
        cache.update_kv(0, positions, keys, values, step=7)
        stored_k, stored_v = cache.kv(0)
        self.assertTrue(torch.equal(stored_k.index_select(2, positions), keys))
        self.assertTrue(torch.equal(stored_v.index_select(2, positions), values))
        hidden = torch.arange(12, dtype=torch.float32).view(1, 2, 6)
        cache.update_boundary(1, positions, hidden)
        self.assertTrue(torch.equal(cache.boundary(1, positions), hidden))
        cache.mark_refreshed(torch.tensor([[False, True, False, True]]), step=7)
        self.assertEqual(cache.token_age(9).tolist(), [[10, 2, 10, 2]])
        self.assertEqual(cache.layer_age(0, 9).tolist(), [[10, 2, 10, 2]])

    def test_decode_step_tracks_monotonic_mask_commitment(self) -> None:
        state = DenoisingTokenState.create(2, 5, torch.ones((1, 2)), mask_token_id=99)
        previous = torch.tensor([[10, 11, 99, 99, 99]])
        updated = torch.tensor([[10, 11, 99, 4, 99]])
        state.update_after_sampling(previous, updated, step=3)
        labels = state.classify(updated, step=4)
        self.assertEqual(labels[0, 3].item(), int(DenoisingTokenCategory.NEWLY_DECODED))
        self.assertEqual(state.decoded_age(4, torch.device("cpu"))[0, 3].item(), 1)


if __name__ == "__main__":
    unittest.main()
