import unittest

import torch
from torch import nn

from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend, _grouped_attention, _repeat_kv
from dlm_kv_policy.fixed_token_refresh_policy import FixedTokenRefreshPolicy
from dlm_kv_policy.denoising_token_state import DenoisingTokenState


class FakeAttention(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.num_heads = 2
        self.num_key_value_heads = 1
        self.head_dim = hidden_size // self.num_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)
        self.o_proj = nn.Linear(hidden_size, hidden_size)


class FakeLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_attn = FakeAttention(hidden_size)
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = nn.Linear(hidden_size, hidden_size)


class FakeCore(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([FakeLayer(hidden_size), FakeLayer(hidden_size)])
        self.norm = nn.Identity()

    def rotary_emb(self, values: torch.Tensor, position_ids: torch.Tensor):
        del position_ids
        shape = (values.shape[0], values.shape[1], values.shape[-1] // 2)
        return torch.ones(shape, dtype=values.dtype), torch.zeros(shape, dtype=values.dtype)


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeCore(vocab_size=32, hidden_size=8)
        self.lm_head = nn.Linear(8, 32, bias=False)
        self.config = type("Config", (), {"mask_token_id": 31})()


class BackendCpuTests(unittest.TestCase):
    def test_active_backend_populates_cache_and_projects_only_needed_logits(self) -> None:
        model = FakeModel().eval()
        backend = DiffuCoderActiveTokenKVBackend(
            model, FixedTokenRefreshPolicy(name="dense", force_dense=True), collect_diagnostics=True
        )
        tokens = torch.tensor([[4, 31, 31, 31]])
        state = DenoisingTokenState.create(1, 4, torch.ones((1, 1)), mask_token_id=31)
        result = backend.forward_mask_logits(
            input_ids=tokens,
            attention_mask="full",
            position_ids=torch.arange(4).unsqueeze(0),
            state=state,
            step=0,
            total_steps=4,
        )
        self.assertEqual(result.mask_logits.shape, (3, 32))
        self.assertTrue(backend.cache.ready)
        self.assertEqual(backend.diagnostics.to_dict()["lm_head"]["active_rows"], 3)

    def test_grouped_attention_matches_repeated_kv_attention_on_cpu(self) -> None:
        torch.manual_seed(0)
        query = torch.randn(1, 2, 3, 4)
        keys = torch.randn(1, 1, 5, 4)
        values = torch.randn(1, 1, 5, 4)
        mask = torch.tensor([[[[True, True, False, True, True]]]]).expand(1, 1, 3, 5)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, _repeat_kv(keys, 2), _repeat_kv(values, 2), attn_mask=mask, dropout_p=0.0
        )
        actual = _grouped_attention(query, keys, values, mask)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_immediate_reuse_keeps_a_newly_decoded_row_inactive(self) -> None:
        model = FakeModel().eval()
        policy = FixedTokenRefreshPolicy(name="immediate", additional_decoded_forwards=0)
        backend = DiffuCoderActiveTokenKVBackend(model, policy)
        state = DenoisingTokenState.create(1, 4, torch.ones((1, 1)), mask_token_id=31)
        first = torch.tensor([[4, 31, 31, 31]])
        backend.forward_mask_logits(
            input_ids=first,
            attention_mask="full",
            position_ids=torch.arange(4).unsqueeze(0),
            state=state,
            step=0,
            total_steps=4,
        )
        second = torch.tensor([[4, 7, 31, 31]])
        state.update_after_sampling(first, second, step=0)
        result = backend.forward_mask_logits(
            input_ids=second,
            attention_mask="full",
            position_ids=torch.arange(4).unsqueeze(0),
            state=state,
            step=1,
            total_steps=4,
        )
        self.assertFalse(result.decision.active[0, 1].item())
        self.assertTrue(result.decision.active[0, 2].item())


if __name__ == "__main__":
    unittest.main()
