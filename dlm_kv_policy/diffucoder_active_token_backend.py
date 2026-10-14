"""Active-row DiffuCoder forward path with temporal K/V and hidden-state reuse."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from .active_token_kv_cache import ActiveTokenKVCache
from .cache_and_trajectory_diagnostics import KVCacheDiagnostics
from .fixed_token_refresh_policy import TokenRefreshDecision
from .denoising_token_state import DenoisingTokenState


def _repeat_kv(values: torch.Tensor, groups: int) -> torch.Tensor:
    if groups == 1:
        return values
    batch, heads, length, dimension = values.shape
    return values[:, :, None].expand(batch, heads, groups, length, dimension).reshape(
        batch, heads * groups, length, dimension
    )


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values[..., : values.shape[-1] // 2], values[..., values.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def _apply_rope(
    query: torch.Tensor, keys: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        keys * cos + _rotate_half(keys) * sin,
    )


def _grouped_attention(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """GQA attention without expanding DiffuCoder's 4 K/V heads to 28 heads."""

    batch, query_heads, query_length, head_dim = query.shape
    kv_heads = keys.shape[1]
    groups = query_heads // kv_heads
    grouped_query = query.reshape(batch, kv_heads, groups, query_length, head_dim)
    scores = torch.einsum("bhgqd,bhkd->bhgqk", grouped_query, keys) * (head_dim**-0.5)
    if attention_mask is not None:
        mask = attention_mask.unsqueeze(2)
        scores = scores.masked_fill(~mask, -torch.inf) if mask.dtype == torch.bool else scores + mask
    weights = torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
    output = torch.einsum("bhgqk,bhkd->bhgqd", weights, values)
    return output.reshape(batch, query_heads, query_length, head_dim)


@dataclass
class ActiveTokenForwardResult:
    mask_logits: torch.Tensor
    decision: TokenRefreshDecision


class DiffuCoderActiveTokenKVBackend:
    """A deliberately direct batch-one implementation of active-token recomputation.

    It never uses DiffuCoder's append-only ``DynamicCache``.  K/V cache entries
    are replaced at their token positions, while ``boundary_hidden[l]`` lets an
    active stable token resume at layer ``l`` rather than at the embedding.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        policy: Any,
        *,
        collect_diagnostics: bool = False,
        grouped_gqa: bool = False,
    ) -> None:
        self.model = model
        self.policy = policy
        core = getattr(model, "model", None)
        self.layers = list(getattr(core, "layers", []))
        if core is None or not self.layers or not hasattr(core, "embed_tokens"):
            raise TypeError("Expected a loaded DiffuCoder DreamModel")
        attention = self.layers[0].self_attn
        self.num_layers = len(self.layers)
        self.num_heads = int(attention.num_heads)
        self.num_kv_heads = int(attention.num_key_value_heads)
        self.head_dim = int(attention.head_dim)
        self.hidden_size = int(attention.hidden_size)
        self.groups = self.num_heads // self.num_kv_heads
        self.core = core
        self.collect_diagnostics = collect_diagnostics
        self.grouped_gqa = grouped_gqa
        self.cache = ActiveTokenKVCache(
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            device=next(model.parameters()).device,
            dtype=next(model.parameters()).dtype,
        )
        self.diagnostics = KVCacheDiagnostics()

    def set_policy(self, policy: Any) -> None:
        policy.validate()
        self.policy = policy

    def start_generation(
        self, sequence_length: int, *, collect_diagnostics: bool | None = None
    ) -> None:
        if collect_diagnostics is not None:
            self.collect_diagnostics = collect_diagnostics
        self.cache.reset(sequence_length)
        self.diagnostics = KVCacheDiagnostics()

    @staticmethod
    def _select_mask(
        attention_mask: torch.Tensor | str | None, positions: torch.Tensor
    ) -> torch.Tensor | None:
        if not isinstance(attention_mask, torch.Tensor):
            return None
        return attention_mask.index_select(2, positions)

    def _attention(
        self,
        layer: torch.nn.Module,
        layer_id: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        states: torch.Tensor,
        cache_age: torch.Tensor,
    ) -> torch.Tensor:
        attention = layer.self_attn
        batch, count, _ = hidden.shape
        normalized = layer.input_layernorm(hidden)
        query = attention.q_proj(normalized).view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        keys = attention.k_proj(normalized).view(batch, count, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = attention.v_proj(normalized).view(batch, count, self.num_kv_heads, self.head_dim).transpose(1, 2)
        active_cos = cos.index_select(1, positions)
        active_sin = sin.index_select(1, positions)
        query, keys = _apply_rope(query, keys, active_cos, active_sin)

        old_keys, old_values = self.cache.kv(layer_id)
        if self.collect_diagnostics and self.cache.ready:
            self.diagnostics.observe_kv(
                step=self._step,
                layer=layer_id,
                states=states,
                cache_age=self.cache.layer_age(layer_id, self._step),
                positions=positions,
                old_keys=old_keys,
                old_values=old_values,
                new_keys=keys,
                new_values=values,
            )
        self.cache.update_kv(layer_id, positions, keys, values, self._step)
        all_keys, all_values = self.cache.kv(layer_id)
        mask = self._select_mask(attention_mask, positions)
        if self.grouped_gqa:
            output = _grouped_attention(query, all_keys, all_values, mask)
        else:
            output = F.scaled_dot_product_attention(
                query,
                _repeat_kv(all_keys, self.groups),
                _repeat_kv(all_values, self.groups),
                attn_mask=mask,
                dropout_p=0.0,
                is_causal=False,
            )
        output = output.transpose(1, 2).reshape(batch, count, self.hidden_size)
        return hidden + attention.o_proj(output)

    def _layer_forward(
        self,
        *,
        layer: torch.nn.Module,
        layer_id: int,
        positions: torch.Tensor,
        embedded: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        states: torch.Tensor,
        cache_age: torch.Tensor,
    ) -> None:
        hidden = embedded.index_select(1, positions) if layer_id == 0 else self.cache.boundary(layer_id, positions)
        hidden = self._attention(
            layer, layer_id, positions, hidden, cos, sin, attention_mask, states, cache_age
        )
        residual = hidden
        hidden = residual + layer.mlp(layer.post_attention_layernorm(hidden))
        self.cache.update_boundary(layer_id + 1, positions, hidden)
        if layer_id == self.num_layers - 1:
            self.cache.update_final(positions, self.core.norm(hidden))

    @torch.no_grad()
    def forward_mask_logits(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | str | None,
        position_ids: torch.Tensor,
        state: DenoisingTokenState,
        step: int,
        total_steps: int,
    ) -> ActiveTokenForwardResult:
        """Compute logits only for the predecessor rows needed by masked tokens."""

        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DiffuCoderActiveTokenKVBackend supports batch_size=1")
        if self.cache.boundary_hidden is None or self.cache.boundary_hidden[0].shape[1] != input_ids.shape[1]:
            self.start_generation(input_ids.shape[1])
        self._step = step
        states = state.classify(input_ids, step)
        decoded_age = state.decoded_age(step, input_ids.device)
        cache_age = self.cache.token_age(step)
        decision = self.policy.decide(
            states=states,
            decoded_age=decoded_age,
            token_cache_age=cache_age,
            cache_ready=self.cache.ready,
            step=step,
            total_steps=total_steps,
            num_layers=self.num_layers,
        )
        if self.collect_diagnostics:
            self.diagnostics.record_policy(decision.reasons)
        embedded = self.core.embed_tokens(input_ids)
        self.cache.update_boundary(0, torch.arange(input_ids.shape[1], device=input_ids.device), embedded)
        cos, sin = self.core.rotary_emb(embedded, position_ids)
        valid_count = int((states.ne(0)).sum().item())
        for layer_id, layer in enumerate(self.layers):
            selected = decision.active & decision.refresh_depth.le(layer_id + 1)
            positions = torch.where(selected[0])[0]
            if self.collect_diagnostics:
                self.diagnostics.record_active(step, layer_id, int(positions.numel()), valid_count)
            if positions.numel():
                self._layer_forward(
                    layer=layer,
                    layer_id=layer_id,
                    positions=positions,
                    embedded=embedded,
                    cos=cos,
                    sin=sin,
                    attention_mask=attention_mask,
                    states=states,
                    cache_age=cache_age,
                )
        self.cache.mark_refreshed(decision.active, step)

        mask_positions = torch.where(states[0].eq(2))[0]
        prediction_positions = torch.where(mask_positions.eq(0), mask_positions, mask_positions - 1)
        unique_positions, inverse = torch.unique(prediction_positions, sorted=True, return_inverse=True)
        logits = self.model.lm_head(self.cache.final(unique_positions))[0, inverse]
        if self.collect_diagnostics:
            self.diagnostics.record_lm_head(int(unique_positions.numel()), input_ids.shape[1])
        return ActiveTokenForwardResult(mask_logits=logits, decision=decision)

    def diagnostics_dict(self) -> dict[str, Any]:
        data = self.diagnostics.to_dict()
        data["enabled"] = self.collect_diagnostics
        data["grouped_gqa_without_kv_repeat"] = self.grouped_gqa
        return data
