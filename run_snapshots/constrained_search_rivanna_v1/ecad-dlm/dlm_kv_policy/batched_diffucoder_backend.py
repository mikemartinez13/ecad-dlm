"""Batched orchestration of the existing DiffuCoder active-row layer forward.

Supports DKVDecodeRefreshPolicy and its MaskedRowReusePolicy wrapper. Policies
return [B,L] active/depth tensors. Persistent boundaries are declared at creation;
changing policy requires a new backend. No transformer arithmetic is duplicated.
"""
from __future__ import annotations
import torch
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend, ActiveTokenForwardResult
from .batched_active_token_cache import BatchedActiveTokenKVCache
from .batched_row_index import pack_rows
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from .masked_row_reuse_policy import MaskedRowReusePolicy


class BatchedDiffuCoderKVBackend(DiffuCoderActiveTokenKVBackend):
    supports_batch = True

    def __init__(self, model, policy):
        super().__init__(model, policy)
        self.reuse = isinstance(policy, MaskedRowReusePolicy)
        base = policy.base if self.reuse else policy
        if not isinstance(base, DKVDecodeRefreshPolicy):
            raise TypeError("Batched backend currently supports DKVDecodeRefreshPolicy and masked reuse")
        policy.validate()
        self.depth = min(base.refresh_depth, self.num_layers)
        self.cache = BatchedActiveTokenKVCache(
            num_layers=self.num_layers, hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            device=next(model.parameters()).device, dtype=next(model.parameters()).dtype,
            retained_boundaries=(() if self.depth == 1 else (self.depth - 1,)),
        )

    def set_policy(self, policy):
        raise ValueError("Construct a new batched backend when changing policy")

    def start_generation(self, sequence_length, *, batch_size=1, collect_diagnostics=False):
        self.cache.reset(sequence_length, batch_size)
        # Legacy observer diagnostics require 1-D positions. Batched diagnostics
        # record row decisions separately, outside the numerical forward.
        self.collect_diagnostics = False
        self.record_rows = collect_diagnostics
        self.row_history = []
        if self.reuse:
            self.policy.reset(sequence_length, self.cache.device, batch_size=batch_size)

    @torch.no_grad()
    def forward_mask_logits(self, *, input_ids, attention_mask, position_ids, state, step, total_steps):
        if self.cache.shape != tuple(input_ids.shape):
            self.start_generation(input_ids.shape[1], batch_size=input_ids.shape[0])
        self._step, self._input_ids = step, input_ids
        states = state.classify(input_ids, step)
        decision = self.policy.decide(
            states=states, decoded_age=state.decoded_age(step, input_ids.device),
            token_cache_age=None, cache_ready=self.cache.ready, step=step,
            total_steps=total_steps, num_layers=self.num_layers, collect_counts=self.record_rows,
        )
        embedded = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(embedded, position_ids)
        # Key-only mask avoids fully masked dummy-query rows. Real query semantics
        # are identical to the sampler's pairwise mask. Unpadded B=1 keeps None.
        if isinstance(attention_mask, torch.Tensor):
            valid = state.valid_mask.to(input_ids.device)
            attention_mask = valid[:, None, None, :].expand(-1, 1, input_ids.shape[1], -1)
        groups = {1: pack_rows(decision.active & decision.refresh_depth.le(1))}
        if self.depth > 1:
            groups[self.depth] = pack_rows(decision.active)
        for layer_id, layer in enumerate(self.layers):
            positions = groups[1 if layer_id + 1 < self.depth else self.depth]
            if positions.shape[1]:
                self._layer_forward(layer=layer, layer_id=layer_id, positions=positions,
                    embedded=embedded, cos=cos, sin=sin, attention_mask=attention_mask,
                    states=states, cache_age=None)
        self.cache.mark_refreshed(decision.active, step)
        masked = states.eq(2)
        positions = pack_rows(masked)
        predecessors = (positions - 1).clamp_min(0)
        logits = self.model.lm_head(self.cache.final(predecessors))
        logits = logits[positions.ge(0)]  # request-major, same ordering as x[mask_index]
        result = ActiveTokenForwardResult(logits, decision)
        if self.reuse:
            self.policy.observe(logits)
            result.candidates = self.policy.protected[masked]
        if self.record_rows:
            self.row_history.append({"step": step, "active": decision.active.sum(-1).cpu().tolist(),
                "full_depth": (decision.active & decision.refresh_depth.le(1)).sum(-1).cpu().tolist(),
                "padded_rows": {str(k): v.shape[1] for k, v in groups.items()}})
        return result

    def diagnostics_dict(self):
        return {"backend": "batched_active_rows", "rows": self.row_history,
                "retained_boundaries": sorted(self.cache.retained_boundaries),
                "count_syncs_per_step": (3 if self.depth > 1 else 2),
                "note": "row packing uses item(); valid-only scatter/boolean indexing may also synchronize"}
