"""Batched active-row forward for DiffuCoder.

Supports DKVDecodeRefreshPolicy and its MaskedRowReusePolicy wrapper. Policies
return [B,L] active/depth tensors. Persistent boundaries are declared at creation;
changing policy requires a new backend.

The forward reads from the GPU once per step (the padded row counts). Everything
else is gathers and scatters against tensors with a spare slot for dummy rows, so
the CPU can queue all 28 layers without waiting. The layer body is therefore
written out here instead of calling the batch-one ``_layer_forward`` per layer.

Two entry points share that forward:
  * ``forward_step`` is what the sampler uses. It returns logits only for the
    masked rows it had to project, packed per request.
  * ``forward_mask_logits`` keeps the batch-one interface (logits of every masked
    row, request-major) for tests and calibration. It reads back a row mask.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from .diffucoder_active_token_backend import (
    DiffuCoderActiveTokenKVBackend, ActiveTokenForwardResult, _apply_rope, _repeat_kv)
from .batched_active_token_cache import BatchedActiveTokenKVCache
from utils.row_index import pack_rows_padded
from .denoising_token_state import DenoisingTokenCategory
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from .entropy_confidence import entropy_confidence
from .fixed_token_refresh_policy import TokenRefreshDecision
from .masked_row_reuse_policy import MaskedRowReusePolicy


@dataclass
class BatchedStepResult:
    """Logits of the masked rows projected this step.

    ``positions`` [B,n] are their sequence positions; the sequence length marks a
    dummy row. Masked rows that are absent kept their hidden state, so their
    previous logits still hold. ``committable`` [B,L] marks masked rows the policy
    allows the sampler to commit.
    """
    logits: torch.Tensor
    positions: torch.Tensor
    committable: torch.Tensor
    decision: TokenRefreshDecision


class BatchedDiffuCoderKVBackend(DiffuCoderActiveTokenKVBackend):
    supports_batch = True

    def __init__(self, model, policy, *, grouped_attention=True, lean_head=True, compile_layers=False):
        """``grouped_attention`` attends on the K/V heads without copying them per query
        head; it applies to batches, not to a single unpadded request. ``lean_head``
        projects only masked rows whose predecessor was recomputed. Both leave generated
        tokens unchanged on the audit set at batch 8; turn them off to reproduce the
        plainest arithmetic.

        ``compile_layers`` runs batched layers through one torch.compile'd function
        (compiled_layer.py): about 15% faster, but it changes some tokens and compiles
        during the first batches, so it is off unless asked for."""
        super().__init__(model, policy)
        self.reuse = isinstance(policy, MaskedRowReusePolicy)
        base = policy.base if self.reuse else policy
        if not isinstance(base, DKVDecodeRefreshPolicy):
            raise TypeError("Batched backend currently supports DKVDecodeRefreshPolicy and masked reuse")
        policy.validate()
        if self.num_heads % self.num_kv_heads:
            raise ValueError("Query heads must be a multiple of key/value heads")
        self.grouped_attention, self.lean_head = grouped_attention, lean_head
        self.depth = min(base.refresh_depth, self.num_layers)
        self.cache = BatchedActiveTokenKVCache(
            num_layers=self.num_layers, hidden_size=self.hidden_size,
            num_kv_heads=self.num_kv_heads, head_dim=self.head_dim,
            device=next(model.parameters()).device, dtype=next(model.parameters()).dtype,
            retained_boundaries=(() if self.depth == 1 else (self.depth - 1,)),
        )
        self.record_rows = False
        self.row_history = []
        self.compile_layers = compile_layers
        if compile_layers:
            if not grouped_attention:
                raise ValueError("compile_layers uses grouped attention")
            from .compiled_layer import check_layer, compiled_layer_function, layer_arguments
            self._layer_arguments = [layer_arguments(layer, self.num_heads, self.num_kv_heads) for layer in self.layers]
            check_layer(self.layers[0], self._layer_arguments[0], self._reference_layer)
            self._compiled_layer = compiled_layer_function()

    def _reference_layer(self, hidden, keys_cache, values_cache, positions, cos, sin, keymask):
        """Layer 0 through its modules against the given caches; what check_layer compares to."""
        saved = getattr(self.cache, "keys", None), getattr(self.cache, "values", None)
        self.cache.keys, self.cache.values = [keys_cache], [values_cache]
        try:
            return self._layer(self.layers[0], 0, hidden, positions, cos, sin, keymask)
        finally:
            self.cache.keys, self.cache.values = saved

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

    def _attend(self, layer_id, query, keymask):
        keys, values = self.cache.kv(layer_id)
        if keymask is None:
            # One unpadded request has no dummy rows: attend without the spare slot or a
            # mask, and with copied K/V heads. That is the arithmetic of the batch-one
            # backend, so the two stay identical at every step (the search's check stage).
            # Grouped attention rounds differently and is used only when batching.
            keys, values = keys[:, :, :-1], values[:, :, :-1]
        if keymask is None or not self.grouped_attention:
            return F.scaled_dot_product_attention(query, _repeat_kv(keys, self.groups),
                                                  _repeat_kv(values, self.groups), attn_mask=keymask)
        # Consecutive query heads share a K/V head: fold each group into one longer query.
        batch, _, count, width = query.shape
        output = F.scaled_dot_product_attention(
            query.reshape(batch, self.num_kv_heads, self.groups * count, width), keys, values, attn_mask=keymask)
        return output.reshape(batch, self.num_heads, count, width)

    def _layer(self, layer, layer_id, hidden, positions, cos, sin, keymask):
        attention = layer.self_attn
        batch, count, _ = hidden.shape
        normalized = layer.input_layernorm(hidden)
        query = attention.q_proj(normalized).view(batch, count, self.num_heads, self.head_dim).transpose(1, 2)
        keys = attention.k_proj(normalized).view(batch, count, self.num_kv_heads, self.head_dim).transpose(1, 2)
        values = attention.v_proj(normalized).view(batch, count, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query, keys = _apply_rope(query, keys, cos, sin)
        self.cache.update_kv(layer_id, positions, keys, values)
        output = self._attend(layer_id, query, keymask).transpose(1, 2).reshape(batch, count, self.hidden_size)
        hidden = hidden + attention.o_proj(output)
        return hidden + layer.mlp(layer.post_attention_layernorm(hidden))

    def _forward(self, *, input_ids, attention_mask, position_ids, state, step, total_steps, every_masked_row):
        if self.kv_observer is not None or self.kv_correction is not None or self.attention_observer is not None:
            raise NotImplementedError("K/V observers and corrections are batch-one only")
        if self.cache.shape != tuple(input_ids.shape):
            self.start_generation(input_ids.shape[1], batch_size=input_ids.shape[0])
        batch, length = input_ids.shape
        self._step, self._input_ids = step, input_ids
        states = state.classify(input_ids, step)
        decision = self.policy.decide(
            states=states, decoded_age=state.decoded_age(step, input_ids.device),
            token_cache_age=None, cache_ready=self.cache.ready, step=step,
            total_steps=total_steps, num_layers=self.num_layers, collect_counts=self.record_rows,
        )
        masked = states.eq(int(DenoisingTokenCategory.MASK))
        selected = {"full": decision.active & decision.refresh_depth.le(1)}
        if self.depth > 1:
            selected["deep"] = decision.active
        last = "deep" if self.depth > 1 else "full"
        # A masked row is predicted from its predecessor's final hidden state, which
        # only changes when the predecessor is recomputed.
        head = masked
        if self.lean_head and self.cache.ready and not every_masked_row:
            head = masked.clone()
            head[:, 1:] &= selected[last][:, :-1]
        # The only wait of the step: every padded row count in one read.
        *counts, head_count = torch.stack(
            [rows.sum(-1).max() for rows in (*selected.values(), head)]).tolist()
        embedded = self.core.embed_tokens(input_ids)
        cos, sin = self.core.rotary_emb(embedded, position_ids)
        cos, sin = cos.expand(batch, -1, -1), sin.expand(batch, -1, -1)
        # Keys only: valid positions, never the spare slot. Dummy queries then
        # attend to real keys and stay finite; their outputs go to the spare slot.
        keymask = None
        if batch > 1 or isinstance(attention_mask, torch.Tensor):
            keymask = F.pad(state.valid_mask.to(input_ids.device), (0, 1), value=False)[:, None, None, :]
        groups = {}
        for (name, rows), count in zip(selected.items(), counts):
            positions = pack_rows_padded(rows, count)
            real = positions.clamp(max=length - 1)[..., None]
            groups[name] = (positions, real, cos.gather(1, real.expand(-1, -1, cos.shape[-1])),
                            sin.gather(1, real.expand(-1, -1, sin.shape[-1])))
        hidden = None
        for layer_id, layer in enumerate(self.layers):
            name = "deep" if self.depth > 1 and layer_id + 1 >= self.depth else "full"
            positions, real, group_cos, group_sin = groups[name]
            if positions.shape[1] == 0:
                continue
            if layer_id == 0:
                hidden = embedded.gather(1, real.expand(-1, -1, self.hidden_size))
            elif self.depth > 1 and layer_id + 1 == self.depth:
                # Rows refreshed from the embedding join the rows that resume here.
                if hidden is not None:
                    self.cache.update_boundary(layer_id, groups["full"][0], hidden)
                hidden = self.cache.boundary(layer_id, positions)
            if self.compile_layers and keymask is not None:
                hidden = self._compiled_layer(hidden, *self.cache.kv(layer_id), positions, group_cos, group_sin,
                                              keymask, *self._layer_arguments[layer_id])
            else:
                hidden = self._layer(layer, layer_id, hidden, positions, group_cos, group_sin, keymask)
        if hidden is not None:
            self.cache.update_final(groups[last][0], self.core.norm(hidden))
        self.cache.mark_refreshed(decision.active, step)
        positions = pack_rows_padded(head, head_count)
        predecessors = (positions - 1).clamp(min=0, max=length - 1)
        logits = self.model.lm_head(self.cache.final(predecessors))
        if self.reuse and head_count:
            self.policy.observe_rows(positions, entropy_confidence(logits, 0.0, 1.0, None).float(),
                                     logits.argmax(dim=-1))
        if self.record_rows:
            self.row_history.append({"step": step, "active": decision.active.sum(-1).cpu().tolist(),
                "full_depth": selected["full"].sum(-1).cpu().tolist(),
                "padded_rows": {("1" if name == "full" else str(self.depth)): group[0].shape[1]
                                for name, group in groups.items()},
                "head_rows": head_count})
        committable = masked & self.policy.protected if self.reuse else masked
        return BatchedStepResult(logits, positions, committable, decision), masked

    @torch.no_grad()
    def forward_step(self, *, input_ids, attention_mask, position_ids, state, step, total_steps):
        return self._forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             state=state, step=step, total_steps=total_steps, every_masked_row=False)[0]

    @torch.no_grad()
    def forward_mask_logits(self, *, input_ids, attention_mask, position_ids, state, step, total_steps):
        step_result, masked = self._forward(input_ids=input_ids, attention_mask=attention_mask,
                                            position_ids=position_ids, state=state, step=step,
                                            total_steps=total_steps, every_masked_row=True)
        # Request-major, same ordering as x[mask_index].
        logits = step_result.logits[step_result.positions.lt(input_ids.shape[1])]
        result = ActiveTokenForwardResult(logits, step_result.decision)
        if self.reuse:
            result.candidates = self.policy.protected[masked]
        return result

    def diagnostics_dict(self):
        return {"backend": "batched_active_rows", "rows": self.row_history,
                "retained_boundaries": sorted(self.cache.retained_boundaries),
                "count_syncs_per_step": 1,
                "grouped_attention": self.grouped_attention, "lean_head": self.lean_head,
                "compile_layers": self.compile_layers,
                "note": "forward_step reads padded row counts once per step; forward_mask_logits also reads a row mask"}
