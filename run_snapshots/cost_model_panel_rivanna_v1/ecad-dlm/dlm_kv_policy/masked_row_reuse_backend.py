"""Active-token backend for MaskedRowReusePolicy, shared by replay and generation."""

from __future__ import annotations

from typing import Any

from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend


class MaskedRowReuseBackend(DiffuCoderActiveTokenKVBackend):
    """The active-token backend for a MaskedRowReusePolicy: marks committable rows, counts rows per step."""

    def start_generation(self, sequence_length: int, **kwargs: Any) -> None:
        super().start_generation(sequence_length, **kwargs)
        self.policy.reset(sequence_length, next(self.model.parameters()).device)
        self.active_rows: list[int] = []
        self.full_depth_rows: list[int] = []

    def forward_mask_logits(self, **inputs: Any):
        result = super().forward_mask_logits(**inputs)
        self.policy.observe(result.mask_logits)
        result.candidates = self.policy.protected[self.policy.mask_positions]
        decision = result.decision
        self.active_rows.append(int(decision.active.sum().item()))
        # Rows through every layer; the others (depth_22's decoded rows) start at layer refresh_depth.
        self.full_depth_rows.append(int((decision.active & decision.refresh_depth.le(1)).sum().item()))
        return result
