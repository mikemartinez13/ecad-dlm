"""Dense DiffuCoder adapter used for paired generation and timing baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .compute_cost import ModelShape, dense_forward_flops
from .denoising_token_state import DenoisingTokenState


@dataclass
class DenseForwardResult:
    mask_logits: torch.Tensor


class DenseDiffuCoderBaseline:
    """Calls DiffuCoder's normal full forward behind the new sampler interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model
        # Deterministic cost of every forward since this backend was created.
        self.shape = ModelShape.from_model(model)
        self.flops = 0

    def start_generation(self, sequence_length: int, *, collect_diagnostics: bool = False) -> None:
        del sequence_length, collect_diagnostics

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
    ) -> DenseForwardResult:
        del total_steps
        model_mask = attention_mask if isinstance(attention_mask, torch.Tensor) else None
        logits = self.model(
            input_ids=input_ids,
            attention_mask=model_mask,
            position_ids=position_ids,
            use_cache=False,
        ).logits
        logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
        self.flops += dense_forward_flops(self.shape, input_ids.shape[1])
        states = state.classify(input_ids, step)
        return DenseForwardResult(mask_logits=logits[states.eq(2)])

    @staticmethod
    def diagnostics_dict() -> dict[str, Any]:
        return {"backend": "native_dense_forward"}


class BatchedDenseDiffuCoderBaseline(DenseDiffuCoderBaseline):
    """Native decoder core, with the same masked-predecessor head as caching."""
    supports_batch = True

    def start_generation(self, sequence_length, *, batch_size=1, collect_diagnostics=False):
        pass

    @torch.no_grad()
    def forward_mask_logits(self, *, input_ids, attention_mask, position_ids, state, step, total_steps):
        from .batched_row_index import pack_rows, gather_rows
        if isinstance(attention_mask, torch.Tensor):
            attention_mask = state.valid_mask.to(input_ids.device)[:, None, None, :].expand(
                -1, 1, input_ids.shape[1], -1)
        hidden = self.model.model(input_ids=input_ids,
            attention_mask=attention_mask if isinstance(attention_mask, torch.Tensor) else None,
            position_ids=position_ids, use_cache=False, return_dict=True).last_hidden_state
        positions = pack_rows(state.classify(input_ids, step).eq(2))
        logits = self.model.lm_head(gather_rows(hidden, (positions - 1).clamp_min(0)))
        return DenseForwardResult(mask_logits=logits[positions.ge(0)])

    @staticmethod
    def diagnostics_dict():
        return {"backend": "batched_native_dense_masked_predecessor_head"}
