"""Dense DiffuCoder adapter used for paired generation and timing baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .denoising_token_state import DenoisingTokenState


@dataclass
class DenseForwardResult:
    mask_logits: torch.Tensor


class DenseDiffuCoderBaseline:
    """Calls DiffuCoder's normal full forward behind the new sampler interface."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

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
        states = state.classify(input_ids, step)
        return DenseForwardResult(mask_logits=logits[states.eq(2)])

    @staticmethod
    def diagnostics_dict() -> dict[str, Any]:
        return {"backend": "native_dense_forward"}
