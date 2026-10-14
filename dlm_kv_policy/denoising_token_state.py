"""Token-state bookkeeping for DiffuCoder's monotonic mask-to-token sampler."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class DenoisingTokenCategory(IntEnum):
    PAD = 0
    PROMPT = 1
    MASK = 2
    NEWLY_DECODED = 3
    STABLE_DECODED = 4


@dataclass
class DenoisingTokenState:
    """State immediately before one denoising forward, for batch size one."""

    prompt_mask: torch.Tensor
    generated_mask: torch.Tensor
    decode_step: torch.Tensor
    mask_token_id: int

    @classmethod
    def create(
        cls,
        input_width: int,
        full_width: int,
        attention_mask: torch.Tensor | None,
        mask_token_id: int,
    ) -> "DenoisingTokenState":
        if attention_mask is None:
            prompt = torch.ones((1, input_width), dtype=torch.bool)
        else:
            if attention_mask.shape != (1, input_width):
                raise ValueError("DenoisingTokenState requires batch_size=1 attention masks")
            prompt = attention_mask.to(dtype=torch.bool, device="cpu")
        prompt_mask = torch.zeros((1, full_width), dtype=torch.bool)
        prompt_mask[:, :input_width] = prompt
        generated_mask = torch.zeros_like(prompt_mask)
        generated_mask[:, input_width:] = True
        return cls(
            prompt_mask=prompt_mask,
            generated_mask=generated_mask,
            decode_step=torch.full((1, full_width), -1, dtype=torch.long),
            mask_token_id=mask_token_id,
        )

    @property
    def valid_mask(self) -> torch.Tensor:
        return self.prompt_mask | self.generated_mask

    def classify(self, token_ids: torch.Tensor, step: int) -> torch.Tensor:
        """Return :class:`DenoisingTokenCategory` values on the same device as token_ids."""

        if token_ids.shape != self.prompt_mask.shape:
            raise ValueError("Token IDs do not match DenoisingTokenState shape")
        result = torch.full_like(token_ids, int(DenoisingTokenCategory.PAD))
        prompt = self.prompt_mask.to(token_ids.device)
        generated = self.generated_mask.to(token_ids.device)
        decoded_at = self.decode_step.to(token_ids.device)
        masked = generated & token_ids.eq(self.mask_token_id)
        decoded = generated & ~masked
        result[prompt] = int(DenoisingTokenCategory.PROMPT)
        result[masked] = int(DenoisingTokenCategory.MASK)
        result[decoded & decoded_at.eq(step - 1)] = int(DenoisingTokenCategory.NEWLY_DECODED)
        result[decoded & decoded_at.lt(step - 1)] = int(DenoisingTokenCategory.STABLE_DECODED)
        return result

    def decoded_age(self, step: int, device: torch.device) -> torch.Tensor:
        """Denoising forwards elapsed since each generated token was committed."""

        committed = self.decode_step.to(device)
        return torch.where(committed.ge(0), step - committed, torch.full_like(committed, -1))

    def update_after_sampling(
        self, previous_tokens: torch.Tensor, next_tokens: torch.Tensor, step: int
    ) -> None:
        """Record positions that changed from MASK to a committed token."""

        if previous_tokens.shape != next_tokens.shape:
            raise ValueError("Sampler token tensors have different shapes")
        generated = self.generated_mask.to(previous_tokens.device)
        newly = (
            generated
            & previous_tokens.eq(self.mask_token_id)
            & ~next_tokens.eq(self.mask_token_id)
        )
        decode_step = self.decode_step.to(previous_tokens.device)
        decode_step[newly] = step
        self.decode_step = decode_step.cpu()
