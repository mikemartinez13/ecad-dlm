"""DiffuCoder's native MaskGIT-style sampler, routed through the active backend."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from .denoising_token_state import DenoisingTokenState


@dataclass
class DenoisingGenerationResult:
    sequences: torch.Tensor
    elapsed_seconds: float
    history: list[torch.Tensor]
    diagnostics: dict[str, Any]


def _sample_tokens(model: torch.nn.Module):
    module_name = model.__class__.__module__.replace("modeling_dream", "generation_utils")
    module = importlib.import_module(module_name)
    return module.sample_tokens


class DiffuCoderDenoisingSampler:
    """Keeps DiffuCoder sampling semantics while requesting only masked logits."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend
        self.model = backend.model
        self.sample_tokens = _sample_tokens(self.model)

    @staticmethod
    def _attention_inputs(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        gen_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor | str, torch.Tensor]:
        full_length = input_ids.shape[1] + gen_length
        if attention_mask is None or not bool(attention_mask.eq(0).any()):
            positions = torch.arange(full_length, device=input_ids.device).unsqueeze(0)
            return input_ids, "full", positions
        padded = F.pad(attention_mask, (0, gen_length), value=1.0)
        positions = padded.long().cumsum(-1) - 1
        positions.masked_fill_(padded.eq(0), 1)
        pairwise = torch.logical_and(
            padded.unsqueeze(1).unsqueeze(-2), padded.unsqueeze(1).unsqueeze(-1)
        )
        return input_ids, pairwise, positions

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        steps: int,
        gen_length: int,
        temperature: float,
        top_p: float,
        top_k: int | None,
        algorithm: str,
        algorithm_temperature: float,
        collect_history: bool = False,
        collect_diagnostics: bool = False,
    ) -> DenoisingGenerationResult:
        if prompt.ndim != 2 or prompt.shape[0] != 1:
            raise ValueError("The active sampler initially supports one prompt at a time")
        mask_token_id = int(self.model.config.mask_token_id)
        original_width = prompt.shape[1]
        _, model_attention_mask, position_ids = self._attention_inputs(prompt, attention_mask, gen_length)
        x = F.pad(prompt, (0, gen_length), value=mask_token_id)
        state = DenoisingTokenState.create(original_width, x.shape[1], attention_mask, mask_token_id)
        self.backend.start_generation(x.shape[1], collect_diagnostics=collect_diagnostics)
        history: list[torch.Tensor] = []
        timesteps = torch.linspace(1, 1e-12, steps + 1, device=x.device)
        torch.cuda.synchronize(x.device)
        started = perf_counter()
        for step in range(steps):
            mask_index = x.eq(mask_token_id)
            result = self.backend.forward_mask_logits(
                input_ids=x,
                attention_mask=model_attention_mask,
                position_ids=position_ids,
                state=state,
                step=step,
                total_steps=steps,
            )
            mask_logits = result.mask_logits
            t, s = timesteps[step], timesteps[step + 1]
            previous = x.clone()
            if algorithm == "origin":
                probability = 1 - s / t if step < steps - 1 else 1
                proposed = torch.full_like(x[mask_index], mask_token_id)
                selected = torch.rand(*proposed.shape, device=x.device) < probability
                _, proposed[selected] = self.sample_tokens(
                    mask_logits[selected], temperature=temperature, top_p=top_p, top_k=top_k
                )
                x[mask_index] = proposed
            else:
                if algorithm == "maskgit_plus":
                    confidence, proposed = self.sample_tokens(mask_logits, temperature=temperature, top_p=top_p, top_k=top_k)
                elif algorithm == "topk_margin":
                    confidence, proposed = self.sample_tokens(
                        mask_logits, temperature=temperature, top_p=top_p, top_k=top_k, margin_confidence=True
                    )
                elif algorithm == "entropy":
                    confidence, proposed = self.sample_tokens(
                        mask_logits, temperature, top_p=top_p, top_k=top_k, neg_entropy=True
                    )
                else:
                    raise ValueError(f"Unsupported DiffuCoder algorithm {algorithm!r}")
                masks_per_row = int(mask_index.sum().item() / x.shape[0])
                transfer = int(masks_per_row * (1 - s / t)) if step < steps - 1 else masks_per_row
                if transfer > 0:
                    full_confidence = torch.full_like(x, -torch.inf, dtype=mask_logits.dtype)
                    full_confidence[mask_index] = confidence
                    if algorithm_temperature == 0:
                        _, selected_positions = torch.topk(full_confidence, transfer)
                    else:
                        selected_positions = torch.multinomial(
                            F.softmax(full_confidence / algorithm_temperature, dim=-1), num_samples=transfer
                        )
                    proposed_full = torch.full_like(x, mask_token_id)
                    proposed_full[mask_index] = proposed
                    rows = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand_as(selected_positions)
                    x[rows, selected_positions] = proposed_full[rows, selected_positions]
            state.update_after_sampling(previous, x, step)
            if collect_history:
                history.append(x.detach().cpu().clone())
        torch.cuda.synchronize(x.device)
        return DenoisingGenerationResult(
            sequences=x,
            elapsed_seconds=perf_counter() - started,
            history=history,
            diagnostics=self.backend.diagnostics_dict(),
        )
