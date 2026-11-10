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
    def commit_tokens(
        self,
        x: torch.Tensor,
        mask_index: torch.Tensor,
        mask_logits: torch.Tensor,
        *,
        step: int,
        steps: int,
        timesteps: torch.Tensor,
        temperature: float,
        top_p: float,
        top_k: int | None,
        algorithm: str,
        algorithm_temperature: float,
        candidates: torch.Tensor | None = None,
        transfer_count: int | None = None,
    ) -> None:
        """Unmask this step's tokens in ``x`` in place, as DiffuCoder's sampler does.

        ``candidates``, if given, marks the rows of ``mask_logits`` that may be
        committed (see masked_row_reuse_replay.py); the others get no confidence.
        """

        mask_token_id = int(self.model.config.mask_token_id)
        t, s = timesteps[step], timesteps[step + 1]
        if algorithm == "origin":
            probability = 1 - s / t if step < steps - 1 else 1
            proposed = torch.full_like(x[mask_index], mask_token_id)
            selected = torch.rand(*proposed.shape, device=x.device) < probability
            _, proposed[selected] = self.sample_tokens(
                mask_logits[selected], temperature=temperature, top_p=top_p, top_k=top_k
            )
            x[mask_index] = proposed
            return
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
        if candidates is not None:
            confidence = confidence.masked_fill(~candidates, -torch.inf)
        if transfer_count is None:
            masks_per_row = int(mask_index.sum().item() / x.shape[0])
            transfer = int(masks_per_row * (1 - s / t)) if step < steps - 1 else masks_per_row
        else:
            transfer = transfer_count
        if transfer > 0:
            if transfer_count is not None and candidates is not None:
                torch._assert_async((candidates.reshape(x.shape[0], -1).sum(-1) >= transfer).all(),
                                    "Protected set is too small for this step's commit count")
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
        batched = getattr(self.backend, "supports_batch", False)
        if prompt.ndim != 2 or prompt.shape[0] == 0 or (prompt.shape[0] != 1 and not batched):
            raise ValueError("The active sampler initially supports one prompt at a time")
        if batched and (algorithm != "entropy" or temperature != 0 or algorithm_temperature != 0
                        or top_p != 1 or top_k is not None):
            raise ValueError("Batched decoding currently requires greedy entropy, top_p=1, top_k=None")
        if steps < 1 or gen_length < 1 or prompt.shape[1] < 1:
            raise ValueError("Positive steps, generation length and nonempty prompts are required")
        mask_token_id = int(self.model.config.mask_token_id)
        original_width = prompt.shape[1]
        _, model_attention_mask, position_ids = self._attention_inputs(prompt, attention_mask, gen_length)
        x = F.pad(prompt, (0, gen_length), value=mask_token_id)
        state = DenoisingTokenState.create(original_width, x.shape[1], attention_mask, mask_token_id,
            batch_size=prompt.shape[0], device=x.device if batched else "cpu")
        if batched:
            torch._assert_async(state.prompt_mask.any(-1).all(), "Every request needs a nonempty prompt")
        start_options = {"batch_size": prompt.shape[0]} if batched else {}
        self.backend.start_generation(x.shape[1], collect_diagnostics=collect_diagnostics, **start_options)
        history: list[torch.Tensor] = []
        timesteps = torch.linspace(1, 1e-12, steps + 1, device=x.device)
        schedule = commit_schedule(timesteps, gen_length) if batched else None
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        started = perf_counter()
        for step in range(steps):
            mask_index = x.eq(mask_token_id)
            if batched:
                mask_index &= state.generated_mask
            result = self.backend.forward_mask_logits(
                input_ids=x,
                attention_mask=model_attention_mask,
                position_ids=position_ids,
                state=state,
                step=step,
                total_steps=steps,
            )
            previous = x.clone()
            self.commit_tokens(
                x,
                mask_index,
                result.mask_logits,
                step=step,
                steps=steps,
                timesteps=timesteps,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                algorithm=algorithm,
                algorithm_temperature=algorithm_temperature,
                candidates=getattr(result, "candidates", None),
                transfer_count=schedule[step] if schedule is not None else None,
            )
            state.update_after_sampling(previous, x, step)
            if batched:
                expected = gen_length - sum(schedule[:step + 1])
                torch._assert_async(((x.eq(mask_token_id) & state.generated_mask).sum(-1) == expected).all(),
                                    "Model emitted MASK or candidate set could not satisfy commit schedule")
            if collect_history:
                history.append(x.detach().cpu().clone())
        if x.is_cuda:
            torch.cuda.synchronize(x.device)
        return DenoisingGenerationResult(
            sequences=x,
            elapsed_seconds=perf_counter() - started,
            history=history,
            diagnostics=self.backend.diagnostics_dict(),
        )


def commit_schedule(timesteps: torch.Tensor, gen_length: int) -> list[int]:
    """Mirror the sampler's float32 truncation, including zero/multiple commits.

    Compute timestep ratios on the original device, then transfer once before
    timing. No per-step host synchronization is needed for this schedule.
    """
    ratios = (1 - timesteps[1:] / timesteps[:-1]).cpu()
    remaining = gen_length
    schedule = []
    for step, ratio in enumerate(ratios):
        count = int(remaining * ratio) if step < len(ratios) - 1 else remaining
        schedule.append(count)
        remaining -= count
    return schedule
