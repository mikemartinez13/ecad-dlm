"""Block-wise masked diffusion generation for LLaDA.

This is a small, dependency-local adaptation of the MIT-licensed official
LLaDA sampler at https://github.com/ML-GSAI/LLaDA/blob/main/generate.py.
Its token transition behavior is intentionally unchanged; the only ECAD hooks
are the explicit denoising context updates around each model forward.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F

from ecad.language_models.residual_output_cache_runtime import ResidualOutputCacheRuntime


@dataclass
class LLaDAGenerationResult:
    token_ids: torch.Tensor
    elapsed_seconds: float
    cache_stats: dict[str, Any] | None


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply the official float64 Gumbel-max transformation."""

    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    if steps < 1:
        raise ValueError("steps must be positive")
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps
    result = torch.zeros(
        mask_num.size(0),
        steps,
        device=mask_index.device,
        dtype=torch.int64,
    ) + base
    for batch_index in range(mask_num.size(0)):
        result[batch_index, : int(remainder[batch_index].item())] += 1
    return result


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def generate_llada(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    cfg_scale: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    eos_token_id: int = 126081,
    eot_token_id: int = 126348,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    cache_runtime: ResidualOutputCacheRuntime | None = None,
) -> LLaDAGenerationResult:
    """Generate a fixed-length answer with optional ECAD residual caching.

    ``cache_runtime.schedule`` is indexed by local step. A schedule therefore
    repeats for each of the ``gen_length // block_length`` generation blocks.
    """

    if prompt.ndim != 2:
        raise ValueError("prompt must have shape (batch, sequence)")
    if gen_length < 1 or block_length < 1 or gen_length % block_length != 0:
        raise ValueError("gen_length must be positive and divisible by block_length")
    num_generation_blocks = gen_length // block_length
    if steps < 1 or steps % num_generation_blocks != 0:
        raise ValueError("steps must be positive and divisible by generation blocks")
    local_steps = steps // num_generation_blocks
    if local_steps > block_length:
        raise ValueError("local denoising steps cannot exceed block_length")
    if remasking not in {"low_confidence", "random"}:
        raise ValueError("remasking must be 'low_confidence' or 'random'")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if cache_runtime is not None:
        if cache_runtime.schedule.num_inference_steps != local_steps:
            raise ValueError(
                "Schedule local step count does not match steps / generation blocks"
            )
        cache_runtime.start_generation()

    device = prompt.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    if attention_mask is not None:
        if attention_mask.shape != prompt.shape:
            raise ValueError("attention_mask must have the same shape as prompt")
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones(
                    (prompt.shape[0], gen_length),
                    dtype=attention_mask.dtype,
                    device=device,
                ),
            ],
            dim=-1,
        )

    prompt_index = x != mask_id
    _synchronize(device)
    start_time = perf_counter()

    for generation_block in range(num_generation_blocks):
        if cache_runtime is not None:
            cache_runtime.begin_block(generation_block)
        block_start = prompt.shape[1] + generation_block * block_length
        block_end = block_start + block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        transfer_counts = get_num_transfer_tokens(block_mask_index, local_steps)

        for local_step in range(local_steps):
            global_step = generation_block * local_steps + local_step
            if cache_runtime is not None:
                cache_runtime.begin_step(generation_block, local_step, global_step)

            mask_index = x == mask_id
            candidate_mask_index = mask_index.clone()
            candidate_mask_index[:, :block_start] = False
            candidate_mask_index[:, block_end:] = False

            if cfg_scale > 0:
                unconditional_x = x.clone()
                unconditional_x[prompt_index] = mask_id
                model_input = torch.cat([x, unconditional_x], dim=0)
                model_attention_mask = (
                    torch.cat([attention_mask, attention_mask], dim=0)
                    if attention_mask is not None
                    else None
                )
                logits = model(
                    model_input,
                    attention_mask=model_attention_mask,
                    use_cache=False,
                ).logits
                conditional_logits, unconditional_logits = torch.chunk(
                    logits, 2, dim=0
                )
                logits = unconditional_logits + (cfg_scale + 1) * (
                    conditional_logits - unconditional_logits
                )
            else:
                logits = model(
                    x, attention_mask=attention_mask, use_cache=False
                ).logits

            if logits_eos_inf:
                logits = logits.clone()
                logits[:, :, eos_token_id] = -torch.inf
            logits_with_noise = add_gumbel_noise(logits, temperature)
            predicted_tokens = torch.argmax(logits_with_noise, dim=-1)

            # Match the official LLaDA sampler: suppress EOS from the
            # confidence ranking and EOT from confidence values when this
            # dataset-specific option is enabled.
            if confidence_eos_eot_inf:
                logits_with_noise = logits_with_noise.clone()
                logits_with_noise[:, :, eos_token_id] = -torch.inf
                logits = logits.clone()
                logits[:, :, eot_token_id] = -torch.inf

            if remasking == "low_confidence":
                probabilities = F.softmax(logits, dim=-1)
                confidence = torch.gather(
                    probabilities,
                    dim=-1,
                    index=predicted_tokens.unsqueeze(-1),
                ).squeeze(-1)
            else:
                confidence = torch.rand(
                    predicted_tokens.shape, device=predicted_tokens.device
                )

            confidence[:, block_end:] = -torch.inf
            predicted_tokens = torch.where(mask_index, predicted_tokens, x)
            confidence = torch.where(mask_index, confidence, -torch.inf)
            transfer_index = torch.zeros_like(
                predicted_tokens, dtype=torch.bool, device=device
            )
            for batch_index in range(confidence.shape[0]):
                count = int(transfer_counts[batch_index, local_step].item())
                if count:
                    selected = torch.topk(confidence[batch_index], k=count).indices
                    transfer_index[batch_index, selected] = True
            x[transfer_index] = predicted_tokens[transfer_index]

    _synchronize(device)
    elapsed_seconds = perf_counter() - start_time
    stats = cache_runtime.stats.to_dict() if cache_runtime is not None else None
    return LLaDAGenerationResult(x, elapsed_seconds, stats)
