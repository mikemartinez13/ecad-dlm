"""Native DiffuCoder generation with ECAD denoising-step instrumentation."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import torch

from ecad.language_models.residual_output_cache_runtime import ResidualOutputCacheRuntime


@dataclass
class DiffuCoderGenerationResult:
    token_ids: torch.Tensor
    history: list[torch.Tensor]
    elapsed_seconds: float
    cache_stats: dict[str, Any] | None


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _DenoisingStepHook:
    """Set the ECAD context immediately before each native model forward."""

    def __init__(self, runtime: ResidualOutputCacheRuntime, steps: int) -> None:
        self.runtime = runtime
        self.steps = steps
        self.calls: list[int | None] = []

    def __call__(
        self,
        completed_step: int | None,
        tokens: torch.Tensor,
        logits: torch.Tensor | None,
    ) -> torch.Tensor:
        del logits
        self.calls.append(completed_step)
        if completed_step is None:
            self.runtime.start_generation()
            # DiffuCoder denoises one full completion canvas without LLaDA-style
            # semi-autoregressive generation blocks. Use one synthetic block so
            # the existing safe cache runtime can be shared.
            self.runtime.begin_block(0)
            self.runtime.begin_step(0, 0, 0)
        elif completed_step + 1 < self.steps:
            next_step = completed_step + 1
            self.runtime.begin_step(0, next_step, next_step)
        return tokens


@torch.no_grad()
def generate_diffucoder(
    model: torch.nn.Module,
    prompt: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
    steps: int = 256,
    gen_length: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.95,
    top_k: int | None = None,
    algorithm: str = "entropy",
    algorithm_temperature: float = 0.0,
    cache_runtime: ResidualOutputCacheRuntime | None = None,
) -> DiffuCoderGenerationResult:
    if prompt.ndim != 2:
        raise ValueError("prompt must have shape (batch, sequence)")
    if attention_mask is not None and attention_mask.shape != prompt.shape:
        raise ValueError("attention_mask must have the same shape as prompt")
    if steps < 1 or gen_length < 1:
        raise ValueError("steps and gen_length must be positive")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if algorithm not in {"origin", "maskgit_plus", "topk_margin", "entropy"}:
        raise ValueError(f"Unsupported DiffuCoder algorithm {algorithm!r}")
    if cache_runtime is not None:
        if cache_runtime.schedule.num_inference_steps != steps:
            raise ValueError("Schedule step count does not match DiffuCoder steps")
        step_hook: _DenoisingStepHook | None = _DenoisingStepHook(
            cache_runtime, steps
        )
    else:
        step_hook = None

    device = prompt.device
    generation_kwargs: dict[str, Any] = {
        "attention_mask": attention_mask,
        "max_new_tokens": gen_length,
        "steps": steps,
        "temperature": temperature,
        "top_p": top_p,
        "alg": algorithm,
        "alg_temp": algorithm_temperature,
        "output_history": True,
        "return_dict_in_generate": True,
    }
    if top_k is not None:
        generation_kwargs["top_k"] = top_k
    if step_hook is not None:
        generation_kwargs["generation_tokens_hook_func"] = step_hook

    _synchronize(device)
    start_time = perf_counter()
    output = model.diffusion_generate(prompt, **generation_kwargs)
    _synchronize(device)
    elapsed_seconds = perf_counter() - start_time

    history = list(output.history or [])
    if len(history) != steps:
        raise RuntimeError(
            f"DiffuCoder returned {len(history)} history states for {steps} steps"
        )
    if step_hook is not None:
        expected_calls: list[int | None] = [None, *range(steps)]
        if step_hook.calls != expected_calls:
            raise RuntimeError(
                "Unexpected DiffuCoder token-hook order: "
                f"expected {expected_calls}, got {step_hook.calls}"
            )

    stats = cache_runtime.stats.to_dict() if cache_runtime is not None else None
    return DiffuCoderGenerationResult(
        token_ids=output.sequences,
        history=history,
        elapsed_seconds=elapsed_seconds,
        cache_stats=stats,
    )
