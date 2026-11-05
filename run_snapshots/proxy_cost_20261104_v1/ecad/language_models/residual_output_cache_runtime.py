"""Shared denoising context, residual cache, and accounting for DLM backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


COMPONENTS = ("attention", "mlp")


@dataclass(frozen=True)
class DenoisingContext:
    block_index: int
    local_step: int
    global_step: int


@dataclass
class DLMCacheStats:
    recomputed: dict[str, int] = field(
        default_factory=lambda: {component: 0 for component in COMPONENTS}
    )
    reused: dict[str, int] = field(
        default_factory=lambda: {component: 0 for component in COMPONENTS}
    )
    forced_recomputes: dict[str, int] = field(
        default_factory=lambda: {component: 0 for component in COMPONENTS}
    )
    blocks_started: int = 0
    steps_started: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        for component in COMPONENTS:
            total = self.recomputed[component] + self.reused[component]
            result[f"{component}_reuse_fraction"] = (
                self.reused[component] / total if total else 0.0
            )
        return result


class ResidualOutputCacheRuntime:
    """Mutable cache state shared by model-specific DLM layer hooks."""

    def __init__(self, schedule: DLMCacheSchedule) -> None:
        self.schedule = schedule
        self.context: DenoisingContext | None = None
        self._active_block: int | None = None
        self.stats = DLMCacheStats()
        self._cache: dict[tuple[int, str], torch.Tensor] = {}

    def set_schedule(self, schedule: DLMCacheSchedule) -> None:
        if schedule.num_layers != self.schedule.num_layers:
            raise ValueError("Cannot change to a schedule with a different layer count")
        if schedule.num_inference_steps != self.schedule.num_inference_steps:
            raise ValueError("Cannot change to a schedule with a different step count")
        self.schedule = schedule
        self.start_generation()

    def start_generation(self) -> None:
        self.context = None
        self._active_block = None
        self.stats = DLMCacheStats()
        self._cache.clear()

    def begin_block(self, block_index: int) -> None:
        """Invalidate residuals when entering a new generation canvas/block."""

        if block_index < 0:
            raise ValueError("block_index must be non-negative")
        self.context = None
        self._active_block = block_index
        self._cache.clear()
        self.stats.blocks_started += 1

    def begin_step(self, block_index: int, local_step: int, global_step: int) -> None:
        if self._active_block != block_index:
            raise RuntimeError(
                "begin_block() must be called before steps in each generation block"
            )
        if not 0 <= local_step < self.schedule.num_inference_steps:
            raise IndexError(f"Schedule step {local_step} is outside the schedule")
        if global_step < 0:
            raise ValueError("global_step must be non-negative")
        self.context = DenoisingContext(block_index, local_step, global_step)
        self.stats.steps_started += 1

    def cached_or_none(
        self, layer: int, component: str, reference: torch.Tensor
    ) -> torch.Tensor | None:
        """Return a compatible cached residual or account for recomputation."""

        if self.context is None:
            raise RuntimeError("begin_step() must be called before model.forward()")
        if component not in COMPONENTS:
            raise ValueError(f"Unknown component {component!r}")

        should_recompute = self.schedule.recompute(
            self.context.local_step, layer, component
        )
        cached = self._cache.get((layer, component))
        compatible = (
            cached is not None
            and cached.shape == reference.shape
            and cached.device == reference.device
            and cached.dtype == reference.dtype
        )
        if should_recompute or not compatible:
            self.stats.recomputed[component] += 1
            if not should_recompute:
                self.stats.forced_recomputes[component] += 1
            return None

        self.stats.reused[component] += 1
        return cached

    def store(self, layer: int, component: str, residual: torch.Tensor) -> None:
        # Inference is no-grad, but detach keeps the cache safe for direct calls.
        self._cache[(layer, component)] = residual.detach()
