"""Attention/MLP caching schedules for diffusion language models.

The serialized schedule follows ECAD's existing convention: ``True`` means
recompute a component and ``False`` means reuse its cached residual output.
The explicit schedule domain distinguishes LLaDA's repeated local-step schedule
from DiffuCoder's single global denoising trajectory.
"""

from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.schedulers.cache_scheduler.cache_schedule import CacheSchedule
from ecad.types import DLMBlockScheduleDict, DLMCacheScheduleDict


class DLMCacheSchedule(CacheSchedule[DLMCacheScheduleDict]):
    """Layer-wise attention/MLP recomputation decisions for a DLM.

    The evolutionary genome omits schedule step zero because there is no valid
    residual cache at the beginning of a denoising canvas/block. The JSON
    schedule is expanded to every step and layer so it remains inspectable and
    uses the same schema as the original ECAD schedules.
    """

    COMPONENTS = ("attention", "mlp")
    SCHEDULE_DOMAINS = {
        "local_step_repeated_per_block",
        "global_denoising_step",
    }

    def __init__(
        self,
        num_blocks: int,
        num_inference_steps: int,
        name: str,
        schedule: dict[int | str, DLMBlockScheduleDict] | DLMCacheScheduleDict,
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        step_group_size: int = 1,
        layer_group_size: int = 1,
        schedule_domain: str = "local_step_repeated_per_block",
        **kwargs: Any,
    ) -> None:
        if step_group_size < 1 or layer_group_size < 1:
            raise ValueError("step_group_size and layer_group_size must be positive")
        if schedule_domain not in self.SCHEDULE_DOMAINS:
            raise ValueError(
                f"Unsupported DLM schedule domain {schedule_domain!r}; choose from "
                f"{sorted(self.SCHEDULE_DOMAINS)}"
            )
        self.step_group_size = int(step_group_size)
        self.layer_group_size = int(layer_group_size)
        self.schedule_domain = schedule_domain
        super().__init__(
            num_blocks=num_blocks,
            num_inference_steps=num_inference_steps,
            name=name,
            schedule=schedule,
            top_level_config=top_level_config,
            attributes=attributes,
            metrics=metrics,
            **kwargs,
        )
        self._validate()

    @property
    def num_layers(self) -> int:
        """Alias ECAD's generic ``num_blocks`` field for language models."""

        return self.num_blocks

    @property
    def components(self) -> list[str]:
        return list(self.COMPONENTS)

    @property
    def num_step_groups(self) -> int:
        return ceil(max(self.num_inference_steps - 1, 0) / self.step_group_size)

    @property
    def num_layer_groups(self) -> int:
        return ceil(self.num_layers / self.layer_group_size)

    @property
    def genome_shape(self) -> tuple[int, int, int]:
        return self.num_step_groups, self.num_layer_groups, len(self.COMPONENTS)

    @property
    def genome_size(self) -> int:
        return int(np.prod(self.genome_shape))

    def _validate(self) -> None:
        if self.num_layers < 1:
            raise ValueError("num_blocks/num_layers must be positive")
        if self.num_inference_steps < 2:
            raise ValueError("DLM schedules require at least two local denoising steps")

        expected_steps = set(range(self.num_inference_steps))
        if set(self.schedule) != expected_steps:
            raise ValueError(
                f"Schedule steps must be exactly {sorted(expected_steps)}, got "
                f"{sorted(self.schedule)}"
            )

        expected_layers = {str(layer) for layer in range(self.num_layers)}
        for step, layer_schedule in self.schedule.items():
            if set(layer_schedule) != expected_layers:
                raise ValueError(
                    f"Step {step} must contain layers 0..{self.num_layers - 1}"
                )
            for layer, component_schedule in layer_schedule.items():
                if set(component_schedule) != set(self.COMPONENTS):
                    raise ValueError(
                        f"Step {step}, layer {layer} must contain exactly "
                        f"{self.COMPONENTS}"
                    )
                if not all(
                    isinstance(component_schedule[name], bool)
                    for name in self.COMPONENTS
                ):
                    raise TypeError("All DLM schedule decisions must be bool values")

        if not self.to_numpy(flatten=False)[0].all():
            raise ValueError(
                "Every component must recompute at local step zero; no cache exists yet"
            )

    def recompute(self, schedule_step: int, layer: int, component: str) -> bool:
        if component not in self.COMPONENTS:
            raise ValueError(f"Unknown DLM component {component!r}")
        if not 0 <= schedule_step < self.num_inference_steps:
            raise IndexError(f"Step {schedule_step} is outside the schedule")
        if not 0 <= layer < self.num_layers:
            raise IndexError(f"Layer {layer} is outside the schedule")
        return bool(self.schedule[schedule_step][str(layer)][component])

    def to_numpy(self, flatten: bool = False) -> npt.NDArray[np.bool_]:
        array = np.empty(
            (self.num_inference_steps, self.num_layers, len(self.COMPONENTS)),
            dtype=np.bool_,
        )
        for step in range(self.num_inference_steps):
            for layer in range(self.num_layers):
                for component_index, component in enumerate(self.COMPONENTS):
                    array[step, layer, component_index] = self.recompute(
                        step, layer, component
                    )
        return array.flatten() if flatten else array

    def to_grouped_numpy(self, flatten: bool = True) -> npt.NDArray[np.bool_]:
        """Recover the compact evolutionary genome from an expanded schedule."""

        grouped = np.empty(self.genome_shape, dtype=np.bool_)
        expanded = self.to_numpy(flatten=False)
        for step_group in range(self.num_step_groups):
            step_start = 1 + step_group * self.step_group_size
            step_end = min(step_start + self.step_group_size, self.num_inference_steps)
            for layer_group in range(self.num_layer_groups):
                layer_start = layer_group * self.layer_group_size
                layer_end = min(layer_start + self.layer_group_size, self.num_layers)
                region = expanded[step_start:step_end, layer_start:layer_end, :]
                value = region[0, 0, :]
                if not np.all(region == value):
                    raise ValueError(
                        "Expanded schedule is not constant inside its configured groups"
                    )
                grouped[step_group, layer_group, :] = value
        return grouped.flatten() if flatten else grouped

    def to_dict(self) -> dict[str, Any]:
        data = super().to_dict()
        data["cache_schedule"]["step_group_size"] = self.step_group_size
        data["cache_schedule"]["layer_group_size"] = self.layer_group_size
        data["cache_schedule"]["schedule_domain"] = self.schedule_domain
        return data

    def compute_fraction(
        self,
        attention_weight: float = 1.0,
        mlp_weight: float = 1.0,
        fixed_step_weight: float = 0.0,
    ) -> float:
        """Return weighted recomputed work divided by all-recompute work."""

        if attention_weight <= 0 or mlp_weight <= 0 or fixed_step_weight < 0:
            raise ValueError("Component weights must be positive and fixed work non-negative")
        array = self.to_numpy(flatten=False)
        numerator = (
            array[:, :, 0].sum() * attention_weight
            + array[:, :, 1].sum() * mlp_weight
            + self.num_inference_steps * fixed_step_weight
        )
        denominator = (
            self.num_inference_steps
            * self.num_layers
            * (attention_weight + mlp_weight)
            + self.num_inference_steps * fixed_step_weight
        )
        return float(numerator / denominator)

    @classmethod
    def from_genome(
        cls,
        genome: npt.ArrayLike,
        *,
        num_layers: int,
        num_inference_steps: int,
        step_group_size: int,
        layer_group_size: int,
        name: str,
        top_level_config: dict[str, Any] | None = None,
        attributes: dict[str, Any] | None = None,
        metrics: dict[str, Any] | None = None,
        schedule_domain: str = "local_step_repeated_per_block",
    ) -> "DLMCacheSchedule":
        num_step_groups = ceil((num_inference_steps - 1) / step_group_size)
        num_layer_groups = ceil(num_layers / layer_group_size)
        expected_shape = (num_step_groups, num_layer_groups, len(cls.COMPONENTS))
        array = np.asarray(genome, dtype=np.bool_)
        if array.size != int(np.prod(expected_shape)):
            raise ValueError(
                f"Genome has {array.size} decisions; expected "
                f"{int(np.prod(expected_shape))} for shape {expected_shape}"
            )
        array = array.reshape(expected_shape)

        schedule: DLMCacheScheduleDict = {}
        for step in range(num_inference_steps):
            layer_schedule: DLMBlockScheduleDict = {}
            for layer in range(num_layers):
                if step == 0:
                    decisions = (True, True)
                else:
                    decisions = array[
                        (step - 1) // step_group_size,
                        layer // layer_group_size,
                    ]
                layer_schedule[str(layer)] = {
                    "attention": bool(decisions[0]),
                    "mlp": bool(decisions[1]),
                }
            schedule[step] = layer_schedule

        merged_attributes = {
            "genome_shape": list(expected_shape),
            "num_grouped_decisions": int(np.prod(expected_shape)),
            "num_cached_groups": int((~array).sum()),
        }
        if attributes:
            merged_attributes.update(attributes)
        return cls(
            num_blocks=num_layers,
            num_inference_steps=num_inference_steps,
            name=name,
            schedule=schedule,
            step_group_size=step_group_size,
            layer_group_size=layer_group_size,
            schedule_domain=schedule_domain,
            top_level_config=top_level_config,
            attributes=merged_attributes,
            metrics=metrics,
        )

    @classmethod
    def all_recompute(
        cls,
        *,
        num_layers: int,
        num_inference_steps: int,
        step_group_size: int = 1,
        layer_group_size: int = 1,
        name: str = "all_recompute",
        top_level_config: dict[str, Any] | None = None,
        schedule_domain: str = "local_step_repeated_per_block",
    ) -> "DLMCacheSchedule":
        num_step_groups = ceil((num_inference_steps - 1) / step_group_size)
        num_layer_groups = ceil(num_layers / layer_group_size)
        genome = np.ones(
            (num_step_groups, num_layer_groups, len(cls.COMPONENTS)),
            dtype=np.bool_,
        )
        return cls.from_genome(
            genome,
            num_layers=num_layers,
            num_inference_steps=num_inference_steps,
            step_group_size=step_group_size,
            layer_group_size=layer_group_size,
            name=name,
            top_level_config=top_level_config,
            schedule_domain=schedule_domain,
        )
