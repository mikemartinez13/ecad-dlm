import json
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.genetic.population_io_manager import PopulationIOManager
from ecad.schedulers.cache_scheduler.dream_cache_schedule import Dream7bCacheSchedule
from ecad.types import CacheScheduleDict

# NOTE: Update these defaults if you want Dream-specific output roots.
DEFAULT_POPULATIONS_DIR = Path(__file__).parents[2] / Path("results/genetic/populations/")
DEFAULT_BENCHMARKS_DIR = Path(__file__).parents[2] / Path("results/benchmark/genetic/populations/")


# ----------------------------
# Dream MVP cache schedule types
# ----------------------------
# MVP schema (maps PixArt's {attn1, attn2, ff} -> Dream {layer, mlp, kv}).
# - "layer": coarse toggle for caching the layer-level hidden/state (acts like a broad "reuse layer output" gate)
# - "mlp": whether to cache MLP block outputs
# - "kv":  whether to cache attention KV (future extension; for MVP you can ignore at runtime if not supported)
#
# You can add more keys later ("attn_qkv", "attn_out", "cross_attn", etc.) without changing the IO pattern.
DreamCacheScheduleDict = dict[int, dict[str, dict[str, bool]]]


def dream_gen_default(num_blocks: int, num_inference_steps: int) -> Dream7bCacheSchedule:
    """
    Default schedule generator (MVP):
      - Cache nothing everywhere (all False).
    You can change this to match your real default policy.
    """
    schedule: DreamCacheScheduleDict = {}
    for step in range(num_inference_steps):
        schedule[step] = {
            str(layer_idx): {"layer": False, "mlp": False, "kv": False}
            for layer_idx in range(num_blocks)
        }

    return Dream7bCacheSchedule(
        num_blocks=num_blocks,
        num_inference_steps=num_inference_steps,
        name="dream_default",
        schedule=schedule,
        top_level_config={},
        attributes={},
        metrics={},
    )


class DreamPopulationIOManager(PopulationIOManager):
    """
    Dream 7B adaptation of PixArtPopulationIOManager.

    Key differences vs PixArt:
      - "blocks" -> "layers"
      - component keys map:
          PixArt attn1 -> Dream layer (coarse layer-level caching)
          PixArt attn2 -> Dream kv    (future extension; can be ignored for MVP)
          PixArt ff    -> Dream mlp
      - schedule dict keys are (diff_step -> layer_idx -> toggles)
    """

    def __init__(
        self,
        name: str,
        all_populations_dir: Path = DEFAULT_POPULATIONS_DIR,
        all_benchmarks_dir: Path = DEFAULT_BENCHMARKS_DIR,
        generation_num: int | None = None,
        num_inference_steps: int = 20,   # diffusion denoise steps for Dream
        min_diff_from_default: int = 1,
        population_size: int = 72,
        num_blocks: int = 32,            # set to Dream 7B's layer count as used in your implementation
        num_component_types: int = 3,     # [layer, mlp, kv] (MVP keeps kv, even if runtime ignores)
        maximize_macs: bool = False,
    ) -> None:
        default_schedule = dream_gen_default(num_blocks, num_inference_steps)

        super().__init__(
            name=name,
            all_populations_dir=all_populations_dir,
            all_benchmarks_dir=all_benchmarks_dir,
            generation_num=generation_num,
            num_inference_steps=num_inference_steps,
            min_diff_from_default=min_diff_from_default,
            population_size=population_size,
            default_schedule=default_schedule,
            maximize_macs=maximize_macs,
        )

        self.num_blocks = num_blocks
        self.num_component_types = num_component_types

    def to_dict(self) -> dict[str, Any]:
        return super().to_dict() | {
            "num_blocks": self.num_blocks,
            "num_component_types": self.num_component_types,
        }

    @classmethod
    def from_json(cls, file_path: Path) -> "PopulationIOManager":
        with open(file_path, "r") as f:
            config = json.load(f)

        all_population_dir = Path(config["population_dir"]).parent
        all_benchmarks_dir = Path(config["benchmark_dir"]).parent

        args: dict[str, Any] = {
            "name": config["name"],
            "all_populations_dir": all_population_dir,
            "all_benchmarks_dir": all_benchmarks_dir,
        }

        optional_keys = [
            "generation_num",
            "num_inference_steps",
            "num_blocks",
            "num_component_types",
            "min_diff_from_default",
            "population_size",
            "maximize_macs",
        ]
        for key in optional_keys:
            if key in config:
                args[key] = config[key]

        return cls(**args)

    def save_population(
        self,
        population: npt.NDArray,
        generation: int | None = None,
    ) -> None:
        for i, candidate_vector in enumerate(population):
            schedule_dict = self.binary_vector_to_schedule_dict(
                candidate_vector,
                num_inference_steps=self.num_inference_steps,
                num_layers=self.num_blocks,
                num_component_types=self.num_component_types,
            )
            additional_info = self.compute_additional_info(schedule_dict)

            cache_schedule = Dream7bCacheSchedule(
                num_blocks=self.num_blocks,
                num_inference_steps=self.num_inference_steps,
                name=f"{self.name}_gen_{self.generation_num:03d}_cand_{i:03d}",
                schedule=schedule_dict,
                attributes=additional_info,
                top_level_config=self.candidate_config,
            )

            candidate_file = self._get_candidate_filename(i, generation)
            cache_schedule.to_json(candidate_file)
            print(
                f"Saved candidate {i} in generation {self.generation_num} to {candidate_file}"
            )

    def load_population_schedules(
        self, generation: int | None = None
    ) -> list[tuple[int, Dream7bCacheSchedule]]:
        candidates: list[tuple[int, Dream7bCacheSchedule]] = []
        for file_path in self._get_candidates_dir(generation).glob("cand_*.json"):
            parts = file_path.stem.split("_")
            try:
                candidate_index = int(parts[-1])
            except ValueError:
                candidate_index = -1
                print(f"WARNING: Could not parse candidate index from {file_path}")

            cache_schedule = Dream7bCacheSchedule.from_json(file_path)
            schedule_config = cache_schedule.top_level_config

            if not self.candidate_config:
                self.candidate_config = schedule_config
            elif self.candidate_config != schedule_config:
                print(
                    f"WARNING: Candidate config mismatch. "
                    f"Expected {self.candidate_config}, got {schedule_config}"
                )

            candidates.append((candidate_index, cache_schedule))

        candidates.sort(key=lambda t: t[0])
        return candidates

    def compute_additional_info(
        self,
        schedule: CacheScheduleDict,  # base type; we treat it as DreamCacheScheduleDict
    ) -> dict[str, Any]:
        """
        Same spirit as PixArt: quantify how much this schedule deviates from the default.
        """
        sched: DreamCacheScheduleDict = schedule  # type: ignore[assignment]
        default_sched: DreamCacheScheduleDict = self.default_schedule.schedule  # type: ignore[assignment]

        num_affected_steps = 0
        affected_layers: set[int] = set()
        total_num_affected_layers = 0

        for step in range(self.num_inference_steps):
            step_sched = sched.get(step, {})
            step_default = default_sched.get(step, {})
            if step_sched != step_default:
                num_affected_steps += 1

            for layer_idx in range(self.num_blocks):
                k = str(layer_idx)
                toggles = step_sched.get(k, {})
                toggles_default = step_default.get(k, {})
                if toggles != toggles_default:
                    total_num_affected_layers += 1
                    affected_layers.add(layer_idx)

        return {
            "num_affected_steps": num_affected_steps,
            "num_affected_layers": len(affected_layers),
            "total_num_affected_layers": total_num_affected_layers,
        }

    @staticmethod
    def binary_vector_to_schedule_dict(
        x: np.ndarray,
        num_inference_steps: int,
        **kwargs: Any,
    ) -> CacheScheduleDict:
        """
        MVP mapping from PixArt schema -> Dream schema.

        Expected x length:
          num_inference_steps * num_layers * num_component_types

        Component ordering for Dream MVP:
          0 -> "layer"  (coarse layer caching)   [PixArt attn1]
          1 -> "kv"     (attention KV caching)   [PixArt attn2]  (can be ignored in runtime for MVP)
          2 -> "mlp"    (MLP caching)            [PixArt ff]

        NOTE: PixArt used keys {"attn1","attn2","ff"}.
              This keeps the same "3-component" chromosome shape for MVP.
        """
        if "num_layers" not in kwargs or "num_component_types" not in kwargs:
            raise ValueError("binary_vector_to_schedule_dict requires num_layers and num_component_types")

        num_layers = int(kwargs["num_layers"])
        num_component_types = int(kwargs["num_component_types"])
        if num_component_types != 3:
            raise ValueError("Dream MVP expects num_component_types=3 for [layer, kv, mlp]")

        arr = x.reshape((num_inference_steps, num_layers, num_component_types))

        schedule: DreamCacheScheduleDict = {}
        for step in range(num_inference_steps):
            layer_schedule: dict[str, dict[str, bool]] = {}
            for layer_idx in range(num_layers):
                # order: [layer, kv, mlp]
                layer_schedule[str(layer_idx)] = {
                    "layer": bool(arr[step, layer_idx, 0]),
                    "kv": bool(arr[step, layer_idx, 1]),
                    "mlp": bool(arr[step, layer_idx, 2]),
                }
            schedule[step] = layer_schedule

        return schedule  # type: ignore[return-value]
