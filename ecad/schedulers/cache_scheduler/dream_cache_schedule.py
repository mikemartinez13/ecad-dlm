from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt

from ecad.schedulers.cache_scheduler.cache_schedule import CacheSchedule
from ecad.types import CustomFuncDict  # reuse if your project already defines this


# --------------------------------------------------------------------
# Dream 7B MVP schedule schema
# --------------------------------------------------------------------
# We keep the same "block schedule" shape as CacheSchedule expects:
#   schedule[step][block_num][component] -> bool
#
# For Dream, interpret:
#   - block_num  == layer index (stringified int)
#   - components == ["layer", "kv", "mlp"]  (MVP)
#
# This maps PixArt's 3-component chromosome naturally:
#   PixArt attn1 -> Dream layer  (coarse layer-level caching gate)
#   PixArt attn2 -> Dream kv     (attention KV caching)
#   PixArt ff    -> Dream mlp    (MLP caching)
#
# The runtime can ignore "kv" if you haven't implemented it yet.
#
# Optional: allow "custom_compute_attn"/"custom_compute_mlp" hooks per layer per step.
# --------------------------------------------------------------------


class Dream7bCacheSchedule(CacheSchedule):
    """
    Cache schedule for a Dream 7B text diffusion language model.

    Conventions:
      - num_blocks refers to the number of transformer *layers* (Dream layers)
      - num_inference_steps refers to the number of diffusion denoising steps
      - schedule is indexed as: schedule[step][layer_idx_str][component] -> bool

    Components (MVP):
      - "layer": coarse caching toggle (e.g., reuse layer output / hidden state)
      - "kv": cache attention KV (optional; can be ignored if not supported yet)
      - "mlp": cache MLP output (or skip recompute / reuse cached MLP activations)
    """

    @property
    def components(self) -> list[str]:
        # keep 3 components to mirror PixArt chromosome dimensionality (MVP)
        return ["layer", "kv", "mlp"]

    def to_numpy(self, flatten: bool = False):
        arr = np.zeros((self.num_inference_steps, self.num_blocks, 3), dtype=np.bool_)
        for step in range(self.num_inference_steps):
            if step not in self.schedule:
                raise KeyError(f"Missing step {step} in schedule")
            layer_schedule = self.schedule[step]
            for layer_idx in range(self.num_blocks):
                k = str(layer_idx)
                if k not in layer_schedule:
                    raise KeyError(f"Missing layer {k} at step {step}")
                comp_sched = layer_schedule[k]
                for i, component in enumerate(self.components):
                    if component not in comp_sched:
                        raise KeyError(f"Missing component {component} for layer {k} step {step}")
                    arr[step, layer_idx, i] = bool(comp_sched[component])
        return arr.flatten() if flatten else arr


    # --- Optional "custom compute" hooks (mirrors PixArt style) ---

    def get_custom_compute_attn(self, layer_num: str) -> CustomFuncDict:
        """
        Optional per-step, per-layer override for attention computation.

        If your schedule dict includes:
          schedule[step][layer]["custom_compute_attn"] = {...}
        this returns it, otherwise {}.
        """
        return self.schedule[self.curr_step][layer_num].get("custom_compute_attn", {})  # type: ignore[return-value]

    def get_custom_compute_mlp(self, layer_num: str) -> CustomFuncDict:
        """
        Optional per-step, per-layer override for MLP computation.

        If your schedule dict includes:
          schedule[step][layer]["custom_compute_mlp"] = {...}
        this returns it, otherwise {}.
        """
        return self.schedule[self.curr_step][layer_num].get("custom_compute_mlp", {})  # type: ignore[return-value]

    # Convenience aliases (if you prefer Dream naming)
    def get_custom_compute_ff(self, layer_num: str) -> CustomFuncDict:
        # allow old callers that expect "ff"
        return self.get_custom_compute_mlp(layer_num)
