"""Fixed active-token refresh rules for the legacy multi-policy screen."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .denoising_token_state import DenoisingTokenCategory


_STATE_NAMES = {
    DenoisingTokenCategory.PROMPT: "prompt",
    DenoisingTokenCategory.NEWLY_DECODED: "newly_decoded",
    DenoisingTokenCategory.STABLE_DECODED: "stable_decoded",
}


@dataclass(frozen=True)
class TokenRefreshDecision:
    active: torch.Tensor
    refresh_depth: torch.Tensor
    reasons: dict[str, int]


@dataclass(frozen=True)
class FixedTokenRefreshPolicy:
    """An interpretable fixed token-refresh rule used by the multi-policy screen.

    ``additional_decoded_forwards`` is deliberately phrased in forwards, not
    token age: 0 permits reuse on the next forward after commitment, 1 forces
    one extra forward, and so on.  ``refresh_depths`` maps phase names
    (early/middle/late/default) to per-state, one-based layer boundaries.
    """

    name: str
    additional_decoded_forwards: int = 1
    max_cache_age: int | None = None
    prompt_refresh_interval: int | None = None
    local_refresh_radius: int = 0
    terminal_full_refresh_steps: int = 0
    refresh_depths: dict[str, dict[str, int]] = field(
        default_factory=lambda: {"default": {"prompt": 1, "stable_decoded": 1}}
    )
    force_dense: bool = False

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> "FixedTokenRefreshPolicy":
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"Policy {name!r} must be a mapping")
        depths = raw.get("refresh_depths", raw.get("refresh_depth", {}))
        if isinstance(depths, int):
            depths = {"default": {"prompt": depths, "stable_decoded": depths}}
        if not depths:
            depths = {"default": {"prompt": 1, "stable_decoded": 1}}
        if not isinstance(depths, dict):
            raise ValueError(f"Policy {name!r}.refresh_depths must be a mapping")
        normalized: dict[str, dict[str, int]] = {}
        for phase, values in depths.items():
            if isinstance(values, int):
                values = {"prompt": values, "stable_decoded": values}
            if not isinstance(values, dict):
                raise ValueError(f"Policy {name!r} phase {phase!r} must be a mapping")
            normalized[str(phase)] = {str(key): int(value) for key, value in values.items()}
        return cls(
            name=name,
            additional_decoded_forwards=int(raw.get("additional_decoded_forwards", 1)),
            max_cache_age=(
                None if raw.get("max_cache_age") is None else int(raw["max_cache_age"])
            ),
            prompt_refresh_interval=(
                None
                if raw.get("prompt_refresh_interval") is None
                else int(raw["prompt_refresh_interval"])
            ),
            local_refresh_radius=int(raw.get("local_refresh_radius", 0)),
            terminal_full_refresh_steps=int(raw.get("terminal_full_refresh_steps", 0)),
            refresh_depths=normalized,
            force_dense=bool(raw.get("force_dense", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "additional_decoded_forwards": self.additional_decoded_forwards,
            "max_cache_age": self.max_cache_age,
            "prompt_refresh_interval": self.prompt_refresh_interval,
            "local_refresh_radius": self.local_refresh_radius,
            "terminal_full_refresh_steps": self.terminal_full_refresh_steps,
            "refresh_depths": self.refresh_depths,
            "force_dense": self.force_dense,
        }

    def validate(self) -> None:
        values = (
            self.additional_decoded_forwards,
            self.local_refresh_radius,
            self.terminal_full_refresh_steps,
        )
        if any(value < 0 for value in values):
            raise ValueError(f"Policy {self.name!r} has a negative parameter")
        if self.max_cache_age is not None and self.max_cache_age < 1:
            raise ValueError("max_cache_age must be positive when set")
        if self.prompt_refresh_interval is not None and self.prompt_refresh_interval < 1:
            raise ValueError("prompt_refresh_interval must be positive when set")
        for phase, states in self.refresh_depths.items():
            if not states:
                raise ValueError(f"Policy {self.name!r} phase {phase!r} is empty")
            if any(depth < 1 for depth in states.values()):
                raise ValueError("refresh depths are one-based positive layer indices")

    @staticmethod
    def _phase(step: int, total_steps: int) -> str:
        fraction = step / max(total_steps, 1)
        return "early" if fraction < 1 / 3 else "middle" if fraction < 2 / 3 else "late"

    def _depth_for(self, states: torch.Tensor, step: int, total_steps: int) -> torch.Tensor:
        table = self.refresh_depths.get(self._phase(step, total_steps))
        table = table or self.refresh_depths.get("default", {})
        default = int(table.get("default", 1))
        depths = torch.full_like(states, default, dtype=torch.long)
        for state, name in _STATE_NAMES.items():
            depths[states.eq(int(state))] = int(table.get(name, default))
        # A mask/newly committed token has changed or still needs a current
        # bidirectional representation. It must start from layer one.
        depths[states.eq(int(DenoisingTokenCategory.MASK))] = 1
        depths[states.eq(int(DenoisingTokenCategory.NEWLY_DECODED))] = 1
        return depths

    def decide(
        self,
        *,
        states: torch.Tensor,
        decoded_age: torch.Tensor,
        token_cache_age: torch.Tensor,
        cache_ready: bool,
        step: int,
        total_steps: int,
        num_layers: int,
    ) -> TokenRefreshDecision:
        """Select active rows and their one-based refresh boundaries."""

        if states.ndim != 2 or states.shape[0] != 1:
            raise ValueError("The first backend supports a single prompt per forward")
        valid = ~states.eq(int(DenoisingTokenCategory.PAD))
        depths = self._depth_for(states, step, total_steps).clamp(max=num_layers)
        masked = states.eq(int(DenoisingTokenCategory.MASK))
        newly = states.eq(int(DenoisingTokenCategory.NEWLY_DECODED))
        prompt = states.eq(int(DenoisingTokenCategory.PROMPT))
        decoded = states.eq(int(DenoisingTokenCategory.STABLE_DECODED))
        active = masked.clone()
        reasons = {"mask": int(masked.sum().item())}

        delayed = (newly | decoded) & decoded_age.le(self.additional_decoded_forwards)
        active |= delayed
        reasons["decode_delay"] = int(delayed.sum().item())

        stale = torch.zeros_like(active)
        if self.max_cache_age is not None:
            stale = valid & token_cache_age.ge(self.max_cache_age)
            active |= stale
        reasons["cache_age"] = int(stale.sum().item())

        prompt_refresh = torch.zeros_like(active)
        if self.prompt_refresh_interval and step % self.prompt_refresh_interval == 0:
            prompt_refresh = prompt
            active |= prompt_refresh
        reasons["prompt"] = int(prompt_refresh.sum().item())

        local = torch.zeros_like(active)
        if self.local_refresh_radius and newly.any():
            centers = torch.where(newly[0])[0]
            positions = torch.arange(states.shape[1], device=states.device)
            local[0] = (positions[:, None] - centers[None, :]).abs().le(
                self.local_refresh_radius
            ).any(dim=1)
            local &= valid
            active |= local
        reasons["local"] = int(local.sum().item())

        terminal = step >= total_steps - self.terminal_full_refresh_steps
        if self.force_dense or terminal or not cache_ready:
            active = valid
            depths = torch.ones_like(depths)
        reasons["terminal_or_dense"] = int(active.sum().item()) if (
            self.force_dense or terminal or not cache_ready
        ) else 0
        return TokenRefreshDecision(active=active, refresh_depth=depths, reasons=reasons)
