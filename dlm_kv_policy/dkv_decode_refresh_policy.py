"""dKV-Decode policies adapted to DiffuCoder's one-canvas sampler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .fixed_token_refresh_policy import TokenRefreshDecision
from .denoising_token_state import DenoisingTokenCategory

IMPLEMENTATION_REVISION = "depth_boundary_v2"


@dataclass(frozen=True)
class DKVDecodeRefreshPolicy:
    """Fixed dKV-Decode and its small, interpretable ablation family.

    DiffuCoder has one 256-step denoising canvas rather than LLaDA's optional
    semi-autoregressive blocks. Thus refresh is defined over global steps.

    ``refresh_depth > 1`` enables the depth-boundary ablation: on ordinary
    cached steps, stable decoded rows resume from that cached layer boundary.
    Masks and recently decoded rows still begin at layer one.
    """

    name: str = "dkv_decode_reference"
    cache_reloading_step: int = 8
    additional_decoded_forwards: int = 1
    refresh_depth: int = 1
    prompt_refresh_interval: int | None = None
    local_refresh_radius: int = 0
    terminal_full_refresh_steps: int = 0
    force_dense: bool = False

    @classmethod
    def from_dict(cls, name: str, raw: Any) -> "DKVDecodeRefreshPolicy":
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"dKV policy {name!r} must be a mapping")
        return cls(
            name=name,
            cache_reloading_step=int(raw.get("cache_reloading_step", 8)),
            additional_decoded_forwards=int(raw.get("additional_decoded_forwards", 1)),
            refresh_depth=int(raw.get("refresh_depth", 1)),
            prompt_refresh_interval=(
                None if raw.get("prompt_refresh_interval") is None
                else int(raw["prompt_refresh_interval"])
            ),
            local_refresh_radius=int(raw.get("local_refresh_radius", 0)),
            terminal_full_refresh_steps=int(raw.get("terminal_full_refresh_steps", 0)),
        )

    def validate(self) -> None:
        if self.cache_reloading_step < 1:
            raise ValueError("cache_reloading_step must be positive")
        if self.additional_decoded_forwards < 0 or self.refresh_depth < 1:
            raise ValueError("delay must be non-negative and refresh_depth positive")
        if self.prompt_refresh_interval is not None and self.prompt_refresh_interval < 1:
            raise ValueError("prompt_refresh_interval must be positive when set")
        if self.local_refresh_radius < 0 or self.terminal_full_refresh_steps < 0:
            raise ValueError("local and terminal refresh parameters must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "implementation_revision": IMPLEMENTATION_REVISION,
            "cache_reloading_step": self.cache_reloading_step,
            "additional_decoded_forwards": self.additional_decoded_forwards,
            "refresh_depth": self.refresh_depth,
            "prompt_refresh_interval": self.prompt_refresh_interval,
            "local_refresh_radius": self.local_refresh_radius,
            "terminal_full_refresh_steps": self.terminal_full_refresh_steps,
            "force_dense": False,
            "reference": "dKV-Decode family adapted to DiffuCoder global denoising steps",
        }

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
        del token_cache_age
        if states.ndim != 2 or states.shape[0] != 1:
            raise ValueError("dKV-Decode reference supports batch size one")
        valid = ~states.eq(int(DenoisingTokenCategory.PAD))
        depths = torch.full_like(states, self.refresh_depth, dtype=torch.long).clamp(
            max=num_layers
        )
        terminal = step >= total_steps - self.terminal_full_refresh_steps
        if not cache_ready or step == 0:
            reason = "initial_full"
        elif step == 1 and self.additional_decoded_forwards:
            reason = "one_step_delay_full"
        elif step % self.cache_reloading_step == 0:
            reason = "periodic_full_refresh"
        elif terminal:
            reason = "terminal_full_refresh"
        else:
            mask = states.eq(int(DenoisingTokenCategory.MASK))
            newly_decoded = states.eq(int(DenoisingTokenCategory.NEWLY_DECODED))
            stable_decoded = states.eq(int(DenoisingTokenCategory.STABLE_DECODED))
            decoded = newly_decoded | states.eq(int(DenoisingTokenCategory.STABLE_DECODED))
            delayed = decoded & decoded_age.ge(0) & decoded_age.le(self.additional_decoded_forwards)
            active = mask | delayed
            # A depth boundary is useful only if stable rows actually enter
            # the backend. They resume from the cached shallow boundary, so
            # only layers refresh_depth..L are recomputed.
            depth_stable = stable_decoded if self.refresh_depth > 1 else torch.zeros_like(active)
            active |= depth_stable
            prompt = torch.zeros_like(active)
            if self.prompt_refresh_interval and step % self.prompt_refresh_interval == 0:
                prompt = states.eq(int(DenoisingTokenCategory.PROMPT))
                active |= prompt
            local = torch.zeros_like(active)
            if self.local_refresh_radius and newly_decoded.any():
                centers = torch.where(newly_decoded[0])[0]
                positions = torch.arange(states.shape[1], device=states.device)
                local[0] = (positions[:, None] - centers[None, :]).abs().le(
                    self.local_refresh_radius
                ).any(dim=1)
                local &= valid
                active |= local
            depths[mask | newly_decoded | local] = 1
            return TokenRefreshDecision(
                active=active,
                refresh_depth=depths,
                reasons={
                    "mask": int(mask.sum().item()),
                    "decode_delay": int(delayed.sum().item()),
                    "depth_stable": int(depth_stable.sum().item()),
                    "prompt": int(prompt.sum().item()),
                    "local": int(local.sum().item()),
                    "cached_stable": int((valid & ~active).sum().item()),
                },
            )
        return TokenRefreshDecision(
            active=valid,
            refresh_depth=torch.ones_like(depths),
            reasons={reason: int(valid.sum().item())},
        )
