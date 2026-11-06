"""Shared masked-row reuse policy for replay and batched decoding."""
from __future__ import annotations
from typing import Any
from numbers import Integral
import torch
from .denoising_token_state import DenoisingTokenCategory
from .fixed_token_refresh_policy import TokenRefreshDecision

PADDING_TOP = 4

def dilate(flags: torch.Tensor) -> torch.Tensor:
    out = flags.clone()
    out[..., 1:] |= flags[..., :-1]
    out[..., :-1] |= flags[..., 1:]
    return out


def top_positions(scores: torch.Tensor, allowed: torch.Tensor, count: int) -> torch.Tensor:
    """The ``count`` allowed positions with the highest score; ties go to the leftmost."""

    order = torch.sort(scores.masked_fill(~allowed, -torch.inf), descending=True, stable=True).indices[..., :count]
    chosen = torch.zeros_like(allowed)
    chosen.scatter_(-1, order, True)
    return chosen & allowed


class MaskedRowReusePolicy:
    """A base policy (refresh_256 or depth_22) plus masked-row reuse between synchronized refreshes."""

    def __init__(self, base: Any, window: int, m: int, rule: str, stop_ids: set[int]) -> None:
        if rule not in ("split", "split+recent"):
            raise ValueError(f"Unknown protected-set rule {rule!r}")
        self.base, self.window, self.m, self.rule = base, window, m, rule
        self.name = f"{rule}_m{m}_r{window}"
        self.stop_ids = sorted(stop_ids)
        self.confidence = self.top1 = self.protected = self.mask_positions = None

    def validate(self) -> None:
        self.base.validate()
        if (not isinstance(self.window, Integral) or not isinstance(self.m, Integral)
                or self.window < 1 or self.m < 1):
            raise ValueError("window and m must be positive integers")

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base.to_dict(), "window": self.window, "m": self.m, "rule": self.rule,
                "padding_top": PADDING_TOP}

    def make_backend(self, model: torch.nn.Module) -> "MaskedRowReuseBackend":
        from .masked_row_reuse_backend import MaskedRowReuseBackend
        return MaskedRowReuseBackend(model, self)

    def reset(self, sequence_length: int, device: torch.device, batch_size: int | None = None) -> None:
        self.batched = batch_size is not None
        shape = (sequence_length,) if batch_size is None else (batch_size, sequence_length)
        # The confidence and top-1 token each position had when its logits were last computed.
        self.confidence = torch.full(shape, -torch.inf, device=device)
        self.top1 = torch.full(shape, -1, dtype=torch.long, device=device)

    def protected_positions(self, mask: torch.Tensor, decoded_age: torch.Tensor) -> torch.Tensor:
        """Step 0's rule on the policy's own (stale) confidences; bool over positions."""

        padding_like = torch.isin(self.top1, torch.tensor(self.stop_ids, device=mask.device))
        protected = top_positions(self.confidence, mask & ~padding_like, self.m)
        protected |= top_positions(self.confidence, mask & padding_like, PADDING_TOP)
        if self.rule == "split+recent":
            # Neighbors of positions committed in the last `window` steps.
            protected |= dilate(decoded_age.ge(1) & decoded_age.le(self.window)) & mask
        return protected

    def decide(self, *, states: torch.Tensor, decoded_age: torch.Tensor, step: int, **kwargs: Any) -> TokenRefreshDecision:
        decision = self.base.decide(states=states, decoded_age=decoded_age, step=step, **kwargs)
        mask = (states if self.batched else states[0]).eq(int(DenoisingTokenCategory.MASK))
        self.mask_positions = mask if self.batched else torch.where(mask)[0]
        # The base's full steps (0, 1 and its periodic refreshes) have no "mask" reason;
        # every `window` steps all masked rows refresh.
        if "mask" not in decision.reasons or step % self.window == 0:
            self.protected = mask
            return decision
        self.protected = self.protected_positions(mask, decoded_age if self.batched else decoded_age[0])
        predictors = torch.zeros_like(mask)
        predictors[..., :-1] = self.protected[..., 1:]
        reused = mask & ~(self.protected | predictors)
        return TokenRefreshDecision(
            active=decision.active & ~(reused if self.batched else reused[None]),
            refresh_depth=decision.refresh_depth,
            reasons={**decision.reasons, "reused_mask": int(reused.sum().item()) if kwargs.get("collect_counts", True) else 0},
        )

    def observe(self, logits: torch.Tensor) -> None:
        """Store the confidences of this step's mask logits (stale ones for reused rows)."""

        from .entropy_confidence import entropy_confidence

        self.confidence[self.mask_positions] = entropy_confidence(logits, 0.0, 1.0, None).float()
        self.top1[self.mask_positions] = logits.argmax(dim=-1)


