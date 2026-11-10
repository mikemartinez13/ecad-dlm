"""Shared masked-row reuse policy for replay and batched decoding."""
from __future__ import annotations
from typing import Any
from numbers import Integral
import torch
from .denoising_token_state import DenoisingTokenCategory
from .fixed_token_refresh_policy import TokenRefreshDecision

PADDING_TOP = 4
# Committed rows the entropy trigger refreshes on each side of a new commit.
ENTROPY_LOCAL_RADIUS = 4

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
    """A base policy (refresh_256 or depth_22) plus masked-row reuse between synchronized refreshes.

    ``padding_window`` gives masked rows whose stale prediction is a stop token their
    own, longer window; they then also skip the base's periodic full refreshes. Zero
    keeps them on ``window``.

    ``entropy_threshold`` is a trigger after EntropyCache: a commit with more entropy
    (nats) than this refreshes rows on the next step. Scope "full" is the paper's
    full recomputation of that request; "local" refreshes only the committed rows
    within ENTROPY_LOCAL_RADIUS of the commit. Zero turns it off.
    """

    def __init__(self, base: Any, window: int, m: int, rule: str, stop_ids: set[int], padding_window: int = 0,
                 entropy_threshold: float = 0.0, entropy_scope: str = "local") -> None:
        if rule not in ("split", "split+recent"):
            raise ValueError(f"Unknown protected-set rule {rule!r}")
        self.base, self.window, self.m, self.rule = base, window, m, rule
        self.padding_window, self.entropy_threshold, self.entropy_scope = padding_window, entropy_threshold, entropy_scope
        self.name = f"{rule}_m{m}_r{window}"
        if padding_window:
            self.name += f"_pad{padding_window}"
        if entropy_threshold:
            self.name += f"_h{entropy_threshold:g}{entropy_scope}"
        self.stop_ids = sorted(stop_ids)
        self.confidence = self.top1 = self.protected = self.mask_positions = None

    def validate(self) -> None:
        self.base.validate()
        if (not isinstance(self.window, Integral) or not isinstance(self.m, Integral)
                or self.window < 1 or self.m < 1):
            raise ValueError("window and m must be positive integers")
        if self.padding_window < 0 or self.padding_window % self.window:
            raise ValueError("padding_window must be zero or a multiple of window")
        if self.entropy_threshold < 0 or self.entropy_scope not in ("local", "full"):
            raise ValueError("entropy_threshold must be non-negative and entropy_scope local or full")

    def to_dict(self) -> dict[str, Any]:
        return {"base": self.base.to_dict(), "window": self.window, "m": self.m, "rule": self.rule,
                "padding_top": PADDING_TOP, "padding_window": self.padding_window,
                "entropy_threshold": self.entropy_threshold, "entropy_scope": self.entropy_scope}

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

    def entropy_rows(self, states: torch.Tensor) -> torch.Tensor:
        """Rows this step's trigger refreshes, [batch, sequence]: the entropy of a commit is its last confidence."""

        new = states.eq(int(DenoisingTokenCategory.NEWLY_DECODED))
        hot = new & (-self.confidence.reshape(states.shape)).gt(self.entropy_threshold)
        if self.entropy_scope == "full":
            return hot.any(dim=-1, keepdim=True) & states.ne(int(DenoisingTokenCategory.PAD))
        rows = hot.clone()
        for offset in range(1, ENTROPY_LOCAL_RADIUS + 1):
            rows[:, offset:] |= hot[:, :-offset]
            rows[:, :-offset] |= hot[:, offset:]
        return rows & (new | states.eq(int(DenoisingTokenCategory.STABLE_DECODED)))

    def decide(self, *, states: torch.Tensor, decoded_age: torch.Tensor, step: int, **kwargs: Any) -> TokenRefreshDecision:
        decision = self.base.decide(states=states, decoded_age=decoded_age, step=step, **kwargs)
        mask = (states if self.batched else states[0]).eq(int(DenoisingTokenCategory.MASK))
        self.mask_positions = mask if self.batched else torch.where(mask)[0]
        self.protected = mask
        # Steps 0 and 1 and the terminal refresh recompute everything, so nothing is reused.
        if any(reason in decision.reasons for reason in ("initial_full", "one_step_delay_full", "terminal_full_refresh")):
            return decision
        # Base periodic refreshes have no "mask" reason. Every `window` steps all answer-like
        # masked rows refresh, and padding-like ones do too unless they have their own window.
        refresh = "mask" not in decision.reasons or step % self.window == 0
        refresh_padding = step % self.padding_window == 0 if self.padding_window else refresh
        active, depths = decision.active, decision.refresh_depth
        reasons = dict(decision.reasons)
        if not (refresh and refresh_padding):
            self.protected = self.protected_positions(mask, decoded_age if self.batched else decoded_age[0])
            if refresh:
                padding_like = torch.isin(self.top1, torch.tensor(self.stop_ids, device=mask.device))
                self.protected = self.protected | (mask & ~padding_like)
            predictors = torch.zeros_like(mask)
            predictors[..., :-1] = self.protected[..., 1:]
            reused = mask & ~(self.protected | predictors)
            active = active & ~(reused if self.batched else reused[None])
            reasons["reused_mask"] = int(reused.sum().item()) if kwargs.get("collect_counts", True) else 0
        if self.entropy_threshold:
            rows = self.entropy_rows(states)
            active = active | rows
            depths = depths.masked_fill(rows, 1)
            if self.entropy_scope == "full":
                # Masked rows of a triggered request are all fresh, so all may commit.
                self.protected = self.protected | (mask & (rows if self.batched else rows[0]))
        return TokenRefreshDecision(active=active, refresh_depth=depths, reasons=reasons)

    def observe(self, logits: torch.Tensor) -> None:
        """Store the confidences of this step's mask logits (stale ones for reused rows)."""

        from .entropy_confidence import entropy_confidence

        self.confidence[self.mask_positions] = entropy_confidence(logits, 0.0, 1.0, None).float()
        self.top1[self.mask_positions] = logits.argmax(dim=-1)


