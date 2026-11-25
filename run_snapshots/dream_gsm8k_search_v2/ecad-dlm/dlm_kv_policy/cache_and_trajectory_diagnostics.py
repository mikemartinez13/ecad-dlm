"""Streaming scientific diagnostics; none of these are optimization objectives."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F

from .denoising_token_state import DenoisingTokenCategory


def _state_name(value: int) -> str:
    return DenoisingTokenCategory(value).name.lower()


@dataclass
class KVCacheDiagnostics:
    """Aggregate drift and work statistics without retaining full tensors/logits."""

    active_rows: dict[tuple[int, int], list[int]] = field(
        default_factory=lambda: defaultdict(lambda: [0, 0])
    )
    lm_head_rows: list[int] = field(default_factory=lambda: [0, 0])
    policy_reasons: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    drift: dict[tuple[int, int, str, int, int], list[float]] = field(
        default_factory=lambda: defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    )

    def record_policy(self, reasons: dict[str, int]) -> None:
        for key, value in reasons.items():
            self.policy_reasons[key] += int(value)

    def record_active(self, step: int, layer: int, active: int, total: int) -> None:
        entry = self.active_rows[(step, layer)]
        entry[0] += active
        entry[1] += total

    def record_lm_head(self, active: int, total: int) -> None:
        self.lm_head_rows[0] += active
        self.lm_head_rows[1] += total

    @torch.no_grad()
    def observe_kv(
        self,
        *,
        step: int,
        layer: int,
        states: torch.Tensor,
        cache_age: torch.Tensor,
        positions: torch.Tensor,
        old_keys: torch.Tensor,
        old_values: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
    ) -> None:
        """Accumulate relative L2 drift and cosine similarity for recomputed rows."""

        if positions.numel() == 0:
            return
        old_keys = old_keys.index_select(2, positions).transpose(1, 2).flatten(2)
        old_values = old_values.index_select(2, positions).transpose(1, 2).flatten(2)
        new_keys = new_keys.transpose(1, 2).flatten(2)
        new_values = new_values.transpose(1, 2).flatten(2)
        labels = states.index_select(1, positions)[0]
        ages = cache_age.index_select(1, positions)[0]
        for state_value in torch.unique(labels).tolist():
            mask = labels.eq(state_value)
            if not mask.any():
                continue
            key = (step, layer, _state_name(int(state_value)), int(ages[mask].float().mean().item()), int(mask.sum().item()))
            old_k, new_k = old_keys[0, mask], new_keys[0, mask]
            old_v, new_v = old_values[0, mask], new_values[0, mask]
            rel_k = (new_k - old_k).norm(dim=-1) / new_k.norm(dim=-1).clamp_min(1e-8)
            rel_v = (new_v - old_v).norm(dim=-1) / new_v.norm(dim=-1).clamp_min(1e-8)
            cos_k = F.cosine_similarity(new_k.float(), old_k.float(), dim=-1)
            cos_v = F.cosine_similarity(new_v.float(), old_v.float(), dim=-1)
            entry = self.drift[key]
            count = int(mask.sum().item())
            entry[0] += float(rel_k.sum().item())
            entry[1] += float(rel_v.sum().item())
            entry[2] += float(cos_k.sum().item())
            entry[3] += float(cos_v.sum().item())
            entry[4] += count

    def to_dict(self) -> dict[str, Any]:
        active = {
            f"step_{step:03d}/layer_{layer:02d}": {
                "active_rows": values[0],
                "total_rows": values[1],
                "active_fraction": values[0] / values[1] if values[1] else 0.0,
            }
            for (step, layer), values in sorted(self.active_rows.items())
        }
        drift = {}
        for (step, layer, state, age, _), values in self.drift.items():
            count = values[4]
            key = f"step_{step:03d}/layer_{layer:02d}/{state}/cache_age_{age}"
            entry = drift.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            for index, value in enumerate(values):
                entry[index] += value
        return {
            "active_rows_by_step_layer": active,
            "lm_head": {
                "active_rows": self.lm_head_rows[0],
                "total_rows": self.lm_head_rows[1],
                "active_fraction": (
                    self.lm_head_rows[0] / self.lm_head_rows[1]
                    if self.lm_head_rows[1]
                    else 0.0
                ),
            },
            "policy_reason_counts": dict(self.policy_reasons),
            "kv_drift": {
                key: {
                    "relative_l2_k": values[0] / values[4],
                    "relative_l2_v": values[1] / values[4],
                    "cosine_k": values[2] / values[4],
                    "cosine_v": values[3] / values[4],
                    "observations": int(values[4]),
                }
                for key, values in sorted(drift.items())
                if values[4]
            },
        }


def trajectory_metrics(
    baseline_history: list[torch.Tensor], candidate_history: list[torch.Tensor]
) -> dict[str, Any]:
    """Token agreement and first divergence are explanatory diagnostics only."""

    if len(baseline_history) != len(candidate_history):
        raise ValueError("Cannot compare trajectories with different step counts")
    matches = total = 0
    first_divergence: int | None = None
    for step, (baseline, candidate) in enumerate(zip(baseline_history, candidate_history)):
        equal = baseline.eq(candidate)
        matches += int(equal.sum().item())
        total += equal.numel()
        if first_divergence is None and not bool(equal.all()):
            first_divergence = step
    return {
        "token_agreement": matches / total if total else 1.0,
        "first_trajectory_divergence_step": first_divergence,
    }

