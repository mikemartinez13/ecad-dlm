"""Per-generation K/V, boundary-hidden, and final-hidden cache state."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ActiveTokenKVCache:
    """Dense cache storage with sparse row updates; v1 is deliberately batch one."""

    num_layers: int
    hidden_size: int
    num_kv_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype
    boundary_hidden: list[torch.Tensor] | None = None
    keys: list[torch.Tensor] | None = None
    values: list[torch.Tensor] | None = None
    final_hidden: torch.Tensor | None = None
    token_last_refresh: torch.Tensor | None = None
    layer_last_refresh: torch.Tensor | None = None
    ready: bool = False

    def reset(self, sequence_length: int) -> None:
        shape = (1, sequence_length)
        self.boundary_hidden = [
            torch.empty((1, sequence_length, self.hidden_size), device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers + 1)
        ]
        self.keys = [
            torch.empty((1, self.num_kv_heads, sequence_length, self.head_dim), device=self.device, dtype=self.dtype)
            for _ in range(self.num_layers)
        ]
        self.values = [torch.empty_like(key) for key in self.keys]
        self.final_hidden = torch.empty(
            (1, sequence_length, self.hidden_size), device=self.device, dtype=self.dtype
        )
        self.token_last_refresh = torch.full(shape, -1, device=self.device, dtype=torch.long)
        self.layer_last_refresh = torch.full(
            (self.num_layers, 1, sequence_length), -1, device=self.device, dtype=torch.long
        )
        self.ready = False

    def _check(self) -> None:
        if self.boundary_hidden is None or self.keys is None or self.values is None:
            raise RuntimeError("Cache has not been reset for this generation")

    def update_boundary(self, layer_boundary: int, positions: torch.Tensor, values: torch.Tensor) -> None:
        self._check()
        self.boundary_hidden[layer_boundary].index_copy_(1, positions, values)

    def boundary(self, layer_boundary: int, positions: torch.Tensor) -> torch.Tensor:
        self._check()
        return self.boundary_hidden[layer_boundary].index_select(1, positions)

    def update_kv(
        self,
        layer: int,
        positions: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        step: int,
    ) -> None:
        self._check()
        self.keys[layer].index_copy_(2, positions, keys)
        self.values[layer].index_copy_(2, positions, values)
        if self.layer_last_refresh is None:
            raise RuntimeError("Cache has not been reset for this generation")
        self.layer_last_refresh[layer, :, positions] = step

    def kv(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._check()
        return self.keys[layer], self.values[layer]

    def update_final(self, positions: torch.Tensor, values: torch.Tensor) -> None:
        if self.final_hidden is None:
            raise RuntimeError("Cache has not been reset for this generation")
        self.final_hidden.index_copy_(1, positions, values)

    def final(self, positions: torch.Tensor) -> torch.Tensor:
        if self.final_hidden is None:
            raise RuntimeError("Cache has not been reset for this generation")
        return self.final_hidden.index_select(1, positions)

    def mark_refreshed(self, positions: torch.Tensor, step: int) -> None:
        if self.token_last_refresh is None:
            raise RuntimeError("Cache has not been reset for this generation")
        indices = torch.where(positions[0])[0]
        self.token_last_refresh[:, indices] = step
        self.ready = True

    def token_age(self, step: int) -> torch.Tensor:
        if self.token_last_refresh is None:
            raise RuntimeError("Cache has not been reset for this generation")
        return (step - self.token_last_refresh).clamp_min(0)

    def layer_age(self, layer: int, step: int) -> torch.Tensor:
        if self.layer_last_refresh is None:
            raise RuntimeError("Cache has not been reset for this generation")
        return (step - self.layer_last_refresh[layer]).clamp_min(0)
