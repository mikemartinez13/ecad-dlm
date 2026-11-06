"""Minimal persistent cache with one transient layer workspace."""
from __future__ import annotations
import torch
from .batched_row_index import gather_rows, scatter_rows


class BatchedActiveTokenKVCache:
    def __init__(self, *, num_layers, hidden_size, num_kv_heads, head_dim, device, dtype,
                 retained_boundaries=()):
        self.num_layers, self.hidden_size = num_layers, hidden_size
        self.num_kv_heads, self.head_dim = num_kv_heads, head_dim
        self.device, self.dtype = device, dtype
        self.retained_boundaries = set(retained_boundaries)
        self.ready = False
        self.shape = None

    def reset(self, sequence_length, batch_size=1):
        if self.shape == (batch_size, sequence_length):
            for tensor in (self.workspace, self.final_hidden, *self.boundary_hidden.values(), *self.keys, *self.values):
                tensor.zero_()
            self.ready = False
            return
        # Release the previous batch before allocating a different shape.
        self.keys, self.values, self.boundary_hidden = [], [], {}
        self.workspace = self.final_hidden = None
        self.shape = (batch_size, sequence_length)
        hidden_shape = (*self.shape, self.hidden_size)
        self.workspace = torch.zeros(hidden_shape, device=self.device, dtype=self.dtype)
        self.boundary_hidden = {i: torch.zeros_like(self.workspace) for i in self.retained_boundaries}
        self.final_hidden = torch.zeros_like(self.workspace)
        shape = (batch_size, self.num_kv_heads, sequence_length, self.head_dim)
        # Padded keys/values must be finite, even though their attention is masked.
        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.ready = False

    def boundary(self, layer_boundary, positions):
        return gather_rows(self.boundary_hidden.get(layer_boundary, self.workspace), positions)

    def update_boundary(self, layer_boundary, positions, values):
        scatter_rows(self.workspace, positions, values)
        if layer_boundary in self.boundary_hidden:
            scatter_rows(self.boundary_hidden[layer_boundary], positions, values)

    def kv(self, layer):
        return self.keys[layer], self.values[layer]

    def update_kv(self, layer, positions, keys, values, step):
        # View [B,H,L,D] as [B,L,H*D] is not necessarily writable without a copy;
        # index directly in the transposed view instead.
        batch = torch.arange(positions.shape[0], device=positions.device)[:, None].expand_as(positions)
        valid = positions.ge(0)
        self.keys[layer].transpose(1, 2)[batch[valid], positions[valid]] = keys.transpose(1, 2)[valid]
        self.values[layer].transpose(1, 2)[batch[valid], positions[valid]] = values.transpose(1, 2)[valid]

    def final(self, positions):
        return gather_rows(self.final_hidden, positions)

    def update_final(self, positions, values):
        scatter_rows(self.final_hidden, positions, values)

    def mark_refreshed(self, positions, step):
        self.ready = True
