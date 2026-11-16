"""Persistent batched cache. Every row tensor has one spare slot for dummy rows.

Requests in a batch recompute different numbers of rows, so packed row sets are
padded. Padded (dummy) rows carry position ``sequence_length`` and are written to
the spare slot, which attention masks out. Writes are therefore plain scatters:
nothing has to wait for the GPU to report how many rows are real.
"""
from __future__ import annotations
import torch


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
            for tensor in (self.final_hidden, *self.boundary_hidden.values(), *self.keys, *self.values):
                tensor.zero_()
            self.ready = False
            return
        # Release the previous batch before allocating a different shape.
        self.keys, self.values, self.boundary_hidden = [], [], {}
        self.final_hidden = None
        self.shape = (batch_size, sequence_length)
        hidden_shape = (batch_size, sequence_length + 1, self.hidden_size)
        self.final_hidden = torch.zeros(hidden_shape, device=self.device, dtype=self.dtype)
        self.boundary_hidden = {i: torch.zeros_like(self.final_hidden) for i in self.retained_boundaries}
        shape = (batch_size, self.num_kv_heads, sequence_length + 1, self.head_dim)
        # Padded keys/values must be finite, even though their attention is masked.
        self.keys = [torch.zeros(shape, device=self.device, dtype=self.dtype) for _ in range(self.num_layers)]
        self.values = [torch.zeros_like(k) for k in self.keys]
        self.ready = False

    @staticmethod
    def _rows(positions, width):
        return positions[..., None].expand(-1, -1, width)

    def boundary(self, layer_boundary, positions):
        return self.boundary_hidden[layer_boundary].gather(1, self._rows(positions, self.hidden_size))

    def update_boundary(self, layer_boundary, positions, values):
        self.boundary_hidden[layer_boundary].scatter_(1, self._rows(positions, self.hidden_size), values)

    def kv(self, layer):
        return self.keys[layer], self.values[layer]

    def update_kv(self, layer, positions, keys, values, step=None):
        index = positions[:, None, :, None].expand(-1, self.num_kv_heads, -1, self.head_dim)
        self.keys[layer].scatter_(2, index, keys)
        self.values[layer].scatter_(2, index, values)

    def final(self, positions):
        return self.final_hidden.gather(1, self._rows(positions, self.hidden_size))

    def update_final(self, positions, values):
        self.final_hidden.scatter_(1, self._rows(positions, self.hidden_size), values)

    def mark_refreshed(self, positions, step):
        self.ready = True
