"""Deterministic compute cost, used instead of wall-clock time during search.

FLOPs are counted from the number of rows each forward actually computes, so
the same policy gets the same cost on every GPU type. One multiply-add counts
as 2 FLOPs. Norms, RoPE, and softmax are ignored; they are a small fraction.

Wall-clock time is still the number to report, measured on finalists on one
GPU type: on an A40, wall time fell only about 0.38x as fast as this cost.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ModelShape:
    hidden_size: int
    intermediate_size: int
    num_heads: int
    num_kv_heads: int
    num_layers: int
    vocab_size: int

    @classmethod
    def from_model(cls, model: torch.nn.Module) -> "ModelShape":
        config = model.config
        return cls(
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.intermediate_size),
            num_heads=int(config.num_attention_heads),
            num_kv_heads=int(config.num_key_value_heads),
            num_layers=int(config.num_hidden_layers),
            vocab_size=int(model.lm_head.out_features),
        )


def layer_flops(shape: ModelShape, rows: int, key_length: int) -> int:
    """One decoder layer computing ``rows`` query rows against ``key_length`` keys."""

    head_dim = shape.hidden_size // shape.num_heads
    query_and_output = 2 * shape.hidden_size * shape.hidden_size
    key_and_value = 2 * shape.hidden_size * shape.num_kv_heads * head_dim
    mlp = 3 * shape.hidden_size * shape.intermediate_size  # gate, up, down
    attention = 2 * key_length * shape.hidden_size  # scores, then weighted values
    return 2 * rows * (query_and_output + key_and_value + mlp + attention)


def lm_head_flops(shape: ModelShape, rows: int) -> int:
    return 2 * rows * shape.hidden_size * shape.vocab_size


def dense_forward_flops(shape: ModelShape, sequence_length: int) -> int:
    """Native DiffuCoder forward: every row through every layer and the LM head."""

    return shape.num_layers * layer_flops(shape, sequence_length, sequence_length) + lm_head_flops(
        shape, sequence_length
    )
