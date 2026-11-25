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
    def from_checkpoint(cls, path: str) -> "ModelShape":
        """The shape from a checkpoint's config.json, without loading weights."""

        import json
        from pathlib import Path

        config = json.loads((Path(path) / "config.json").read_text())
        return cls(hidden_size=config["hidden_size"], intermediate_size=config["intermediate_size"],
                   num_heads=config["num_attention_heads"], num_kv_heads=config["num_key_value_heads"],
                   num_layers=config["num_hidden_layers"], vocab_size=config["vocab_size"])

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


def padded_batch_flops(shape: ModelShape, tasks: list[dict], batch: int, depth: int, steps: int,
                       gen_length: int) -> tuple[float, float]:
    """FLOPs of a policy and of dense when tasks decode in padded batches of ``batch``.

    A padded batch computes a rectangle: at each step every request is charged the
    batch's largest row count per layer, and its longest sequence. Batches are
    consecutive tasks sorted by prompt length, as the audit's timing builds them.
    Each task is ``{"task_id", "prompt_tokens", "rows"}``, with ``rows`` the per-step
    ``{"active", "full_depth", "masked"}`` counts that teacher forcing records. Rows
    below ``depth`` are ``full_depth`` rows; from ``depth`` up, ``active`` rows. At
    ``batch`` 1 this is each task's own FLOPs, as teacher forcing counts them.
    """

    ordered = sorted(tasks, key=lambda task: (task["prompt_tokens"], task["task_id"]))
    depth = min(depth, shape.num_layers)
    policy = dense = 0.0
    for offset in range(0, len(ordered), batch):
        group = ordered[offset:offset + batch]
        length = max(task["prompt_tokens"] for task in group) + gen_length
        full_layer, head = layer_flops(shape, 1, length), lm_head_flops(shape, 1)
        layer_rows = masked_rows = 0
        for step in range(steps):
            rows = [task["rows"][step] for task in group]
            # Depth 1 has no shallow layers, so full_depth rows are counted once, in ``active``.
            layer_rows += ((depth - 1) * max(row["full_depth"] for row in rows)
                           + (shape.num_layers - depth + 1) * max(row["active"] for row in rows))
            masked_rows += max(row["masked"] for row in rows)
        policy += len(group) * (layer_rows * full_layer + masked_rows * head)
        dense += len(group) * steps * dense_forward_flops(shape, length)
    return policy, dense
