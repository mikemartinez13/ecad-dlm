"""Step 1b of the masked-row reuse study: how much time does recomputing fewer rows save?

Times a plain forward of ``n`` recomputed rows per request, attending to all L
cached positions, followed by the LM head, at several batch sizes. There is no
cache and no policy logic; it is the same forward as decoding_oracles.py's
``MinimalForward``, with a batch dimension. ``commit_position_study.py report
--timing`` combines the curve with each policy's per-step row counts to project
wall-clock speedups. The reference for "dense" is the n = L point.

Results go to <run>/<GPU type>/batched_row_timing/timing.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .decoding_oracles import cuda_seconds
from dlm_kv_policy.diffucoder_active_token_backend import _apply_rope, _repeat_kv

BATCH_SIZES = (1, 8, 16)
ROW_GRID = (1, 2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 160, 192, 224, 256)


class BatchedMinimalForward:
    def __init__(self, model: torch.nn.Module, sequence_length: int, batch: int) -> None:
        core = model.model
        attention = core.layers[0].self_attn
        self.model, self.core, self.length, self.batch = model, core, sequence_length, batch
        self.heads, self.kv_heads, self.head_dim = attention.num_heads, attention.num_key_value_heads, attention.head_dim
        device = next(model.parameters()).device
        positions = torch.arange(sequence_length, device=device)[None]
        self.embedded = core.embed_tokens(torch.zeros_like(positions))
        self.cos, self.sin = core.rotary_emb(self.embedded, positions)
        shape = (batch, self.kv_heads, sequence_length, self.head_dim)
        self.keys = [torch.randn(shape, device=device, dtype=self.embedded.dtype) for _ in core.layers]
        self.values = [torch.randn_like(keys) for keys in self.keys]

    @torch.no_grad()
    def __call__(self, rows: int, *, start_layer: int = 0, end_layer: int | None = None,
                 project_head: bool = True, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, groups = self.batch, self.heads // self.kv_heads
        positions = torch.arange(self.length - rows, self.length, device=self.embedded.device)
        hidden = self.embedded[:, positions].expand(batch, -1, -1).contiguous()
        cos, sin = self.cos[:, positions], self.sin[:, positions]
        end_layer = len(self.core.layers) if end_layer is None else end_layer
        mask = None if attention_mask is None else attention_mask.expand(batch, 1, rows, self.length).contiguous()
        for layer, keys, values in zip(self.core.layers[start_layer:end_layer], self.keys[start_layer:end_layer], self.values[start_layer:end_layer]):
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            query = attention.q_proj(normalized).view(batch, rows, self.heads, self.head_dim).transpose(1, 2)
            key = attention.k_proj(normalized).view(batch, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            value = attention.v_proj(normalized).view(batch, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            query, key = _apply_rope(query, key, cos, sin)
            keys[:, :, positions] = key
            values[:, :, positions] = value
            output = F.scaled_dot_product_attention(query, _repeat_kv(keys, groups), _repeat_kv(values, groups), attn_mask=mask)
            hidden = hidden + attention.o_proj(output.transpose(1, 2).reshape(batch, rows, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        if end_layer == len(self.core.layers):
            hidden = self.core.norm(hidden)
        return self.model.lm_head(hidden) if project_head else hidden


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, default=401,
                        help="prompt + generation length; 401 is the mean of the 64 tasks (decoding_oracles)")
    args = parser.parse_args()

    from language_models.diffucoder_loader import load_diffucoder_model_and_tokenizer

    from dlm_kv_policy.code_correctness_and_latency_evaluator import gpu_folder_name
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config

    config = load_kv_policy_experiment_config(args.config)
    model, _ = load_diffucoder_model_and_tokenizer(
        config.model.path, device=config.model.device, dtype=config.model.dtype,
        local_files_only=config.model.local_files_only,
    )
    output_dir = config.run.run_dir / gpu_folder_name() / "batched_row_timing"
    output_dir.mkdir(parents=True, exist_ok=True)
    length = args.sequence_length
    curves: dict[str, dict[str, float]] = {}
    for batch in BATCH_SIZES:
        forward = BatchedMinimalForward(model, length, batch)
        curves[str(batch)] = {}
        for rows in (*ROW_GRID, length):
            curves[str(batch)][str(rows)] = cuda_seconds(lambda: forward(rows), warmup=2, repeats=5)
        full, one = curves[str(batch)][str(length)], curves[str(batch)]["1"]
        print(f"batch {batch}: t(1) {one * 1e3:.1f} ms, t(256) {curves[str(batch)]['256'] * 1e3:.1f} ms, "
              f"t(L={length}) {full * 1e3:.1f} ms, t(256)/t(L) {curves[str(batch)]['256'] / full:.2f}", flush=True)
        del forward
        torch.cuda.empty_cache()
    result = {"gpu": gpu_folder_name(), "sequence_length": length, "curves": curves}
    (output_dir / "timing.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Results: {output_dir / 'timing.json'}")


if __name__ == "__main__":
    main()
