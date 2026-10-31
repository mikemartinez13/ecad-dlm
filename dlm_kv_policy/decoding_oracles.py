"""Speed ceilings for DiffuCoder decoding, and an autoregressive estimate.

Decoding time is the sum over steps of the cost of one step. Caching lowers
the cost of a step; parallel decoding lowers the number of steps.

1. Timing curve t(n): wall time of one forward that recomputes n rows while
   attending to all L positions. Two curves:
   - minimal: only the math a step needs (projections, attention against L
     cached keys, MLP, LM head on n rows), with the same kernels as our
     backend. What a tight implementation could reach.
   - backend: DiffuCoderActiveTokenKVBackend, including its bookkeeping.
   Native dense is DiffuCoder's own full forward on L tokens; it is the
   numerator of every ceiling below.
2. Caching ceiling (speed only): after the first forward, recompute only the
   still-masked rows and reuse everything else forever, ignoring quality.
   With one token per step, step s has gen_length - s masked rows:
       ceiling = steps * t_dense / (t(L) + sum_{s>=1} t(n_masked(s)))
   The first step is a full forward because the cache is empty.
3. Parallel-decoding oracle: decode toward the greedy dense output y. Each
   step commits every masked position whose argmax already equals y; if none
   does, it commits the most confident masked position, with its y token.
   The output equals y by construction. Steps used is the fewest this greedy
   rule needs; it is a strong reference, not a proven minimum.
4. Combined ceiling (approximate): the parallel oracle's steps at the caching
   ceiling's cost. It ignores that fewer steps make cached entries go stale
   faster, so it is optimistic.
5. Autoregressive estimate: an AR model of the same shape with a KV cache
   answers in t(prompt) + N * t(1), with N the non-padding tokens of the dense
   answer. DiffuCoder-7B has Qwen2.5-Coder-7B's shape, so its t(1) is the AR
   per-token cost. autoregressive_reference.py measures the real AR model.

Every curve is measured at one sequence length L (median prompt + gen_length),
so the modeled numbers are for a typical task. The script also measures the
real dense wall time of every task, and the real wall time of the caching
oracle on the first few tasks, to check the model.

Usage (inside a GPU allocation, from the repository root; greedy config):

    python -m dlm_kv_policy.decoding_oracles \\
        --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any

import torch
import torch.nn.functional as F

from ecad.language_models.code_tasks_and_response_processing import load_code_task_dataset
from ecad.language_models.diffucoder_loader import (
    load_diffucoder_model_and_tokenizer,
    prepare_diffucoder_prompt_batch,
)

from .code_correctness_and_latency_evaluator import gpu_folder_name, seed_everything
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .denoising_token_state import DenoisingTokenCategory, DenoisingTokenState
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend, _apply_rope, _repeat_kv
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .fixed_token_refresh_policy import TokenRefreshDecision
from .forced_trajectory_replay import entropy_confidence
from .kv_policy_experiment_config import load_kv_policy_experiment_config


class MaskedRowsOnly:
    """Speed-only caching oracle: after the first forward, recompute only masked rows."""

    name = "caching_oracle_masked_rows_only"
    force_dense = False

    def validate(self) -> None:
        pass

    def decide(self, *, states: torch.Tensor, cache_ready: bool, step: int, **_: Any) -> TokenRefreshDecision:
        valid = states.ne(int(DenoisingTokenCategory.PAD))
        active = states.eq(int(DenoisingTokenCategory.MASK)) if cache_ready and step > 0 else valid
        return TokenRefreshDecision(active=active, refresh_depth=torch.ones_like(states), reasons={})


def cuda_seconds(run, *, warmup: int = 3, repeats: int = 10) -> float:
    """Median wall time of ``run()`` on the GPU."""

    for _ in range(warmup):
        run()
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / 1000)
    return median(times)


class MinimalForward:
    """n rows through every layer, attending to L cached positions, then the LM head."""

    def __init__(self, model: torch.nn.Module, sequence_length: int) -> None:
        core = model.model
        attention = core.layers[0].self_attn
        self.model, self.core, self.length = model, core, sequence_length
        self.heads, self.kv_heads, self.head_dim = attention.num_heads, attention.num_key_value_heads, attention.head_dim
        device = next(model.parameters()).device
        positions = torch.arange(sequence_length, device=device)[None]
        self.embedded = core.embed_tokens(torch.zeros_like(positions))
        self.cos, self.sin = core.rotary_emb(self.embedded, positions)
        shape = (1, self.kv_heads, sequence_length, self.head_dim)
        self.keys = [torch.zeros(shape, device=device, dtype=self.embedded.dtype) for _ in core.layers]
        self.values = [torch.zeros_like(keys) for keys in self.keys]

    @torch.no_grad()
    def __call__(self, rows: int) -> torch.Tensor:
        positions = torch.arange(self.length - rows, self.length, device=self.embedded.device)
        hidden = self.embedded[:, positions]
        cos, sin = self.cos[:, positions], self.sin[:, positions]
        groups = self.heads // self.kv_heads
        for layer, keys, values in zip(self.core.layers, self.keys, self.values):
            attention = layer.self_attn
            normalized = layer.input_layernorm(hidden)
            query = attention.q_proj(normalized).view(1, rows, self.heads, self.head_dim).transpose(1, 2)
            key = attention.k_proj(normalized).view(1, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            value = attention.v_proj(normalized).view(1, rows, self.kv_heads, self.head_dim).transpose(1, 2)
            query, key = _apply_rope(query, key, cos, sin)
            keys[:, :, positions] = key
            values[:, :, positions] = value
            output = F.scaled_dot_product_attention(query, _repeat_kv(keys, groups), _repeat_kv(values, groups))
            hidden = hidden + attention.o_proj(output.transpose(1, 2).reshape(1, rows, -1))
            hidden = hidden + layer.mlp(layer.post_attention_layernorm(hidden))
        return self.model.lm_head(self.core.norm(hidden))


def backend_inputs(model: torch.nn.Module, prompt_length: int, gen_length: int, masked: int) -> dict[str, Any]:
    """A step where the last ``masked`` generated positions are still masked."""

    length = prompt_length + gen_length
    device = next(model.parameters()).device
    ids = torch.zeros((1, length), dtype=torch.long, device=device)
    ids[:, length - masked :] = int(model.config.mask_token_id)
    state = DenoisingTokenState.create(prompt_length, length, None, int(model.config.mask_token_id))
    state.decode_step[:, prompt_length : length - masked] = 0
    return dict(
        input_ids=ids,
        attention_mask="full",
        position_ids=torch.arange(length, device=device)[None],
        state=state,
        total_steps=gen_length,
    )


def timing_curves(model: torch.nn.Module, prompt_length: int, gen_length: int) -> dict[str, Any]:
    """t(n) for n = 1..gen_length and for the full length L, plus native dense."""

    length = prompt_length + gen_length
    minimal = MinimalForward(model, length)
    curves: dict[str, dict[int, float]] = {"minimal": {}, "backend": {}}
    for rows in [*range(1, gen_length + 1), length]:
        curves["minimal"][rows] = cuda_seconds(lambda: minimal(rows))
    backend = DiffuCoderActiveTokenKVBackend(model, MaskedRowsOnly())
    for masked in range(1, gen_length + 1):
        inputs = backend_inputs(model, prompt_length, gen_length, masked)
        backend.start_generation(length)
        backend.forward_mask_logits(step=0, **inputs)  # fills the cache
        curves["backend"][masked] = cuda_seconds(lambda: backend.forward_mask_logits(step=5, **inputs))
    full = backend_inputs(model, prompt_length, gen_length, gen_length)
    curves["backend"][length] = cuda_seconds(lambda: backend.forward_mask_logits(step=0, **full))
    dense = DenseDiffuCoderBaseline(model)
    native_dense = cuda_seconds(lambda: dense.forward_mask_logits(step=0, **full))
    print(f"t(1), t({gen_length}), t(L={length}): " + ", ".join(
        f"{name} {curve[1] * 1e3:.1f}/{curve[gen_length] * 1e3:.1f}/{curve[length] * 1e3:.1f} ms"
        for name, curve in curves.items()
    ) + f"; native dense {native_dense * 1e3:.1f} ms", flush=True)
    return {"prompt_length": prompt_length, "sequence_length": length, "native_dense": native_dense, "curves": curves}


def interpolate(curve: dict[int, float], rows: int) -> float:
    """t(rows), linear between measured points."""

    points = sorted(curve)
    rows = min(rows, points[-1])
    if rows in curve:
        return curve[rows]
    above = next(point for point in points if point > rows)
    below = max(point for point in points if point < rows)
    weight = (rows - below) / (above - below)
    return (1 - weight) * curve[below] + weight * curve[above]


def parallel_decoding_oracle(
    backend: Any,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor | None,
    target: list[int],
    mask_token_id: int,
) -> list[dict[str, Any]]:
    """Commit every masked position whose argmax already equals the target token."""

    gen_length = len(target)
    _, model_attention_mask, position_ids = DiffuCoderDenoisingSampler._attention_inputs(prompt, attention_mask, gen_length)
    x = F.pad(prompt, (0, gen_length), value=mask_token_id)
    wanted_sequence = torch.cat([prompt[0], torch.tensor(target, device=prompt.device)])
    state = DenoisingTokenState.create(prompt.shape[1], x.shape[1], attention_mask, mask_token_id)
    backend.start_generation(x.shape[1])
    steps: list[dict[str, Any]] = []
    while bool(x.eq(mask_token_id).any()):
        positions = torch.where(x[0].eq(mask_token_id))[0]
        logits = backend.forward_mask_logits(
            input_ids=x,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            state=state,
            step=len(steps),
            total_steps=gen_length,
        ).mask_logits
        wanted = wanted_sequence[positions]
        commit = logits.argmax(dim=-1).eq(wanted)
        fallback = not bool(commit.any())
        if fallback:
            commit[entropy_confidence(logits, 0.0, 1.0, None).argmax()] = True
        previous = x.clone()
        x[0, positions[commit]] = wanted[commit]
        state.update_after_sampling(previous, x, len(steps))
        steps.append({"masked": int(positions.numel()), "committed": int(commit.sum()), "fallback": fallback})
    if x[0, prompt.shape[1] :].tolist() != target:
        raise RuntimeError("The parallel-decoding oracle did not reproduce the target")
    return steps


def answer_length(token_ids: list[int], stop_ids: set[int]) -> int:
    """Tokens before the first end-of-text or padding token."""

    return next((index for index, token in enumerate(token_ids) if token in stop_ids), len(token_ids))


def ceilings(timing: dict[str, Any], tasks: list[dict[str, Any]], gen_length: int) -> dict[str, Any]:
    """Modeled speedups over native dense, for each timing curve."""

    length = timing["sequence_length"]
    dense_seconds = len(tasks) * gen_length * timing["native_dense"]
    one_token_per_step = list(range(gen_length - 1, 0, -1))  # masked rows at steps 1..gen_length-1
    oracle_steps = sum(len(task["parallel_oracle_steps"]) for task in tasks)
    # Dense commits one token per step, so this is also tokens per step.
    result: dict[str, Any] = {
        "parallel_decoding": len(tasks) * gen_length / oracle_steps,
        "parallel_oracle_fallback_steps": sum(
            step["fallback"] for task in tasks for step in task["parallel_oracle_steps"]
        ),
    }
    for name, curve in timing["curves"].items():
        t = lambda rows: interpolate(curve, rows)  # noqa: E731
        caching = len(tasks) * (t(length) + sum(t(rows) for rows in one_token_per_step))
        combined = sum(
            t(length) + sum(t(step["masked"]) for step in task["parallel_oracle_steps"][1:]) for task in tasks
        )
        autoregressive = sum(t(task["prompt_length"]) + task["answer_tokens"] * t(1) for task in tasks)
        result[name] = {
            "caching_speed_only": dense_seconds / caching,
            "combined_approximate": dense_seconds / combined,
            "autoregressive_estimate": dense_seconds / autoregressive,
            "t_n_over_t_1": {str(rows): t(rows) / t(1) for rows in (8, 32, 128, gen_length)},
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--measured-oracle-tasks", type=int, default=8,
                        help="Run the caching oracle for real on the first N tasks, to check the timing model")
    args = parser.parse_args()

    config = load_kv_policy_experiment_config(args.config)
    generation = config.generation
    if generation.temperature != 0 or generation.steps != generation.gen_length:
        raise ValueError("The oracles need greedy decoding with one token per step")
    tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
    model, tokenizer = load_diffucoder_model_and_tokenizer(
        config.model.path,
        device=config.model.device,
        dtype=config.model.dtype,
        local_files_only=config.model.local_files_only,
    )
    output_dir = config.run.run_dir / gpu_folder_name() / "decoding_oracles"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir}", flush=True)
    mask_token_id = int(model.config.mask_token_id)
    stop_ids = {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}
    sampler_settings = dict(
        steps=generation.steps,
        gen_length=generation.gen_length,
        temperature=generation.temperature,
        top_p=generation.top_p,
        top_k=generation.top_k,
        algorithm=generation.algorithm,
        algorithm_temperature=generation.algorithm_temperature,
    )
    prompt_lengths = [
        prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], config.model.device)[0].shape[1] for task in tasks
    ]
    timing = timing_curves(model, int(median(prompt_lengths)), generation.gen_length)
    (output_dir / "timing.json").write_text(json.dumps(timing), encoding="utf-8")

    dense_sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    caching_oracle_sampler = DiffuCoderDenoisingSampler(DiffuCoderActiveTokenKVBackend(model, MaskedRowsOnly()))

    task_results = []
    for index, task in enumerate(tasks):
        seed_everything(generation.seed + index)
        prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], config.model.device)
        dense = dense_sampler.generate(prompt, attention_mask=attention_mask, **sampler_settings)
        target = dense.sequences[0, prompt.shape[1] :].tolist()
        record = {
            "task_id": task.task_id,
            "prompt_length": prompt.shape[1],
            "answer_tokens": answer_length(target, stop_ids),
            "dense_token_ids": target,
            "dense_seconds": dense.elapsed_seconds,
            "parallel_oracle_steps": parallel_decoding_oracle(
                dense_sampler.backend, prompt, attention_mask, target, mask_token_id
            ),
        }
        if index < args.measured_oracle_tasks:
            record["caching_oracle_seconds"] = caching_oracle_sampler.generate(
                prompt, attention_mask=attention_mask, **sampler_settings
            ).elapsed_seconds
        task_results.append(record)
        print(f"task {index + 1}/{len(tasks)} {task.task_id}: {len(record['parallel_oracle_steps'])} oracle steps", flush=True)
    (output_dir / "tasks.json").write_text(json.dumps(task_results), encoding="utf-8")

    measured = task_results[: args.measured_oracle_tasks]
    summary = {
        "gpu": gpu_folder_name(),
        "task_count": len(task_results),
        "sequence_length": timing["sequence_length"],
        "native_dense_step_seconds": timing["native_dense"],
        "measured_dense_seconds_per_task": sum(task["dense_seconds"] for task in task_results) / len(task_results),
        "measured_caching_oracle_speedup": (
            sum(task["dense_seconds"] for task in measured) / sum(task["caching_oracle_seconds"] for task in measured)
        ),
        "mean_answer_tokens": sum(task["answer_tokens"] for task in task_results) / len(task_results),
        "ceilings": ceilings(timing, task_results, generation.gen_length),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    result = summary["ceilings"]
    print(f"\nSpeedups over native dense ({summary['gpu']}, L = {summary['sequence_length']})")
    print(f"parallel decoding (fewer steps):   {result['parallel_decoding']:.2f}x "
          f"(= tokens per step; {result['parallel_oracle_fallback_steps']} fallback steps)")
    print(f"{'':34} {'minimal':>8} {'backend':>8}")
    for key, label in (
        ("caching_speed_only", "caching, speed only"),
        ("combined_approximate", "combined, approximate"),
        ("autoregressive_estimate", "autoregressive estimate"),
    ):
        print(f"{label:<34} {result['minimal'][key]:>7.2f}x {result['backend'][key]:>7.2f}x")
    print(f"measured caching oracle (backend, first {len(measured)} tasks): "
          f"{summary['measured_caching_oracle_speedup']:.2f}x")


if __name__ == "__main__":
    main()
