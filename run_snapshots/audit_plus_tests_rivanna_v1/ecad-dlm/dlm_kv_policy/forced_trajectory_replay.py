"""Experiment E4: forced-trajectory replay of cached KV policies.

For each task the dense model generates as usual. At every step, each cached
policy is shown the exact sequence the dense model saw at that step. Each
policy keeps its own cache along that path: rows it does not recompute keep
the values it last computed, so stale entries accumulate on the policy's own
refresh schedule. Only the token sequence is forced, so one early mistake
cannot cascade into different tokens. Each step is a separate measurement.

At the position the dense model commits in each step, it records:

- token_tv: total variation distance between the dense and cached decoding
  distributions (after temperature and top-p). Under greedy decoding it is 1
  if the token changes, else 0. TV is the minimum chance two samplers differ,
  so it is a lower bound.
- kl: KL(dense || cached) of the temperature-1 next-token distributions
- top1_flip: the cached argmax token differs from the dense argmax token
- position_flip: the cached confidences would commit a different position
- dense_margin: dense top-1 minus top-2 probability (small means a near-tie)
- dense_position_margin: dense confidence of the committed position minus the
  best other masked position (small means the commit order was a near-tie)

It also records the mean and max KL over every still-masked position, and each
policy's FLOPs, so the summary reports compute_fraction (policy FLOPs / dense
FLOPs) next to the drift.

The search objective is divergence_events_per_task: per step, token_tv plus 1
if the committed position changes, summed per task and averaged over tasks. A
position change only counts when dense's own position margin is at least
POSITION_TIE_MARGIN. Below that, dense was choosing between tied positions and
any rounding difference breaks the tie the other way, usually without changing
the final text (see PROXY_VALIDATION.md). The summary recomputes
this from the saved rows, so changing the margin needs no new replay.

The dense trajectory uses the configured temperature and seeds. It is
therefore the same output the dense baseline produces for that config.
Include a policy with ``cache_reloading_step: 1`` to measure how far the
cached backend is from native dense when nothing is reused.

Usage (inside a GPU allocation, from the repository root):

    python -m dlm_kv_policy.forced_trajectory_replay --config CONFIG.yaml \\
        --policies full_refresh_every_step,dkv_reference,depth_22 [--max-tasks 4]
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F

from language_models.code_tasks_and_response_processing import load_code_task_dataset
from language_models.diffucoder_loader import (
    load_diffucoder_model_and_tokenizer,
    prepare_diffucoder_prompt_batch,
)

from .code_correctness_and_latency_evaluator import gpu_folder_name, output_hash, seed_everything
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .denoising_token_state import DenoisingTokenState
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .kv_policy_experiment_config import load_kv_policy_experiment_config

# Dense probability margin below which a top-1 flip counts as a near-tie.
NEAR_TIE_MARGIN = 0.1
# Dense confidence margin below which a position flip is a tie-break and does
# not count. None counts every position flip (the proxy as first registered).
POSITION_TIE_MARGIN = 0.01
# Tokens in the Expected Acceptance Rate, as in Helcig, Kurtic and Alistarh (2026).
EAR_TOP_K = 10


from .entropy_confidence import entropy_confidence


def decoding_distribution(logits: torch.Tensor, temperature: float, top_p: float, top_p_logits) -> torch.Tensor:
    """The distribution DiffuCoder samples a token from; one-hot argmax when greedy."""

    if temperature == 0:
        return F.one_hot(logits.argmax(dim=-1), logits.shape[-1]).float()
    logits = logits.float() / temperature
    if top_p < 1:
        logits = top_p_logits(logits, top_p)
    return torch.softmax(logits, dim=-1)


def expected_acceptance(dense_log_probs: torch.Tensor, cached_log_probs: torch.Tensor) -> torch.Tensor:
    """Per-row EAR (their Eq. 3): the sum of min(p, q) over dense's top tokens.

    p and q stay full-vocabulary probabilities at temperature 1, not renormalized
    over the top tokens, so a row is at most 1 - TV(p, q) plus the excluded tail.
    """

    values, tokens = dense_log_probs.topk(min(EAR_TOP_K, dense_log_probs.shape[-1]), dim=-1)
    return torch.minimum(values, cached_log_probs.gather(-1, tokens)).exp().sum(dim=-1)


def step_metrics(
    dense_logits: torch.Tensor,
    cached_logits: torch.Tensor,
    committed_rows: torch.Tensor,
    cached_confidence: torch.Tensor,
    *,
    dense_confidence: torch.Tensor | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_p_logits=None,
    answer_rows: torch.Tensor | None = None,
) -> dict[str, float | bool]:
    """Compare dense and cached logits for one step.

    Rows of every tensor are the still-masked positions in sequence order.
    ``committed_rows`` are the rows the dense trajectory unmasks this step.
    ``answer_rows``, if given, marks the rows inside dense's final answer.
    """

    dense_log_probs = F.log_softmax(dense_logits.float(), dim=-1)
    cached_log_probs = F.log_softmax(cached_logits.float(), dim=-1)
    kl_per_row = (dense_log_probs.exp() * (dense_log_probs - cached_log_probs)).sum(dim=-1)
    ear_per_row = expected_acceptance(dense_log_probs, cached_log_probs)

    dense_top2 = dense_log_probs[committed_rows].exp().topk(2, dim=-1).values
    dense_margin = (dense_top2[:, 0] - dense_top2[:, 1]).min()
    top1_flip = (
        dense_logits[committed_rows].argmax(dim=-1) != cached_logits[committed_rows].argmax(dim=-1)
    ).any()
    # A position flip means the cached confidences rank a committed position
    # strictly below some other position. Ties count as agreement because
    # bf16 confidences often tie exactly.
    kth_best = cached_confidence.topk(committed_rows.numel()).values.min()
    position_flip = bool((cached_confidence[committed_rows].min() < kth_best).item())
    dense_position_margin = None
    others = torch.ones_like(cached_confidence, dtype=torch.bool)
    others[committed_rows] = False
    if dense_confidence is not None and others.any():
        dense_position_margin = (
            dense_confidence[committed_rows].min() - dense_confidence[others].max()
        ).float().item()

    dense_decoding = decoding_distribution(dense_logits[committed_rows], temperature, top_p, top_p_logits)
    cached_decoding = decoding_distribution(cached_logits[committed_rows], temperature, top_p, top_p_logits)
    token_tv = 0.5 * (dense_decoding - cached_decoding).abs().sum(dim=-1).sum().item()
    return {
        "ear": ear_per_row[committed_rows].tolist(),
        "ear_masked_answer": (
            ear_per_row[answer_rows].mean().item() if answer_rows is not None and answer_rows.any() else None
        ),
        "token_tv": token_tv,
        "kl": kl_per_row[committed_rows].mean().item(),
        "kl_masked_mean": kl_per_row.mean().item(),
        "kl_masked_max": kl_per_row.max().item(),
        "top1_flip": bool(top1_flip.item()),
        "position_flip": position_flip,
        "dense_margin": dense_margin.item(),
        "dense_position_margin": dense_position_margin,
    }


def counts_as_position_change(row: dict[str, Any], position_tie_margin: float | None = POSITION_TIE_MARGIN) -> bool:
    """A position flip, unless dense was choosing between tied positions."""

    if not row["position_flip"]:
        return False
    return position_tie_margin is None or row["dense_position_margin"] >= position_tie_margin


def step_divergence(row: dict[str, Any], position_tie_margin: float | None = POSITION_TIE_MARGIN) -> float:
    """One step's contribution to divergence_events_per_task."""

    return row["token_tv"] + float(counts_as_position_change(row, position_tie_margin))


def large_margin_decision_changes(
    rows: list[dict[str, Any]], position_tie_margin: float | None = POSITION_TIE_MARGIN
) -> int:
    """Steps where the cached model changes a decision the dense model was sure of.

    That is a top-1 flip at a dense margin of at least NEAR_TIE_MARGIN, or a
    counted change of committed position.
    """

    return sum(
        (row["top1_flip"] and row["dense_margin"] >= NEAR_TIE_MARGIN)
        or counts_as_position_change(row, position_tie_margin)
        for row in rows
    )


def load_top_p_logits(model: torch.nn.Module):
    """DiffuCoder's own top-p filter, from the generation_utils next to its modeling code."""

    return importlib.import_module(
        model.__class__.__module__.replace("modeling_dream", "generation_utils")
    ).top_p_logits


def replay_task(
    model: torch.nn.Module,
    sampler: DiffuCoderDenoisingSampler,
    cached_backends: dict[str, DiffuCoderActiveTokenKVBackend],
    prompt: torch.Tensor,
    attention_mask: torch.Tensor,
    generation: Any,
    top_p_logits,
    on_step: Callable[[int, torch.Tensor, DenoisingTokenState], None] | None = None,
    answer_end: int | None = None,
) -> dict[str, Any]:
    """Generate one task densely while scoring every cached policy at every step.

    ``on_step(step, x, state)``, if given, runs after every backend's forward
    and before the dense commit. ``answer_end``, if given, is the first generated
    position after dense's answer, known from an earlier run of the same task.
    """

    mask_token_id = int(model.config.mask_token_id)
    _, model_attention_mask, position_ids = sampler._attention_inputs(
        prompt, attention_mask, generation.gen_length
    )
    x = F.pad(prompt, (0, generation.gen_length), value=mask_token_id)
    state = DenoisingTokenState.create(prompt.shape[1], x.shape[1], attention_mask, mask_token_id)
    timesteps = torch.linspace(1, 1e-12, generation.steps + 1, device=x.device)
    for backend in cached_backends.values():
        backend.start_generation(x.shape[1])
    flops_before = {name: backend.flops for name, backend in cached_backends.items()}
    dense_flops_before = sampler.backend.flops
    per_policy: dict[str, list[dict[str, Any]]] = {name: [] for name in cached_backends}

    for step in range(generation.steps):
        forward_inputs = dict(
            input_ids=x,
            attention_mask=model_attention_mask,
            position_ids=position_ids,
            state=state,
            step=step,
            total_steps=generation.steps,
        )
        mask_index = x.eq(mask_token_id)
        dense_logits = sampler.backend.forward_mask_logits(**forward_inputs).mask_logits
        cached = {name: backend.forward_mask_logits(**forward_inputs) for name, backend in cached_backends.items()}
        if on_step is not None:
            on_step(step, x, state)

        # Advance the dense trajectory exactly as the normal sampler would.
        previous = x.clone()
        sampler.commit_tokens(
            x,
            mask_index,
            dense_logits,
            step=step,
            steps=generation.steps,
            timesteps=timesteps,
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            algorithm=generation.algorithm,
            algorithm_temperature=generation.algorithm_temperature,
        )
        state.update_after_sampling(previous, x, step)
        committed_rows = torch.where(x[mask_index].ne(mask_token_id))[0]
        if committed_rows.numel() == 0:
            continue

        dense_confidence = entropy_confidence(dense_logits, generation.temperature, generation.top_p, top_p_logits)
        positions = mask_index[0].nonzero()[:, 0] - prompt.shape[1]
        answer_rows = None if answer_end is None else positions < answer_end
        for name, result in cached.items():
            confidence = entropy_confidence(result.mask_logits, generation.temperature, generation.top_p, top_p_logits)
            if result.candidates is not None:
                # Positions the policy did not recompute cannot be committed this step.
                confidence = confidence.masked_fill(~result.candidates, -torch.inf)
            metrics = step_metrics(
                dense_logits,
                result.mask_logits,
                committed_rows,
                confidence,
                dense_confidence=dense_confidence,
                temperature=generation.temperature,
                top_p=generation.top_p,
                top_p_logits=top_p_logits,
                answer_rows=answer_rows,
            )
            metrics["committed_positions"] = positions[committed_rows].tolist()
            if result.candidates is not None:
                # Generated positions the dense model committed outside the candidates.
                outside = committed_rows[~result.candidates[committed_rows]]
                metrics["outside_positions"] = positions[outside].tolist()
            per_policy[name].append({"step": step, **metrics})

    generated = x[0, prompt.shape[1] :].tolist()
    return {
        "dense_token_ids": generated,
        "dense_flops": sampler.backend.flops - dense_flops_before,
        "flops": {name: backend.flops - flops_before[name] for name, backend in cached_backends.items()},
        "policies": per_policy,
    }


def replay_policies(
    model: torch.nn.Module,
    tokenizer: Any,
    tasks: list[Any],
    policies: dict[str, Any],
    generation: Any,
    device: str,
    task_dir: Path,
) -> list[dict[str, Any]]:
    """Replay every task for ``policies``; finished tasks in ``task_dir`` are reused."""

    if generation.algorithm != "entropy" or generation.algorithm_temperature != 0:
        raise ValueError("Replay supports DiffuCoder's entropy ordering with algorithm_temperature 0")
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)
    cached_backends = {name: DiffuCoderActiveTokenKVBackend(model, policy) for name, policy in policies.items()}

    task_dir.mkdir(parents=True, exist_ok=True)
    task_results = []
    for index, task in enumerate(tasks):
        path = task_dir / f"task_{index:03d}.json"
        if path.is_file():
            saved = json.loads(path.read_text())
            if set(policies) <= set(saved["policies"]):
                task_results.append(saved)
                continue
        # Same per-task seed as the normal evaluator, so the dense trajectory
        # matches the dense baseline for this config.
        seed_everything(generation.seed + index)
        prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], device)
        result = replay_task(model, sampler, cached_backends, prompt, attention_mask, generation, top_p_logits)
        result["task_id"] = task.task_id
        path.write_text(json.dumps(result), encoding="utf-8")
        task_results.append(result)
        print(f"task {index + 1}/{len(tasks)} {task.task_id} done", flush=True)
    return task_results


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(fraction * (len(ordered) - 1))]


def summarize(
    task_results: list[dict[str, Any]],
    policy_names: list[str],
    position_tie_margin: float | None = POSITION_TIE_MARGIN,
) -> dict[str, Any]:
    summary = {}
    dense_flops = sum(task["dense_flops"] for task in task_results)
    for name in policy_names:
        per_task = [task["policies"][name] for task in task_results]
        steps = [row for rows in per_task for row in rows]
        kl = [row["kl"] for row in steps]
        flip_margins = [row["dense_margin"] for row in steps if row["top1_flip"]]
        summary[name] = {
            "divergence_events_per_task": (
                sum(step_divergence(row, position_tie_margin) for row in steps) / len(per_task)
            ),
            "compute_fraction": sum(task["flops"][name] for task in task_results) / dense_flops,
            "large_margin_decision_changes_per_task": (
                sum(large_margin_decision_changes(rows, position_tie_margin) for rows in per_task) / len(per_task)
            ),
            "measured_steps": len(steps),
            "kl_mean": sum(kl) / len(kl),
            "kl_p99": percentile(kl, 0.99),
            "kl_max": max(kl),
            "kl_masked_mean": sum(row["kl_masked_mean"] for row in steps) / len(steps),
            "top1_flips": len(flip_margins),
            "position_flips": sum(row["position_flip"] for row in steps),
            "position_flips_counted": sum(counts_as_position_change(row, position_tie_margin) for row in steps),
            "top1_flips_at_near_ties": sum(margin < NEAR_TIE_MARGIN for margin in flip_margins),
            "top1_flip_dense_margins": sorted(flip_margins),
            "tasks_with_any_top1_flip": sum(any(row["top1_flip"] for row in rows) for rows in per_task),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policies", help="Comma-separated names under dkv_policies; default is all of them")
    parser.add_argument("--max-tasks", type=int, help="Replay only the first N tasks (for a quick check)")
    parser.add_argument("--output-name", default="forced_trajectory_replay")
    args = parser.parse_args()

    config = load_kv_policy_experiment_config(args.config)
    policy_names = args.policies.split(",") if args.policies else list(config.dkv_policies)
    unknown = sorted(set(policy_names) - set(config.dkv_policies))
    if unknown:
        raise ValueError(f"Unknown dkv_policies: {unknown}")

    tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
    if args.max_tasks:
        tasks = tasks[: args.max_tasks]
    model, tokenizer = load_diffucoder_model_and_tokenizer(
        config.model.path,
        device=config.model.device,
        dtype=config.model.dtype,
        local_files_only=config.model.local_files_only,
    )
    # Results go to <run.name>/<GPU type>/ so GPU types never mix.
    output_dir = config.run.run_dir / gpu_folder_name() / args.output_name
    print(f"Results: {output_dir}", flush=True)
    task_results = replay_policies(
        model,
        tokenizer,
        tasks,
        {name: config.dkv_policies[name] for name in policy_names},
        config.generation,
        config.model.device,
        output_dir / "tasks",
    )

    summary = {
        "policies": {name: config.dkv_policies[name].to_dict() for name in policy_names},
        "generation": dict(config.generation.__dict__),
        "task_count": len(task_results),
        "dense_output_hash": output_hash([task["dense_token_ids"] for task in task_results]),
        "near_tie_margin": NEAR_TIE_MARGIN,
        "position_tie_margin": POSITION_TIE_MARGIN,
        "results": summarize(task_results, policy_names),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"{'policy':<28} {'compute':>8} {'diverge/task':>12} {'big changes':>11} "
        f"{'KL p99':>9} {'top1 flips':>10} {'pos flips':>9} {'counted':>8}"
    )
    for name, row in summary["results"].items():
        print(
            f"{name:<28} {row['compute_fraction']:>8.3f} {row['divergence_events_per_task']:>12.2f} "
            f"{row['large_margin_decision_changes_per_task']:>11.2f} {row['kl_p99']:>9.2e} "
            f"{row['top1_flips']:>10} {row['position_flips']:>9} {row['position_flips_counted']:>8}"
        )


if __name__ == "__main__":
    main()
