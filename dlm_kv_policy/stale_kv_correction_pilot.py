"""Exploratory pilot: does subtracting the fitted stale K/V error change decisions?

The stale K/V error study fit M4, a per-cell affine map e_hat = A c + b, that
predicts how a stale cached row differs from dense. This pilot adds
strength * e_hat to every stale row before attention reads it, in layers
1-27 (layer 0 has only rounding), and measures the forced-trajectory
divergence events against the same policy without the correction. The cache
itself is never changed, and every task uses the fit of its own held-out
fold. Strength 0.5 separates "does not help" from "over-corrects".

Any GPU may run it: every comparison is between runs of one task on one GPU.
Data are read from, and results written to, the study GPU's folder; each task
records the GPU it ran on. Strength 0 runs the correction path with nothing
added, so it must match the uncorrected run exactly on every GPU. On the
study GPU, the uncorrected run must also reproduce the earlier replay.

Row types are computed as a decoder could: prompt by position, padding if the
committed token is a stop token, otherwise answer. On the 64 tasks this
agrees with the study's definition (first stop token onward) on every
committed row. The success criterion is in STALE_KV_ERROR_STUDY.md.

  run --fold K --part P
                 GPU. Replays half of fold K's tasks, corrected and uncorrected.
  report         Paired differences per policy and strength. No model.

Usage (from the repository root; run inside a GPU allocation):

    python -m dlm_kv_policy.stale_kv_correction_pilot run \\
        --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml --fold 0 --part 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .code_correctness_and_latency_evaluator import gpu_folder_name
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend, _rotate_half
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .forced_trajectory_replay import (
    large_margin_decision_changes,
    load_top_p_logits,
    replay_task,
    step_divergence,
)
from .kv_policy_experiment_config import load_kv_policy_experiment_config
from .stale_kv_error_study import (
    AGE_EDGES,
    BOOTSTRAP_RESAMPLES,
    FOLDS,
    PARTS,
    POLICIES,
    ROW_TYPES,
    SCORED_FIRST_LAYER,
    STUDY_GPU,
    PreRopeKV,
    Study,
    same_steps,
)

STRENGTHS = (0.5, 1.0)
PRIMARY_POLICY = "refresh_256"
# Two strengths are tested on the primary policy, so each interval is 97.5% (Bonferroni).
INTERVAL = 0.975
# A pipeline check, not a tested strength: must equal the uncorrected run exactly.
CHECK_STRENGTH = 0.0


def pilot_dir(config: Any, gpu_folder: str) -> Path:
    return config.run.run_dir / gpu_folder / "stale_kv_correction_pilot"


class StaleRowCorrection:
    """A backend's kv_correction: adds strength * (A c + b) to each stale row, layers 1 and up.

    c is the row's pre-RoPE K or V as last computed. Keys in the cache are
    after RoPE, which is linear, so the key correction is rotated too.
    """

    def __init__(self, backend: DiffuCoderActiveTokenKVBackend, fit: dict[str, torch.Tensor], strength: float, stop_ids: set[int]) -> None:
        self.backend = backend
        self.store = PreRopeKV(backend)
        self.matrix = fit["m4_matrix"]  # [layers, kinds, heads, combos, head_dim, head_dim]
        self.intercept = fit["m4_intercept"]  # [layers, kinds, heads, combos, head_dim]
        self.strength = strength
        self.edges = torch.tensor(AGE_EDGES, device=self.matrix.device)
        self.stop_ids = torch.tensor(sorted(stop_ids), device=self.matrix.device)
        self.prompt_length = 0
        backend.kv_correction = self

    def start(self, prompt_length: int, length: int) -> None:
        self.prompt_length = prompt_length
        self.store.reset(length)

    def row_combos(self, positions: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
        tokens = self.backend._input_ids[0, positions]
        types = torch.full_like(positions, ROW_TYPES.index("answer"))
        types[torch.isin(tokens, self.stop_ids)] = ROW_TYPES.index("padding")
        types[positions < self.prompt_length] = ROW_TYPES.index("prompt")
        return torch.bucketize(ages, self.edges, right=True) * len(ROW_TYPES) + types

    def __call__(self, layer: int, keys: torch.Tensor, values: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        if layer < SCORED_FIRST_LAYER:
            return keys, values
        step = self.backend._step
        last = self.backend.cache.layer_last_refresh[layer, 0]
        positions = torch.where(last.ge(0) & last.lt(step))[0]
        if positions.numel() == 0:
            return keys, values
        combo = self.row_combos(positions, step - last[positions])
        corrected = []
        for kind, (cached, pre_rope) in enumerate(((keys, self.store.keys[layer]), (values, self.store.values[layer]))):
            c = pre_rope[:, positions].float()  # [heads, rows, head_dim]
            e_hat = torch.einsum("hnoc,hnc->hno", self.matrix[layer, kind][:, combo], c) + self.intercept[layer, kind][:, combo]
            if kind == 0:
                e_hat = e_hat * cos[0, positions] + _rotate_half(e_hat) * sin[0, positions]
            adjusted = cached.clone()
            adjusted[0, :, positions] = (cached[0, :, positions].float() + self.strength * e_hat).to(cached.dtype)
            corrected.append(adjusted)
        return corrected[0], corrected[1]


def task_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "divergence_events": sum(step_divergence(row) for row in rows),
        "top1_flips": sum(row["top1_flip"] for row in rows),
        "large_margin_decision_changes": large_margin_decision_changes(rows),
        "token_tv": sum(row["token_tv"] for row in rows),
        "kl_mean": float(np.mean([row["kl"] for row in rows])) if rows else 0.0,
    }


def run(config: Any, fold: int, part: int) -> None:
    study = Study(config, STUDY_GPU)
    gpu = gpu_folder_name()
    study.write_job_info("pilot", fold, part)
    output_dir = pilot_dir(config, STUDY_GPU)
    device = config.model.device
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(study.model))
    top_p_logits = load_top_p_logits(study.model)
    backends, corrections = {}, []
    for name in POLICIES:
        with np.load(study.output_dir / "fit" / name / f"fold{fold}.npz") as saved:
            fit = {key: torch.from_numpy(saved[key]).to(device) for key in ("m4_matrix", "m4_intercept")}
        policy = config.dkv_policies[name]
        backends[name] = DiffuCoderActiveTokenKVBackend(study.model, policy)
        for strength in (CHECK_STRENGTH, *STRENGTHS):
            backend = DiffuCoderActiveTokenKVBackend(study.model, policy)
            corrections.append(StaleRowCorrection(backend, fit, strength, study.stop_ids))
            backends[f"{name}@{strength}"] = backend

    indices = study.part_tasks(fold, part)
    for count, index in enumerate(indices):
        prompt, attention_mask, _, saved = study.task_inputs(index)
        for correction in corrections:
            correction.start(prompt.shape[1], prompt.shape[1] + config.generation.gen_length)
        result = replay_task(study.model, sampler, backends, prompt, attention_mask, config.generation, top_p_logits)
        steps = result["policies"]
        dense_matches = result["dense_token_ids"] == saved["dense_token_ids"]
        if gpu == STUDY_GPU and not dense_matches:
            raise RuntimeError(f"Task {index}: dense output differs from the earlier replay")
        # Without a correction the backend is unchanged, so on the study GPU it must reproduce the earlier replay.
        reproduces = {name: same_steps(steps[name], saved["policies"][name]) for name in POLICIES}
        zero_matches = {name: same_steps(steps[name], steps[f"{name}@{CHECK_STRENGTH}"]) for name in POLICIES}
        record = {
            "task_id": study.tasks[index].task_id,
            "fold": fold,
            "gpu": gpu,
            "dense_matches_earlier_replay": dense_matches,
            "uncorrected_reproduces_earlier_replay": reproduces,
            "strength_zero_matches_uncorrected": zero_matches,
            "metrics": {name: task_metrics(rows) for name, rows in result["policies"].items()},
            "steps": result["policies"],
        }
        path = output_dir / "tasks" / f"task_{index:03d}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record), encoding="utf-8")
        events = " ".join(f"{name} {m['divergence_events']:.1f}" for name, m in record["metrics"].items())
        print(
            f"fold {fold} part {part} on {gpu}: task {count + 1}/{len(indices)} {record['task_id']}: {events}; "
            f"strength 0 matches {zero_matches}; reproduces {reproduces}",
            flush=True,
        )


def paired_interval(values: np.ndarray, rng: np.random.Generator, level: float) -> tuple[float, float]:
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[samples].mean(axis=1)
    tail = 100 * (1 - level) / 2
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def report(output_dir: Path, task_count: int) -> dict[str, Any]:
    records = [json.loads(path.read_text()) for path in sorted((output_dir / "tasks").glob("task_*.json"))]
    if len(records) != task_count:
        raise SystemExit(f"{len(records)} piloted tasks, expected {task_count}. Finish run first.")
    rng = np.random.default_rng(0)
    metrics = tuple(records[0]["metrics"][POLICIES[0]])
    rows, comparisons = [], {}
    for name in POLICIES:
        for strength in STRENGTHS:
            corrected = f"{name}@{strength}"
            entry = {}
            for metric in metrics:
                base = np.array([record["metrics"][name][metric] for record in records], dtype=float)
                change = np.array([record["metrics"][corrected][metric] for record in records], dtype=float) - base
                low, high = paired_interval(change, rng, INTERVAL)
                entry[metric] = {
                    "uncorrected_mean": float(base.mean()),
                    "corrected_mean": float(base.mean() + change.mean()),
                    "change_mean": float(change.mean()),
                    "interval_low": low,
                    "interval_high": high,
                    "relative_change": float(change.mean() / base.mean()) if base.mean() else float("nan"),
                }
                rows.append({"policy": name, "strength": strength, "metric": metric, **entry[metric]})
            comparisons[corrected] = entry

    primary = {strength: comparisons[f"{PRIMARY_POLICY}@{strength}"]["divergence_events"] for strength in STRENGTHS}
    helps = [strength for strength, entry in primary.items() if entry["interval_high"] < 0]
    harms = [strength for strength, entry in primary.items() if entry["interval_low"] > 0]
    zero_matches = all(all(record["strength_zero_matches_uncorrected"].values()) for record in records)
    on_study_gpu = [record for record in records if record["gpu"] == STUDY_GPU]
    reproduces = all(
        record["dense_matches_earlier_replay"] and all(record["uncorrected_reproduces_earlier_replay"].values())
        for record in on_study_gpu
    )
    gpus = {gpu: sum(record["gpu"] == gpu for record in records) for gpu in sorted({record["gpu"] for record in records})}
    summary = {
        "success": bool(helps),
        "strengths_that_reduce_divergence": helps,
        "strengths_that_increase_divergence": harms,
        "primary_policy": PRIMARY_POLICY,
        "interval": INTERVAL,
        "pipeline_passed": zero_matches and reproduces,
        "strength_zero_matches_uncorrected": zero_matches,
        "uncorrected_reproduces_earlier_replay_on_study_gpu": reproduces,
        "tasks_per_gpu": gpus,
        "comparisons": comparisons,
    }
    (output_dir / "pilot_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "pilot_results.csv").open("w", encoding="utf-8") as handle:
        handle.write(",".join(rows[0]) + "\n")
        handle.writelines(",".join(str(value) for value in row.values()) + "\n" for row in rows)

    print(f"{'policy':<12} {'strength':>8} {'metric':<30} {'uncorrected':>11} {'change':>8} {f'{INTERVAL:.1%} interval':>20}")
    for row in rows:
        print(
            f"{row['policy']:<12} {row['strength']:>8} {row['metric']:<30} {row['uncorrected_mean']:>11.2f} "
            f"{row['change_mean']:>8.2f} [{row['interval_low']:>8.2f}, {row['interval_high']:>8.2f}]"
        )
    print(f"Tasks per GPU: {gpus}")
    print(f"Strength 0 matches uncorrected: {zero_matches}; uncorrected reproduces the earlier replay on {STUDY_GPU}: {reproduces}")
    print(f"Success ({PRIMARY_POLICY} divergence events fall at some strength): {summary['success']}; harmful strengths: {harms}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("run", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Required for run")
    parser.add_argument("--part", type=int, choices=range(PARTS), help="Required for run")
    args = parser.parse_args()
    config = load_kv_policy_experiment_config(args.config)
    if args.step == "run":
        if args.fold is None or args.part is None:
            parser.error("run needs --fold and --part")
        run(config, args.fold, args.part)
    else:
        report(pilot_dir(config, STUDY_GPU), config.evaluation.max_prompts)


if __name__ == "__main__":
    main()
