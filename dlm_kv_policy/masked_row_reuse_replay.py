"""Steps 2 and 3 of the masked-row reuse study: does reusing masked rows change decisions?

Forced-trajectory replay (forced_trajectory_replay.py) of a base policy
(``--base``: refresh_256 for Step 2, depth_22 for Step 3) and of the base with
masked-row reuse. Every ``window`` steps all masked rows are recomputed, as
the base does every step. In between, only the protected positions P (the rule
frozen from Step 0) and the rows that predict them are recomputed: DiffuCoder
reads position p's logits from row p-1. The other masked rows keep their K/V,
hidden states and logits from their last refresh, and their positions cannot
be committed. Prompt and decoded rows are handled exactly as in the base. The
design, amendments and decision rules are in MASKED_ROW_REUSE_STUDY.md.

  run      GPU. Replays the tasks of one part (task index % PARTS); finished
           tasks are skipped. A reuse policy with window 1 is a pipeline check
           and must equal the base exactly. On STUDY_GPU, the dense output and
           the base must also reproduce the earlier replay.
  report   No model. Paired comparisons with the base, live coverage of
           commits, compute fractions, projected batched speedups, and the
           decisions (Step 2: registered rule and Amendment 2).
  generate GPU. End-to-end greedy generation of dense, the base and the
           END_TO_END policies, scored with EvalPlus base tests.
           All variants run in one job, so they share a GPU type.

Replay results go to <run>/NVIDIA_A100-SXM4-80GB/masked_row_reuse_replay[_<base>]/,
whatever GPU ran them; each task records its GPU. Generation results go to
<run>/NVIDIA_A100-SXM4-80GB/masked_row_reuse_generation[_<base>]/<GPU type>/.
The suffix is left out for refresh_256.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .commit_position_study import PADDING_TOP
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .denoising_token_state import DenoisingTokenCategory
from .fixed_token_refresh_policy import TokenRefreshDecision
from .forced_trajectory_replay import counts_as_position_change, entropy_confidence

CHECK = "reuse_check_r1"
M = 32
# Base policy -> {name: (rule, m, window)}. split+recent is the primary rule,
# split the one the plan named. Step 3 (depth_22) tests only the windows that
# passed Step 2 under Amendment 2.
REUSE_POLICIES = {
    "refresh_256": {
        CHECK: ("split+recent", M, 1),
        **{f"split+recent_m{M}_r{window}": ("split+recent", M, window) for window in (2, 4, 8)},
        **{f"split_m{M}_r{window}": ("split", M, window) for window in (2, 4, 8)},
    },
    "depth_22": {
        CHECK: ("split+recent", M, 1),
        **{f"split+recent_m{M}_r{window}": ("split+recent", M, window) for window in (2, 4)},
    },
}
END_TO_END = (f"split+recent_m{M}_r2", f"split+recent_m{M}_r4")
PARTS = 4
STUDY_GPU = "NVIDIA_A100-SXM4-80GB"
# Decision rule (MASKED_ROW_REUSE_STUDY.md).
MAX_COMPUTE_FRACTION = 0.22
COVERAGE_TARGET = 0.99
INTERVAL = 0.95
SPEEDUP_INDICATOR = 1 / 0.85
# Amendment 2: criterion 2 as non-inferiority. The margin is the 95% half-width
# that GPU rounding alone gives a 64-task mean: refresh_256 on A40 against the
# A100 80GB replay, per-task SD 2.44 events (48 tasks of the 2026-11-02 pilot).
NONINFERIORITY_MARGIN = 1.96 * 2.44 / 64**0.5
# Amendment 3 (Step 3): reuse on a base must cost at most this share of the base's compute.
MAX_SHARE_OF_BASE = 0.85


def suffix(base: str) -> str:
    return "" if base == "refresh_256" else f"_{base}"


def reuse_dir(config: Any, base: str) -> Path:
    return config.run.run_dir / STUDY_GPU / f"masked_row_reuse_replay{suffix(base)}"


# Re-export for existing experiment scripts and tests.
from .masked_row_reuse_policy import MaskedRowReusePolicy, dilate, top_positions


class MaskedRowReuseBackend(DiffuCoderActiveTokenKVBackend):
    """The active-token backend for a MaskedRowReusePolicy: marks committable rows, counts rows per step."""

    def start_generation(self, sequence_length: int, **kwargs: Any) -> None:
        super().start_generation(sequence_length, **kwargs)
        self.policy.reset(sequence_length, next(self.model.parameters()).device)
        self.active_rows: list[int] = []
        self.full_depth_rows: list[int] = []

    def forward_mask_logits(self, **inputs: Any):
        result = super().forward_mask_logits(**inputs)
        self.policy.observe(result.mask_logits)
        result.candidates = self.policy.protected[self.policy.mask_positions]
        decision = result.decision
        self.active_rows.append(int(decision.active.sum().item()))
        # Rows through every layer; the others (depth_22's decoded rows) start at layer refresh_depth.
        self.full_depth_rows.append(int((decision.active & decision.refresh_depth.le(1)).sum().item()))
        return result


def task_record_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    from .stale_kv_correction_pilot import task_metrics

    return {**task_metrics(rows), "position_changes": sum(counts_as_position_change(row) for row in rows)}


def run(config_path: Path, part: int, base_name: str) -> None:
    from ecad.language_models.code_tasks_and_response_processing import load_code_task_dataset
    from ecad.language_models.diffucoder_loader import load_diffucoder_model_and_tokenizer, prepare_diffucoder_prompt_batch

    from .code_correctness_and_latency_evaluator import gpu_folder_name, seed_everything
    from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
    from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
    from .forced_trajectory_replay import load_top_p_logits, replay_task
    from .kv_policy_experiment_config import load_kv_policy_experiment_config
    from .stale_kv_error_study import same_steps

    config = load_kv_policy_experiment_config(config_path)
    generation = config.generation
    if generation.algorithm != "entropy" or generation.algorithm_temperature != 0 or generation.temperature != 0:
        raise ValueError("The study assumes greedy decoding with DiffuCoder's entropy order")
    tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
    gpu = gpu_folder_name()
    output_dir = reuse_dir(config, base_name)
    replay_dir = config.run.run_dir / STUDY_GPU / "forced_trajectory_replay" / "tasks"
    (output_dir / "tasks").mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir}", flush=True)
    pending = [i for i in range(part, len(tasks), PARTS) if not (output_dir / "tasks" / f"task_{i:03d}.json").exists()]
    if not pending:
        print("All tasks of this part already finished", flush=True)
        return

    model, tokenizer = load_diffucoder_model_and_tokenizer(
        config.model.path, device=config.model.device, dtype=config.model.dtype,
        local_files_only=config.model.local_files_only,
    )
    stop_ids = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    base = config.dkv_policies[base_name]
    policies = REUSE_POLICIES[base_name]
    backends: dict[str, DiffuCoderActiveTokenKVBackend] = {base_name: DiffuCoderActiveTokenKVBackend(model, base)}
    for name, (rule, m, window) in policies.items():
        backends[name] = MaskedRowReuseBackend(model, MaskedRowReusePolicy(base, window, m, rule, stop_ids))
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)

    for count, index in enumerate(pending):
        seed_everything(generation.seed + index)
        prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [tasks[index].prompt], config.model.device)
        result = replay_task(model, sampler, backends, prompt, attention_mask, generation, top_p_logits)
        steps = result["policies"]
        saved = json.loads((replay_dir / f"task_{index:03d}.json").read_text())
        record = {
            "task_id": tasks[index].task_id,
            "gpu": gpu,
            "stop_ids": sorted(stop_ids),
            "dense_token_ids": result["dense_token_ids"],
            "dense_matches_earlier_replay": result["dense_token_ids"] == saved["dense_token_ids"],
            "base": base_name,
            "baseline_reproduces_earlier_replay": same_steps(steps[base_name], saved["policies"][base_name]),
            "check_matches_baseline": same_steps(steps[CHECK], steps[base_name]),
            "dense_flops": result["dense_flops"],
            "flops": result["flops"],
            # The check policy makes exactly the base's decisions, so its counts are the base's.
            "active_rows": {base_name: backends[CHECK].active_rows,
                            **{name: backends[name].active_rows for name in policies}},
            "full_depth_rows": {base_name: backends[CHECK].full_depth_rows,
                                **{name: backends[name].full_depth_rows for name in policies}},
            "shallow_layer_share": (base.refresh_depth - 1) / len(model.model.layers),
            "metrics": {name: task_record_metrics(rows) for name, rows in steps.items()},
            "steps": steps,
        }
        if not record["check_matches_baseline"]:
            raise RuntimeError(f"Task {index}: reuse with window 1 differs from {base_name}")
        if gpu == STUDY_GPU and not (record["dense_matches_earlier_replay"] and record["baseline_reproduces_earlier_replay"]):
            raise RuntimeError(f"Task {index}: does not reproduce the earlier {STUDY_GPU} replay")
        (output_dir / "tasks" / f"task_{index:03d}.json").write_text(json.dumps(record), encoding="utf-8")
        events = " ".join(f"{name} {m['divergence_events']:.1f}" for name, m in record["metrics"].items() if name != CHECK)
        print(f"part {part} on {gpu}: task {count + 1}/{len(pending)} {record['task_id']}: {events}", flush=True)


# -- report ------------------------------------------------------------------


def answer_length(token_ids: list[int], stop_ids: set[int]) -> int:
    return next((index for index, token in enumerate(token_ids) if token in stop_ids), len(token_ids))


def live_coverage(records: list[dict[str, Any]], name: str) -> dict[str, float]:
    """Fraction of answer and padding commits that were inside the candidates."""

    outside, total = np.zeros(2), np.zeros(2)
    for record in records:
        end = answer_length(record["dense_token_ids"], set(record["stop_ids"]))
        total += (end, len(record["dense_token_ids"]) - end)
        for row in record["steps"][name]:
            for position in row["outside_positions"]:
                outside[int(position >= end)] += 1
    return {"answer": float(1 - outside[0] / total[0]), "padding": float(1 - outside[1] / total[1]),
            "answer_outside_per_task": float(outside[0] / len(records)),
            "padding_outside_per_task": float(outside[1] / len(records))}


def projected_speedup(records: list[dict[str, Any]], name: str, timing: dict[str, Any]) -> dict[str, float]:
    """Dense time over the time of the policy's per-step row counts, on Step 1b's curves (mean over tasks).

    Layers before the base's refresh depth see only the full-depth rows, so their
    share of the time is read at that row count (Step 2's records predate this
    and have no depth rows).
    """

    speedups = {}
    for batch, curve in timing["curves"].items():
        rows, seconds = zip(*sorted((int(n), s) for n, s in curve.items()))
        dense = curve[str(timing["sequence_length"])]
        ratios = []
        for record in records:
            share = record.get("shallow_layer_share", 0.0)
            all_rows = record["active_rows"][name]
            full_rows = record.get("full_depth_rows", record["active_rows"])[name]
            time = share * np.interp(full_rows, rows, seconds) + (1 - share) * np.interp(all_rows, rows, seconds)
            ratios.append(len(all_rows) * dense / time.sum())
        speedups[batch] = float(np.mean(ratios))
    return speedups


def report(output_dir: Path, task_count: int, timing_path: Path | None, base: str = "refresh_256") -> dict[str, Any]:
    from .stale_kv_correction_pilot import paired_interval

    records = [json.loads(path.read_text()) for path in sorted((output_dir / "tasks").glob("task_*.json"))]
    if len(records) != task_count:
        raise SystemExit(f"{len(records)} replayed tasks, expected {task_count}. Finish run first.")
    timing = json.loads(timing_path.read_text()) if timing_path else None
    rng = np.random.default_rng(0)
    dense_flops = sum(record["dense_flops"] for record in records)
    policies = REUSE_POLICIES[base]
    metrics = tuple(records[0]["metrics"][base])
    summary: dict[str, Any] = {
        "tasks": len(records),
        "tasks_per_gpu": {gpu: sum(r["gpu"] == gpu for r in records) for gpu in sorted({r["gpu"] for r in records})},
        "base": base,
        "pipeline_passed": all(r["check_matches_baseline"] for r in records) and all(
            r["dense_matches_earlier_replay"] and r["baseline_reproduces_earlier_replay"]
            for r in records if r["gpu"] == STUDY_GPU),
        "policies": {},
    }
    for name in (base, *policies):
        if name == CHECK:
            continue
        entry: dict[str, Any] = {
            "compute_fraction": sum(record["flops"][name] for record in records) / dense_flops,
            "mean_rows_per_step": float(np.mean([np.mean(record["active_rows"][name]) for record in records])),
        }
        if timing:
            entry["projected_speedup"] = projected_speedup(records, name, timing)
        for metric in metrics:
            values = np.array([record["metrics"][name][metric] for record in records], dtype=float)
            entry[metric] = {"mean": float(values.mean())}
            if name != base:
                change = values - np.array([record["metrics"][base][metric] for record in records], dtype=float)
                low, high = paired_interval(change, rng, INTERVAL)
                entry[metric].update(change_mean=float(change.mean()), change_low=low, change_high=high)
        if name != base:
            entry["share_of_base_compute"] = entry["compute_fraction"] / summary["policies"][base]["compute_fraction"]
            entry["live_coverage"] = live_coverage(records, name)
            entry["criteria"] = {
                "compute": entry["compute_fraction"] <= MAX_COMPUTE_FRACTION,
                "divergence_no_worse": entry["divergence_events"]["change_high"] <= 0,
                "answer_coverage": entry["live_coverage"]["answer"] >= COVERAGE_TARGET,
            }
            entry["passes"] = all(entry["criteria"].values())
            entry["passes_amendment_2"] = (entry["criteria"]["compute"] and entry["criteria"]["answer_coverage"]
                                           and entry["divergence_events"]["change_high"] < NONINFERIORITY_MARGIN)
            # Amendment 3's replay-side conditions; the end-to-end pass/fail check is in generation_report.
            entry["amendment_3_replay"] = {
                "compute_at_most_share_of_base": entry["share_of_base_compute"] <= MAX_SHARE_OF_BASE,
                "answer_coverage": entry["criteria"]["answer_coverage"],
                "divergence_upper_below_margin": entry["divergence_events"]["change_high"] < NONINFERIORITY_MARGIN,
            }
            if timing:
                entry["speedup_plausible_at_batch_8_16"] = all(
                    entry["projected_speedup"][b] >= SPEEDUP_INDICATOR for b in ("8", "16"))
        summary["policies"][name] = entry

    # Stop if position changes rise above the baseline's and grow with the window, for every rule.
    stops = {}
    for rule in sorted({rule for rule, _, _ in policies.values()}):
        names = [name for name, (r, _, window) in policies.items() if r == rule and window > 1]
        changes = [summary["policies"][name]["position_changes"] for name in names]
        stops[rule] = all(c["change_low"] > 0 for c in changes) and all(
            a["change_mean"] < b["change_mean"] for a, b in zip(changes, changes[1:]))
    summary["position_changes_grow_with_window"] = stops
    def verdict(passing: list[str]) -> str:
        return ("INVALID: pipeline check failed" if not summary["pipeline_passed"] else
                "STOP: position changes grow with r" if all(stops.values()) else
                "PURSUE" if passing else "STOP: no policy meets criteria 1-3")

    passing = [name for name, entry in summary["policies"].items() if entry.get("passes")]
    passing_2 = [name for name, entry in summary["policies"].items() if entry.get("passes_amendment_2")]
    summary["decision"] = {"passing_policies": passing, "verdict": verdict(passing)}
    summary["decision_amendment_2"] = {"margin": NONINFERIORITY_MARGIN, "passing_policies": passing_2,
                                       "verdict": verdict(passing_2)}
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{len(records)} tasks {summary['tasks_per_gpu']}; pipeline passed: {summary['pipeline_passed']}")
    print(f"{'policy':<24}{'compute':>8}{'rows':>7}{'diverge/task':>14}{'change [95%]':>24}{'pos chg':>9}"
          f"{'ans cov':>9}{'pad cov':>9}  speedup b1/b8/b16")
    for name, entry in summary["policies"].items():
        events = entry["divergence_events"]
        interval = (f"{events['change_mean']:+.2f} [{events['change_low']:+.2f}, {events['change_high']:+.2f}]"
                    if "change_mean" in events else "")
        coverage = entry.get("live_coverage", {"answer": 1.0, "padding": 1.0})
        speed = "/".join(f"{entry['projected_speedup'][b]:.2f}" for b in ("1", "8", "16")) if timing else ""
        print(f"{name:<24}{entry['compute_fraction']:>8.3f}{entry['mean_rows_per_step']:>7.1f}{events['mean']:>14.2f}"
              f"{interval:>24}{entry['position_changes']['mean']:>9.2f}{coverage['answer']:>9.4f}{coverage['padding']:>9.4f}"
              f"  {speed}{'  PASSES' if entry.get('passes') else ''}"
              f"{'  passes (amendment 2)' if entry.get('passes_amendment_2') else ''}")
    if base == "refresh_256":
        print(f"Decision as registered: {summary['decision']['verdict']}")
        print(f"Decision under Amendment 2 (margin {NONINFERIORITY_MARGIN:.2f} events/task): "
              f"{summary['decision_amendment_2']['verdict']} {passing_2}")
    else:
        print(f"Amendment 3, replay side (compute <= {MAX_SHARE_OF_BASE} x {base}, answer coverage, "
              f"divergence upper < {NONINFERIORITY_MARGIN:.2f}; only the first two gate):")
        for name in policies:
            if name != CHECK:
                entry = summary["policies"][name]
                print(f"  {name}: {entry['share_of_base_compute']:.3f} x {base}; {entry['amendment_3_replay']}")
    return summary


# -- end-to-end generation (Amendment 2) ---------------------------------------


def generation_dir(config: Any, gpu: str, base: str) -> Path:
    return config.run.run_dir / STUDY_GPU / f"masked_row_reuse_generation{suffix(base)}" / gpu


def generate(config_path: Path, base_name: str) -> None:
    from .code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, gpu_folder_name
    from .kv_policy_experiment_config import load_kv_policy_experiment_config

    config = load_kv_policy_experiment_config(config_path)
    gpu = gpu_folder_name()
    output_dir = generation_dir(config, gpu, base_name)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    tokenizer = evaluator.tokenizer
    stop_ids = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    base = config.dkv_policies[base_name]
    variants = {"dense": config.policies["dense"], base_name: base}
    for name in END_TO_END:
        rule, m, window = REUSE_POLICIES[base_name][name]
        variants[name] = MaskedRowReusePolicy(base, window, m, rule, stop_ids)
    (output_dir / "variants").mkdir(parents=True, exist_ok=True)
    for name, policy in variants.items():
        path = output_dir / "variants" / f"{name}.json"
        if path.is_file():
            print(f"{name}: already done", flush=True)
            continue
        print(f"{name}: generating on {gpu}", flush=True)
        record = evaluator._run_once(policy, collect_history=False, collect_diagnostics=False).record
        record["code_execution"] = evaluator._evalplus(record["task_outputs"], output_dir / "code_execution" / name)
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    generation_report(config, gpu, base_name)


def generation_report(config: Any, gpu: str, base: str) -> dict[str, Any]:
    from .output_variance_study import compare_to_reference

    output_dir = generation_dir(config, gpu, base)
    records = {path.stem: json.loads(path.read_text()) for path in sorted((output_dir / "variants").glob("*.json"))}
    summary: dict[str, Any] = {"gpu": gpu, "base": base, "variants": {}}
    if gpu == STUDY_GPU:
        # The output variance study generated dense and the base on this GPU type before.
        earlier = config.run.run_dir / STUDY_GPU / "variants"
        summary["reproduces_earlier_variants"] = {
            name: records[name]["output_hash"] == json.loads((earlier / f"{name}.json").read_text())["output_hash"]
            for name in ("dense", base) if name in records
        }
    for name, record in records.items():
        summary["variants"][name] = {
            "pass_at_1": record["code_execution"]["aggregate"]["base_pass_at_1"],
            "compute_fraction": record["compute_fraction"],
            "versus_dense": compare_to_reference(records["dense"], record),
            "versus_base": compare_to_reference(records[base], record),
        }
        versus = summary["variants"][name]["versus_base"]
        # Net change in passing tasks; within -2 is consistent with the base (the E3 scale).
        summary["variants"][name]["net_passing_change_versus_base"] = versus["fail_to_pass"] - versus["pass_to_fail"]
    (output_dir / "generation_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"{gpu}; reproduces earlier variants: {summary.get('reproduces_earlier_variants', 'n/a')}")
    print(f"{'variant':<24}{'pass@1':>8}{'compute':>9}   vs dense: changed P->F F->P   vs {base}: changed P->F F->P net")
    for name, entry in summary["variants"].items():
        d, r = entry["versus_dense"], entry["versus_base"]
        print(f"{name:<24}{entry['pass_at_1']:>8.3f}{entry['compute_fraction']:>9.3f}"
              f"   {d['tasks_with_changed_code']:>16} {d['pass_to_fail']:>4} {d['fail_to_pass']:>4}"
              f"   {r['tasks_with_changed_code']:>{14 + len(base)}} {r['pass_to_fail']:>4} {r['fail_to_pass']:>4}"
              f" {entry['net_passing_change_versus_base']:>+3}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--part", type=int, required=True, choices=range(PARTS))
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--config", type=Path, required=True)
    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--config", type=Path, required=True)
    for command in (run_parser, report_parser, generate_parser):
        command.add_argument("--base", choices=sorted(REUSE_POLICIES), default="refresh_256")
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.part, args.base)
        return
    if args.command == "generate":
        generate(args.config, args.base)
        return
    from .kv_policy_experiment_config import load_kv_policy_experiment_config

    config = load_kv_policy_experiment_config(args.config)
    timing = config.run.run_dir / STUDY_GPU / "batched_row_timing" / "timing.json"
    report(reuse_dir(config, args.base), config.evaluation.max_prompts, timing if timing.is_file() else None, args.base)


if __name__ == "__main__":
    main()
