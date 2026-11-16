"""Is the Expected Acceptance Rate a fair fidelity constraint for DiffuCoder caches?

Replays the 96 tasks of proxy_recheck.py with EAR logged, adds free-running
programs for the policies that run did not cover, and tests whether EAR separates
policies inside and outside the outcome margin. The design, the margin and the
validity rule are in EAR_VALIDITY_STUDY.md.

  run      GPU, A100 80GB. One shard of forced-trajectory replay and generation.
           The dense output and every policy already in the proxy recheck must
           reproduce that run exactly.
  report   No model. Scores the new programs, then writes the policy table,
           scatter data, leave-one-policy-out classification and thresholds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dlm_kv_policy.dense_diffucoder_baseline import DenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.forced_trajectory_replay import counts_as_position_change, load_top_p_logits, replay_task

from .batched_decoding_study import source_provenance
from .proxy_recheck import PANEL as RECHECK_PANEL
from .proxy_recheck import auc_score, backend, generation_settings, policies, rank_score, save

SEED = 20261106
GPU = "NVIDIA A100-SXM4-80GB"
NOISE_FLOOR = "dense_math"
# name, full refresh interval, depth, masked window, protected size, rule
NEW_PANEL = [
    ("dkv_reference", 8, 1, None, 32, "split+recent"), ("refresh_16", 16, 1, None, 32, "split+recent"),
    ("refresh_32", 32, 1, None, 32, "split+recent"), ("refresh_32_depth_22", 32, 22, None, 32, "split+recent"),
]
PANEL = [*RECHECK_PANEL, *NEW_PANEL]
ALL_POLICIES = (NOISE_FLOOR, *(row[0] for row in PANEL))
# The panel the verdict is registered on; ALL_POLICIES is the sensitivity check.
REGISTERED = (NOISE_FLOOR, "depth_22", "depth_22_r2", "depth_22_r4", "depth_22_r8", "depth_24_r4", "refresh_256",
              "refresh_256_r2", "refresh_256_r4", "refresh_32", "refresh_32_depth_22")
# Near-lossless: a net loss of at most this many of the 96 tasks against dense.
NET_LOSS_MARGIN = 2
TAIL_QUANTILE = 0.10
BOOTSTRAPS = 2000


class MathKernelDense(DenseDiffuCoderBaseline):
    """Dense with PyTorch's math attention kernel: the E3 noise floor, as a replay policy."""

    def forward_mask_logits(self, **inputs: Any):
        with sdpa_kernel(SDPBackend.MATH):
            result = super().forward_mask_logits(**inputs)
        result.candidates = None
        return result


def answer_length(token_ids: list[int], stop_ids: set[int]) -> int:
    return next((index for index, token in enumerate(token_ids) if token in stop_ids), len(token_ids))


def run(config_path: Path, proxy_output: Path, output: Path, part: int, parts: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, seed_everything
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
    from language_models.code_tasks_and_response_processing import make_code_task_output
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    from .stale_kv_error_study import same_steps

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Study requires {GPU}: it must reproduce the proxy recheck exactly")
    stop_ids = {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}
    provenance = {**source_provenance(), "torch": torch.__version__, "gpu": GPU, "config": asdict(config),
                  "panel": PANEL, "part": part, "parts": parts}
    signature = hashlib.sha256(json.dumps({key: provenance[key] for key in ("source_sha256", "config", "panel")},
                                          sort_keys=True, default=str).encode()).hexdigest()
    provenance["signature"] = signature
    meta = output / f"part_{part}_provenance.json"
    if meta.exists() and json.loads(meta.read_text())["signature"] != signature:
        raise RuntimeError("Refusing mixed-source resume")
    save(meta, provenance)
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)
    settings = generation_settings(config)

    for index in range(part, len(evaluator.tasks), parts):
        path = output / "tasks" / f"task_{index:03d}.json"
        if path.exists():
            if json.loads(path.read_text())["signature"] != signature:
                raise RuntimeError("Task signature mismatch")
            continue
        task = evaluator.tasks[index]
        saved = json.loads((proxy_output / "tasks" / f"task_{index:03d}.json").read_text())
        if saved["task_id"] != task.task_id:
            raise RuntimeError(f"Task {index} is not the proxy recheck's task")
        answer_end = answer_length(saved["replay"]["dense_token_ids"], stop_ids)
        seed_everything(config.generation.seed + index)
        prompt, mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], config.model.device)
        backends = {name: backend(model, policy) for name, policy in policies(stop_ids, PANEL).items()}
        backends[NOISE_FLOOR] = MathKernelDense(model)
        print(f"part {part}: task {index} replay", flush=True)
        replay = replay_task(model, sampler, backends, prompt, mask, config.generation, top_p_logits,
                             answer_end=answer_end)
        if replay["dense_token_ids"] != saved["replay"]["dense_token_ids"]:
            raise RuntimeError(f"Task {index}: dense differs from the proxy recheck")
        for name, *_ in RECHECK_PANEL:
            if not same_steps(replay["policies"][name], saved["replay"]["policies"][name]):
                raise RuntimeError(f"Task {index}: {name} differs from the proxy recheck")
        del backends
        generations = {}
        for name, policy in policies(stop_ids, NEW_PANEL).items():
            print(f"part {part}: task {index} generation {name}", flush=True)
            result = DiffuCoderDenoisingSampler(backend(model, policy)).generate(prompt, attention_mask=mask, **settings)
            ids = result.sequences[0, prompt.shape[1]:].cpu().tolist()
            generations[name] = {"token_ids": ids, "output": make_code_task_output(task, tokenizer.decode(ids, skip_special_tokens=True))}
        save(path, {"signature": signature, "task_id": task.task_id, "index": index, "gpu": GPU,
                    "stop_ids": sorted(stop_ids), "answer_end": answer_end, "replay": replay, "generations": generations})
        print(f"part {part}: {index} {task.task_id} complete", flush=True)


# -- report ------------------------------------------------------------------


def task_metrics(rows: list[dict[str, Any]], answer_end: int) -> dict[str, float]:
    """Means over answer commits: positions before dense's first stop token."""

    ear, kl, flips, changes, masked = [], [], [], [], []
    for row in rows:
        inside = [value for value, position in zip(row["ear"], row["committed_positions"]) if position < answer_end]
        if row["ear_masked_answer"] is not None:
            masked.append(row["ear_masked_answer"])
        if not inside:
            continue
        ear.extend(inside)
        kl.append(max(row["kl"], 0.0))
        flips.append(row["position_flip"])
        changes.append(counts_as_position_change(row))

    def mean(values: list[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    return {"ear": mean(ear), "kl": mean(kl), "position_flip_rate": mean(flips),
            "position_change_rate": mean(changes), "masked_ear": mean(masked)}


def leave_one_out(names: list[str], ear: dict[str, float], within: dict[str, bool],
                  flips: dict[str, float] | None = None) -> dict[str, Any]:
    """Classify each policy by a threshold fitted on the others.

    The threshold is the largest EAR that keeps every within-margin policy safe:
    their lowest EAR. With ``flips``, a policy must also have no more position
    flips than the within-margin policy with the most.
    """

    calls = {}
    for held in names:
        inside = [name for name in names if name != held and within[name]]
        threshold = min(ear[name] for name in inside)
        safe = ear[held] >= threshold
        if flips is not None:
            safe = safe and flips[held] <= max(flips[name] for name in inside)
        calls[held] = {"threshold": threshold, "safe": bool(safe), "within_margin": within[held]}
    false_safe = [name for name, call in calls.items() if call["safe"] and not call["within_margin"]]
    false_reject = [name for name, call in calls.items() if not call["safe"] and call["within_margin"]]
    return {"calls": calls, "false_safe": false_safe, "false_reject": false_reject,
            "passed": not false_safe and len(false_reject) <= 1}


def verdict(names: list[str], ear: dict[str, float], within: dict[str, bool], flips: dict[str, float]) -> dict[str, Any]:
    plain = leave_one_out(names, ear, within)
    with_flips = leave_one_out(names, ear, within, flips)
    result = "valid" if plain["passed"] else "valid_with_position_flips" if with_flips["passed"] else "not_valid"
    return {"verdict": result, "ear_only": plain, "ear_and_position_flips": with_flips}


def outcomes(name: str, directories: list[Path]) -> dict[str, dict[str, Any]]:
    """Per-task base status and test counts, from whichever run scored this policy."""

    path = next(directory / "scores" / f"{name}_full.json" for directory in directories
                if (directory / "scores" / f"{name}_full.json").is_file())
    result = json.loads(path.read_text())
    return {task["task_id"]: task for dataset in result["datasets"].values() for task in dataset["tasks"]}


def score_new_policies(config_path: Path, records: list[dict[str, Any]], output: Path) -> None:
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
    from language_models.code_execution import run_evalplus_apptainer

    settings = load_kv_policy_experiment_config(config_path).evaluation
    for name, *_ in NEW_PANEL:
        path = output / "scores" / f"{name}_full.json"
        if path.exists():
            continue
        save(path, run_evalplus_apptainer(
            [record["generations"][name]["output"] for record in records], image=settings.evalplus_image,
            work_dir=output / "execution" / name, parallel=settings.evalplus_parallel,
            apptainer_bin=settings.evalplus_apptainer_bin, scope="full", timeout_seconds=settings.evalplus_timeout_seconds,
            dataset_paths={"humaneval": settings.evalplus_humaneval_path, "mbpp": settings.evalplus_mbpp_path}))


def pass_fraction(task: dict[str, Any]) -> float:
    return (task["base_passed"] + task.get("plus_passed", 0)) / (task["base_total"] + task.get("plus_total", 0))


def correlations(names: list[str], table: dict[str, dict[str, Any]], metric: str) -> dict[str, float | None]:
    distance = [1 - table[name][metric] for name in names]
    return {"net_loss": rank_score(distance, [table[name]["net_loss"] for name in names]),
            "pass_fraction_drop": rank_score(distance, [table[name]["pass_fraction_drop"] for name in names])}


def report(config_path: Path, proxy_output: Path, output: Path) -> dict[str, Any]:
    records = [json.loads(path.read_text()) for path in sorted((output / "tasks").glob("task_*.json"))]
    if len(records) != 96 or len({record["signature"] for record in records}) != 1:
        raise SystemExit(f"{len(records)} tasks from one source needed, expected 96")
    score_new_policies(config_path, records, output)
    directories = [proxy_output, output]
    dense = outcomes("dense", directories)
    table: dict[str, dict[str, Any]] = {}
    per_task: dict[str, dict[str, np.ndarray]] = {}
    losses: dict[str, np.ndarray] = {}
    dense_flops = sum(record["replay"]["dense_flops"] for record in records)
    for name in ALL_POLICIES:
        rows = [task_metrics(record["replay"]["policies"][name], record["answer_end"]) for record in records]
        values = {key: np.array([row[key] for row in rows]) for key in rows[0]}
        per_task[name] = values
        scored = outcomes(name, directories)
        ids = [record["task_id"] for record in records]
        passes = np.array([scored[task]["base_status"] == "pass" for task in ids])
        reference = np.array([dense[task]["base_status"] == "pass" for task in ids])
        losses[name] = reference & ~passes
        lost, gained = int(losses[name].sum()), int((~reference & passes).sum())
        table[name] = {
            "ear": float(np.nanmean(values["ear"])),
            "tail_ear": float(np.nanquantile(values["ear"], TAIL_QUANTILE)),
            "position_flip_rate": float(np.nanmean(values["position_flip_rate"])),
            "position_change_rate": float(np.nanmean(values["position_change_rate"])),
            "masked_ear": float(np.nanmean(values["masked_ear"])),
            "kl": float(np.nanmean(values["kl"])),
            "lost": lost, "gained": gained, "net_loss": lost - gained,
            "pass_fraction_drop": float(np.mean([pass_fraction(dense[task]) - pass_fraction(scored[task]) for task in ids])),
            "compute_fraction": sum(record["replay"]["flops"][name] for record in records) / dense_flops,
        }
    within = {name: table[name]["net_loss"] <= NET_LOSS_MARGIN for name in ALL_POLICIES}
    ear = {name: table[name]["ear"] for name in ALL_POLICIES}
    flips = {name: table[name]["position_flip_rate"] for name in ALL_POLICIES}
    summary: dict[str, Any] = {
        "tasks": len(records), "margin": NET_LOSS_MARGIN, "table": table, "within_margin": within,
        "registered": verdict(list(REGISTERED), ear, within, flips),
        "all_policies": verdict(list(ALL_POLICIES), ear, within, flips),
    }

    # Thresholds, from the registered panel; a policy's bootstrap lower bound must clear them.
    inside = [name for name in REGISTERED if within[name]]
    rng = np.random.default_rng(SEED)
    samples = rng.integers(0, len(records), size=(BOOTSTRAPS, len(records)))
    lower = {name: float(np.nanquantile(np.nanmean(per_task[name]["ear"][samples], axis=1), 0.05)) for name in ALL_POLICIES}
    thresholds = {"ear": min(ear[name] for name in inside), "tail_ear": min(table[name]["tail_ear"] for name in inside),
                  "position_flip_rate": max(flips[name] for name in inside)}
    summary["thresholds"] = {"values": thresholds, "use_position_flips": summary["registered"]["verdict"] == "valid_with_position_flips",
                             "usable": summary["registered"]["verdict"] != "not_valid", "ear_lower_95": lower,
                             "clears_ear_threshold": {name: lower[name] >= thresholds["ear"] for name in ALL_POLICIES}}

    # Descriptive: the continuous question. Nothing below enters the verdict.
    depth_family = [name for name in ALL_POLICIES if "depth" in name and "refresh" not in name]
    summary["continuous"] = {
        metric: {"all_policies": correlations(list(ALL_POLICIES), table, metric),
                 "depth_family": correlations(depth_family, table, metric)}
        for metric in ("ear", "tail_ear", "masked_ear")
    }
    # Can a task's own EAR say which tasks a policy loses? Tasks dense passes, all policies pooled.
    ids = [record["task_id"] for record in records]
    reference = np.array([dense[task]["base_status"] == "pass" for task in ids])
    summary["task_auroc"] = {}
    for metric in ("ear", "masked_ear"):
        scores = np.concatenate([1 - per_task[name][metric][reference] for name in ALL_POLICIES])
        labels = np.concatenate([losses[name][reference] for name in ALL_POLICIES])
        known = ~np.isnan(scores)
        summary["task_auroc"][metric] = auc_score(scores[known], labels[known])
    summary["scatter"] = [{"policy": name, "one_minus_ear": 1 - table[name]["ear"], "net_loss": table[name]["net_loss"],
                           "pass_fraction_drop": table[name]["pass_fraction_drop"]} for name in ALL_POLICIES]
    save(output / "ear_report.json", summary)

    print(f"{'policy':<22}{'EAR':>8}{'tail':>8}{'flips':>7}{'lost':>5}{'gain':>5}{'net':>4}{'drop':>8}{'compute':>8}"
          f"{'masked':>8}  safe")
    for name, row in sorted(table.items(), key=lambda item: -item[1]["ear"]):
        print(f"{name:<22}{row['ear']:>8.4f}{row['tail_ear']:>8.4f}{row['position_flip_rate']:>7.3f}{row['lost']:>5}"
              f"{row['gained']:>5}{row['net_loss']:>4}{row['pass_fraction_drop']:>8.4f}{row['compute_fraction']:>8.3f}"
              f"{row['masked_ear']:>8.4f}  {'within' if within[name] else 'OUTSIDE'}{'' if name in REGISTERED else ' (extra)'}")
    for label in ("registered", "all_policies"):
        entry = summary[label]
        print(f"{label}: {entry['verdict']}; false safe {entry['ear_only']['false_safe']}; "
              f"false reject {entry['ear_only']['false_reject']}")
    print("continuous:", json.dumps(summary["continuous"]), "task AUROC:", summary["task_auroc"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "report"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--proxy-output", type=Path, required=True, help="proxy_recheck.py output folder")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=16)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.config, args.proxy_output, args.output, args.part, args.parts)
    else:
        report(args.config, args.proxy_output, args.output)


if __name__ == "__main__":
    main()
