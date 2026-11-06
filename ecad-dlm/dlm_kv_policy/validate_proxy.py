"""Apply the pre-registered criteria in PROXY_VALIDATION.md.

Reads the greedy variance study (end-to-end pass/fail) and the forced-trajectory
replay from one GPU-type folder, then prints and saves the verdict. It needs no
GPU and loads no model.

By default a position flip counts only when dense was not tied between
positions (POSITION_TIE_MARGIN). --count-tied-position-flips counts every
position flip, which is the proxy as first pre-registered.

Usage:

    python -m dlm_kv_policy.validate_proxy \\
        --config ecad-dlm/experiments/configs/diffucoder7b_output_variance_greedy_64tasks.yaml \\
        --gpu-folder NVIDIA_A100-SXM4-80GB [--count-tied-position-flips]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .forced_trajectory_replay import (
    NEAR_TIE_MARGIN,
    POSITION_TIE_MARGIN,
    large_margin_decision_changes,
    step_divergence,
    summarize,
)
from .kv_policy_experiment_config import load_kv_policy_experiment_config

# Thresholds from PROXY_VALIDATION.md. Do not change them after results exist.
MIN_SPEARMAN = 0.80
MIN_AUROC = 0.75
MIN_FAILURES_FOR_AUROC = 10


def average_ranks(values: list[float]) -> list[float]:
    """Ranks starting at 1; tied values share the average of their ranks."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            ranks[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return ranks


def pearson(left: list[float], right: list[float]) -> float:
    mean_left, mean_right = sum(left) / len(left), sum(right) / len(right)
    covariance = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    spread = (sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)) ** 0.5
    return covariance / spread if spread else float("nan")


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def auroc(scores: list[float], positives: list[bool]) -> float:
    """Chance a random positive scores above a random negative; ties count half."""

    positive_scores = [score for score, positive in zip(scores, positives) if positive]
    negative_scores = [score for score, positive in zip(scores, positives) if not positive]
    if not positive_scores or not negative_scores:
        return float("nan")
    wins = sum(
        1.0 if positive > negative else 0.5 if positive == negative else 0.0
        for positive in positive_scores
        for negative in negative_scores
    )
    return wins / (len(positive_scores) * len(negative_scores))


def base_statuses(variant_record: dict[str, Any]) -> dict[str, str]:
    return {
        task["task_id"]: task["base_status"]
        for dataset in variant_record["code_execution"]["datasets"].values()
        for task in dataset["tasks"]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--gpu-folder", required=True, help="e.g. NVIDIA_A100-SXM4-80GB")
    parser.add_argument(
        "--count-tied-position-flips",
        action="store_true",
        help="Count every position flip, including those where dense was tied",
    )
    args = parser.parse_args()
    tie_margin = None if args.count_tied_position_flips else POSITION_TIE_MARGIN

    config = load_kv_policy_experiment_config(args.config)
    folder = config.run.run_dir / args.gpu_folder
    policies = list(config.dkv_policies)

    def variant(name: str) -> dict[str, Any]:
        return json.loads((folder / "variants" / f"{name}.json").read_text())

    dense = variant("dense")
    dense_status = base_statuses(dense)
    task_files = sorted((folder / "forced_trajectory_replay" / "tasks").glob("task_*.json"))
    replay = [json.loads(path.read_text()) for path in task_files]
    if len(replay) != len(dense_status):
        raise SystemExit(f"Replay has {len(replay)} tasks; the study has {len(dense_status)}. Finish the replay first.")

    preconditions = {
        "greedy_generation_is_deterministic": variant("dense_repeat")["output_hash"] == dense["output_hash"],
        "replay_follows_the_dense_run": [task["dense_token_ids"] for task in replay] == dense["generated_token_ids"],
    }

    # Criterion A: rank policies by proxy and by pass->fail count.
    replay_summary = summarize(replay, policies, tie_margin)
    rows = []
    for name in policies:
        status = base_statuses(variant(name))
        rows.append(
            {
                "policy": name,
                "proxy": replay_summary[name]["divergence_events_per_task"],
                "pass_to_fail": sum(dense_status[t] == "pass" and status[t] != "pass" for t in dense_status),
                "fail_to_pass": sum(dense_status[t] != "pass" and status[t] == "pass" for t in dense_status),
                "compute_fraction": replay_summary[name]["compute_fraction"],
                "kl_p99": replay_summary[name]["kl_p99"],
                "top1_flips": replay_summary[name]["top1_flips"],
                "large_margin_changes_per_task": replay_summary[name]["large_margin_decision_changes_per_task"],
                "statuses": status,
            }
        )
    pass_to_fail = [row["pass_to_fail"] for row in rows]
    spearman_proxy = spearman([row["proxy"] for row in rows], pass_to_fail)

    # Criterion B: per (policy, task) pairs where dense passes.
    scores, failures, divergence_sums, near_tie_only = [], [], [], []
    for row in rows:
        for task in replay:
            if dense_status[task["task_id"]] != "pass":
                continue
            steps = task["policies"][row["policy"]]
            changes = large_margin_decision_changes(steps, tie_margin)
            scores.append(changes)
            failures.append(row["statuses"][task["task_id"]] != "pass")
            divergence_sums.append(sum(step_divergence(step, tie_margin) for step in steps))
            near_tie_only.append(changes == 0 and any(step["top1_flip"] for step in steps))
    task_auroc = auroc(scores, failures)
    # How much criterion B depends on the exact tie margin.
    auroc_by_tie_margin = {}
    for margin in (None, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
        margin_scores = [
            large_margin_decision_changes(task["policies"][row["policy"]], margin)
            for row in rows
            for task in replay
            if dense_status[task["task_id"]] == "pass"
        ]
        auroc_by_tie_margin[str(margin)] = auroc(margin_scores, failures)
    kept = [index for index, skip in enumerate(near_tie_only) if not skip]

    criterion_a = "pass" if spearman_proxy >= MIN_SPEARMAN else "fail"
    if sum(failures) < MIN_FAILURES_FOR_AUROC:
        criterion_b = "inconclusive"
    else:
        criterion_b = "pass" if task_auroc >= MIN_AUROC else "fail"
    if not all(preconditions.values()):
        verdict = "INCONCLUSIVE"
    elif "fail" in (criterion_a, criterion_b):
        verdict = "FAIL"
    elif criterion_a == criterion_b == "pass":
        verdict = "PASS"
    else:
        verdict = "INCONCLUSIVE"

    report = {
        "position_tie_margin": tie_margin,
        "verdict": verdict,
        "preconditions": preconditions,
        "criterion_a": {"spearman": spearman_proxy, "threshold": MIN_SPEARMAN, "result": criterion_a},
        "criterion_b": {
            "auroc": task_auroc,
            "threshold": MIN_AUROC,
            "failures": sum(failures),
            "pairs": len(failures),
            "result": criterion_b,
        },
        "not_part_of_verdict": {
            "spearman_kl_p99": spearman([row["kl_p99"] for row in rows], pass_to_fail),
            "spearman_top1_flips": spearman([row["top1_flips"] for row in rows], pass_to_fail),
            "spearman_large_margin_changes": spearman(
                [row["large_margin_changes_per_task"] for row in rows], pass_to_fail
            ),
            "auroc_summed_divergence": auroc(divergence_sums, failures),
            "auroc_excluding_near_tie_only_pairs": auroc(
                [scores[i] for i in kept], [failures[i] for i in kept]
            ),
            "total_pass_to_fail": sum(pass_to_fail),
            "total_fail_to_pass": sum(row["fail_to_pass"] for row in rows),
            "near_tie_margin": NEAR_TIE_MARGIN,
            "auroc_by_position_tie_margin": auroc_by_tie_margin,
        },
        "policies": [{key: value for key, value in row.items() if key != "statuses"} for row in rows],
    }
    report_name = "proxy_validation_report_counting_ties.json" if tie_margin is None else "proxy_validation_report.json"
    (folder / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"{'policy':<26} {'proxy':>8} {'P->F':>5} {'F->P':>5} {'compute':>8}")
    for row in sorted(rows, key=lambda row: row["proxy"]):
        print(
            f"{row['policy']:<26} {row['proxy']:>8.2f} {row['pass_to_fail']:>5} "
            f"{row['fail_to_pass']:>5} {row['compute_fraction']:>8.3f}"
        )
    print(f"\nPosition flips counted at dense margin >= {tie_margin} (None: all)")
    print(f"Preconditions: {preconditions}")
    print(f"A. Spearman(proxy, pass->fail) = {spearman_proxy:.3f} (need {MIN_SPEARMAN}): {criterion_a}")
    print(f"B. Per-task AUROC = {task_auroc:.3f} (need {MIN_AUROC}, {sum(failures)} failures): {criterion_b}")
    print(f"Verdict: {verdict}")


if __name__ == "__main__":
    main()
