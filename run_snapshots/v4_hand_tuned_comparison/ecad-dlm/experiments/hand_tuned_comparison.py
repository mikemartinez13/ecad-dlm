"""Run 4's latest population against the hand-tuned policies, on the audit set, in one job.

Measured speedups compare only within one job (CONSTRAINED_SEARCH.md, run 3), so every
policy here is generated and timed in the same job, against the same dense:

  - run 4's population after its latest finished generation, fastest predicted first;
  - run 3's best, a control that links this job to run 3's audit (4.39x there);
  - the hand-tuned policies: dKV-Cache-Decode's setting (refresh every 8 steps, delay 1),
    the fastest feasible hand-tuned policies of the tau study (depth_22_r8,
    gene_padding_16) and its fastest hand-tuned policy overall (refresh_256_r4).

Each policy is generated free-running at batch one on the 96 audit tasks and scored on the
base tests, then timed at batch 8 against dense, as ``constrained_kv_policy_search audit``
does. Dense is timed again at the end, as a check on drift. Results go to
``<output>/audit/audit.json`` in the audit's format, so ``constrained_kv_policy_search plus``
reruns the programs under EvalPlus's extra tests. Generated records are saved per policy,
so a rerun skips generation; every rerun times all policies again.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
from evolution.constrained_kv_policy_search import GPU, generation_dir, stop_ids
from evolution.nsga2_kv_policy_search import EvaluatedKVPolicy, KVRefreshPolicyGenome
from experiments.proxy_recheck import backend, generation_settings, policies
from utils.results_io import save

# name, full refresh interval, depth, masked window, protected size, rule (proxy_recheck.PANEL's format)
HAND_TUNED_PANEL = [
    ("dkv_reference", 8, 1, None, 32, "split+recent"),
    ("depth_22_r8", 8, 22, 8, 32, "split+recent"),
    ("refresh_256_r4", 256, 1, 4, 32, "split+recent"),
]
HAND_TUNED_GENOMES = {"gene_padding_16": KVRefreshPolicyGenome(1, 8, 22, 0, 4, padding_window=16)}
# Measured at batch 8 in the tau study (2026-11-07), on an older, slower backend.
TAU_STUDY_MEASURED = {"dkv_reference": 1.78, "depth_22_r8": 2.07, "refresh_256_r4": 2.64, "gene_padding_16": 2.09}
RUN3_BEST = "ga_d2_r16_l1_t0_p128_n4_w8_m16_split_pad64_h0.5full"


def latest_population(search: Path) -> tuple[int, list[EvaluatedKVPolicy]]:
    path = sorted(search.glob("generation_*/summary.json"))[-1]
    summary = json.loads(path.read_text())
    population = [EvaluatedKVPolicy.from_dict(raw) for raw in summary["population"]]
    return summary["generation"], sorted(population, key=lambda item: -item.constraints["predicted_speedup"])


def search_numbers(item: EvaluatedKVPolicy) -> dict[str, Any]:
    c = item.constraints
    return {"padded_flops_speedup": c["predicted_speedup"], "flops_speedup": c["flops_speedup"], "ear": c["ear"],
            "ear_lower_95": c["ear_lower_95"], "tail_ear": c["tail_ear"], "answer_coverage": c["answer_coverage"]}


def run(config_path: Path, search: Path, output: Path) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline
    from experiments.output_variance_study import compare_to_reference
    from experiments.tau_confirmation import measured_batched_seconds

    config = load_kv_policy_experiment_config(config_path)
    generation, population = latest_population(search)
    # Run 3's best, as run 4's generation 0 re-scored it under padded FLOPs.
    start = {item.genome.name: item for item in map(EvaluatedKVPolicy.from_dict,
             json.loads((generation_dir(search, 0) / "summary.json").read_text())["evaluated"])}
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Comparison requires {GPU}")
    stops = stop_ids(evaluator.tokenizer)
    entries: list[tuple[str, str, Any, dict[str, Any]]] = []  # name, kind, policy, extra fields
    for name, policy in policies(stops, HAND_TUNED_PANEL).items():
        entries.append((name, "hand_tuned", policy, {"tau_study_measured_speedup": TAU_STUDY_MEASURED[name]}))
    for name, genome in HAND_TUNED_GENOMES.items():
        entries.append((name, "hand_tuned", genome.policy(stops), {"genome": genome.to_dict(),
                                                                    "tau_study_measured_speedup": TAU_STUDY_MEASURED[name]}))
    entries.append((RUN3_BEST, "run3_best", start[RUN3_BEST].genome.policy(stops),
                    {"genome": start[RUN3_BEST].genome.to_dict(), **search_numbers(start[RUN3_BEST])}))
    for item in population:
        if item.genome.name != RUN3_BEST:
            entries.append((item.genome.name, f"run4_generation_{generation:03d}", item.genome.policy(stops),
                            {"genome": item.genome.to_dict(), **search_numbers(item)}))
    print(f"Run 4 generation {generation}: {len(population)} policies; {len(entries)} policies in all", flush=True)

    def generate(name: str, policy: Any) -> dict[str, Any]:
        path = output / "audit" / "records" / f"{name}.json"
        if path.exists():
            return json.loads(path.read_text())
        print(f"generating {name}", flush=True)
        record = evaluator._run_once(policy, collect_history=False, collect_diagnostics=False).record
        record["code_execution"] = evaluator._evalplus(record["task_outputs"], output / "audit" / "execution" / name)
        save(path, record)
        return record

    settings = generation_settings(config)

    def seconds(make_backend: Any) -> float:
        return measured_batched_seconds(evaluator.model, evaluator.tokenizer, evaluator.tasks, make_backend, settings, config.model.device)

    dense = generate("dense", config.policies["dense"])
    dense_seconds = seconds(lambda: BatchedDenseDiffuCoderBaseline(evaluator.model))
    print(f"dense: {dense_seconds:.1f} s at batch 8", flush=True)
    header = {"search": str(search), "run4_generation": generation, "dense_seconds": dense_seconds,
              "dense_pass_at_1": dense["code_execution"]["aggregate"]["base_pass_at_1"]}
    rows = []
    for name, kind, policy, extra in entries:
        record = generate(name, policy)
        result = compare_to_reference(dense, record)
        policy_seconds = seconds(lambda: backend(evaluator.model, policy, batched=True))
        rows.append({"policy": name, "kind": kind, "seconds": policy_seconds, "measured_speedup": dense_seconds / policy_seconds,
                     "compute_fraction": record["compute_fraction"], "net_loss": result["pass_to_fail"] - result["fail_to_pass"],
                     **{key: result[key] for key in ("pass_at_1", "pass_to_fail", "fail_to_pass", "tasks_with_changed_code")},
                     **extra})
        row = rows[-1]
        print(f"{name} ({kind}): measured {row['measured_speedup']:.2f}x, pass@1 {row['pass_at_1']:.3f}, "
              f"lost {row['pass_to_fail']}, gained {row['fail_to_pass']}", flush=True)
        save(output / "audit" / "audit.json", {**header, "policies": rows})
    header["dense_seconds_end"] = seconds(lambda: BatchedDenseDiffuCoderBaseline(evaluator.model))
    save(output / "audit" / "audit.json", {**header, "policies": rows})
    print(f"dense again at the end: {header['dense_seconds_end']:.1f} s", flush=True)
    for row in sorted(rows, key=lambda row: -row["measured_speedup"]):
        print(f"  {row['measured_speedup']:.2f}x  pass@1 {row['pass_at_1']:.3f}  {row['pass_to_fail']}/{row['fail_to_pass']}  "
              f"{row['kind']}  {row['policy']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="The audit set's config")
    parser.add_argument("--search", type=Path, required=True, help="Run 4's output folder")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.search, args.output)


if __name__ == "__main__":
    main()
