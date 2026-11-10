"""Test saved cache policies against dense: free-running pass@1 and measured batched speed.

Each policy is one JSON file holding a genome, either the bare dict from
KVRefreshPolicyGenome.to_dict() or a search summary entry with it under "genome".
The file name names the policy. The config supplies the model, tasks and EvalPlus
settings, and must define a `dense` policy; configs/diffucoder7b_audit_96tasks.yaml
is one. GPU, A100 80GB for timings comparable with the search.

  python -m experiments.evaluate_policies --config CONFIG --policies policies/ --output OUT

Writes OUT/policies.json and one EvalPlus folder per policy under OUT/execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome as Genome

from .ear_validity import pass_fraction
from .proxy_recheck import backend, generation_settings, save


def load_policies(directory: Path) -> dict[str, Genome]:
    policies = {}
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        policies[path.stem] = Genome.from_dict(raw.get("genome", raw))
    if not policies:
        raise SystemExit(f"No policy files in {directory}")
    return policies


def passes(task: dict[str, Any]) -> tuple[bool, bool]:
    """Base tests, and base plus EvalPlus's extra tests."""

    return task["base_status"] == "pass", task["base_status"] == task["plus_status"] == "pass"


def evaluate(config_path: Path, directory: Path, output: Path, tasks: int | None, batch: int, timing: bool) -> list[dict[str, Any]]:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline

    from .tau_confirmation import measured_batched_seconds, stop_ids

    policies = load_policies(directory)
    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    chosen = evaluator.tasks[:tasks] if tasks else evaluator.tasks
    stops = stop_ids(evaluator.tokenizer)
    settings = generation_settings(config)

    def run(name: str, policy: Any) -> tuple[dict[str, Any], float]:
        print(f"generating {name}", flush=True)
        record = evaluator._run_once(policy, collect_history=False, collect_diagnostics=False, tasks=chosen).record
        result = evaluator._evalplus(record["task_outputs"], output.resolve() / "execution" / name, scope="full")
        scored = {task["task_id"]: task for dataset in result["datasets"].values() for task in dataset["tasks"]}
        return scored, record.get("compute_fraction", 1.0)

    def seconds(make_backend: Any) -> float:
        return measured_batched_seconds(evaluator.model, evaluator.tokenizer, chosen, make_backend, settings,
                                        config.model.device, batch)

    dense, _ = run("dense", config.policies["dense"])
    dense_seconds = seconds(lambda: BatchedDenseDiffuCoderBaseline(evaluator.model)) if timing else None
    rows = []
    for name, genome in policies.items():
        scored, compute = run(name, genome.policy(stops))
        row: dict[str, Any] = {"policy": name, "genome": genome.to_dict(), "compute_fraction": compute,
                               "pass_fraction": float(np.mean([pass_fraction(task) for task in scored.values()]))}
        for index, kind in enumerate(("base", "plus")):
            reference = np.array([passes(dense[task])[index] for task in dense])
            policy = np.array([passes(scored[task])[index] for task in dense])
            row.update({f"{kind}_pass_at_1": float(policy.mean()), f"{kind}_lost": int((reference & ~policy).sum()),
                        f"{kind}_gained": int((~reference & policy).sum())})
        if timing:
            row["measured_speedup"] = dense_seconds / seconds(lambda: backend(evaluator.model, genome.policy(stops), batched=True))
        rows.append(row)
        save(output / "policies.json", {"tasks": len(dense), "batch": batch, "dense_base_pass_at_1":
                                        float(np.mean([passes(task)[0] for task in dense.values()])),
                                        "dense_plus_pass_at_1": float(np.mean([passes(task)[1] for task in dense.values()])),
                                        "policies": rows})
    for row in rows:
        speed = f", {row['measured_speedup']:.2f}x at batch {batch}" if timing else ""
        print(f"{row['policy']}: base pass@1 {row['base_pass_at_1']:.3f} ({row['base_lost']} lost, {row['base_gained']} gained), "
              f"plus pass@1 {row['plus_pass_at_1']:.3f} ({row['plus_lost']} lost, {row['plus_gained']} gained){speed}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--policies", type=Path, required=True, help="folder of policy JSON files")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", type=int, help="only the first N tasks, for a quick check")
    parser.add_argument("--batch", type=int, default=8, help="batch size for measured speed")
    parser.add_argument("--no-timing", action="store_true", help="skip measured batched speed")
    args = parser.parse_args()
    evaluate(args.config, args.policies, args.output, args.tasks, args.batch, not args.no_timing)


if __name__ == "__main__":
    main()
