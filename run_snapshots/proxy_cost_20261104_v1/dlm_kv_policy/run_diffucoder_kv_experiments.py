"""CLI for fixed-policy evaluation, KV evolution, and held-out scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .kv_policy_experiment_config import load_kv_policy_experiment_config
from .code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Override run.output_dir; created automatically")
    parser.add_argument("--run-name", help="Override run.name to isolate a hardware-specific study")
    parser.add_argument(
        "--fixed-policy-names", help="Comma-separated subset of fixed policies; default is fixed_policy_evaluation.policies"
    )
    dkv_mode = parser.add_mutually_exclusive_group()
    dkv_mode.add_argument("--dkv-decode-reference-evaluation", action="store_true")
    dkv_mode.add_argument("--dkv-decode-ablation-evaluation", action="store_true")
    dkv_mode.add_argument("--kv-policy-nsga2-search", action="store_true")
    parser.add_argument("--ablation-policy-names", help="Comma-separated dKV ablation policy names")
    parser.add_argument(
        "--populations-per-job",
        type=int,
        default=1,
        help="Evaluated populations to run in this resumable Slurm job",
    )
    parser.add_argument(
        "--reuse-dense-baseline-dir",
        type=Path,
        help="Directory containing a signature-matched baseline.json and baseline.base_eval.json",
    )
    parser.add_argument("--skip-code-execution", action="store_true")
    parser.add_argument(
        "--heldout-evalplus-policy",
        metavar="POLICY",
        help="Run EvalPlus Plus for one evaluated policy or dense; never used in selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_kv_policy_experiment_config(args.config)
    if args.output_dir is not None:
        config.run.output_dir = str(args.output_dir.expanduser())
    if args.run_name:
        config.run.name = args.run_name
    if args.skip_code_execution:
        config.fixed_policy_evaluation.run_evalplus = False
    if args.reuse_dense_baseline_dir and not (
        args.dkv_decode_reference_evaluation or args.dkv_decode_ablation_evaluation or args.kv_policy_nsga2_search or args.heldout_evalplus_policy
    ):
        raise ValueError("--reuse-dense-baseline-dir is only valid with a full dKV evaluation or evolution")
    if args.ablation_policy_names and not args.dkv_decode_ablation_evaluation:
        raise ValueError("--ablation-policy-names is only valid with --dkv-decode-ablation-evaluation")
    if (args.dkv_decode_reference_evaluation or args.dkv_decode_ablation_evaluation or args.kv_policy_nsga2_search) and args.skip_code_execution:
        raise ValueError("Full dKV evaluation and evolution always run paired EvalPlus base tests")
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    config.run.run_dir.mkdir(parents=True, exist_ok=True)
    (config.run.run_dir / "resolved_config.json").write_text(
        json.dumps(config.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    if args.heldout_evalplus_policy:
        print(
            json.dumps(
                evaluator.evaluate_heldout_plus(
                    args.heldout_evalplus_policy, reuse_baseline=args.reuse_dense_baseline_dir
                ),
                indent=2,
                sort_keys=True,
            ),
            flush=True,
        )
        return
    if args.dkv_decode_reference_evaluation:
        result = evaluator.evaluate_dkv_decode_reference(reuse_baseline=args.reuse_dense_baseline_dir)
        path = config.run.run_dir / "dkv_decode_reference_evaluation_summary.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    if args.dkv_decode_ablation_evaluation:
        selected = args.ablation_policy_names.split(",") if args.ablation_policy_names else None
        result = evaluator.evaluate_dkv_decode_ablations(selected, reuse_baseline=args.reuse_dense_baseline_dir)
        path = config.run.run_dir / "dkv_decode_ablation_evaluation_summary.json"
        path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    if args.kv_policy_nsga2_search:
        from .nsga2_kv_policy_search import run_kv_policy_nsga2_search

        result = run_kv_policy_nsga2_search(
            config,
            evaluator,
            reuse_baseline=args.reuse_dense_baseline_dir,
            generations_this_job=args.populations_per_job,
        )
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
        return
    selected = args.fixed_policy_names.split(",") if args.fixed_policy_names else None
    print(json.dumps(evaluator.run(selected), indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
