"""Experiments E1-E3: how much does generated code change when it should not?

Each *variant* generates every task once, is scored with EvalPlus base tests,
and is compared with the study's reference variant. The report answers:

- E1 (sampling noise): dense with other seeds at temperature 0.4.
- E2 (greedy decoding): dense twice, then cached policies, all at temperature 0.
- E3 (rounding noise): dense with PyTorch's math attention kernel instead of
  the default fused kernel. The model is unchanged; only rounding differs.

A variant in the YAML ``variance_study.variants`` list looks like:

    - name: dense_seed1000
      policy: dense                 # "dense" or a name under dkv_policies
      generation: {seed: 1000}      # overrides of the generation section
      sdpa_backend: math            # optional; default keeps PyTorch's choice

Usage (inside a GPU allocation, from the repository root):

    python -m dlm_kv_policy.output_variance_study --config CONFIG.yaml
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from torch.nn.attention import SDPBackend, sdpa_kernel
import yaml

from ecad.language_models.code_execution import compare_base_correctness

from .code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, gpu_folder_name
from .kv_policy_experiment_config import load_kv_policy_experiment_config


def load_study(config_path: Path) -> tuple[str, list[dict[str, Any]]]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    study = raw["variance_study"]
    names = [variant["name"] for variant in study["variants"]]
    if len(names) != len(set(names)):
        raise ValueError("Variant names must be unique")
    if study["reference"] not in names:
        raise ValueError("variance_study.reference must name one of the variants")
    return study["reference"], study["variants"]


def run_variant(evaluator: DiffuCoderKVPolicyEvaluator, variant: dict[str, Any]) -> dict[str, Any]:
    """Generate all tasks once for one variant and score them with base tests."""

    config = evaluator.config
    policy_name = variant.get("policy", "dense")
    policy = config.policies["dense"] if policy_name == "dense" else config.dkv_policies[policy_name]

    # The evaluator reads its generation settings on every run, so the
    # overrides apply to this variant only.
    base_generation = config.generation
    config.generation = replace(base_generation, **variant.get("generation", {}))
    if variant.get("sdpa_backend") == "math":
        attention_kernel = sdpa_kernel(SDPBackend.MATH)
    elif variant.get("sdpa_backend") is None:
        attention_kernel = nullcontext()
    else:
        raise ValueError(f"Unknown sdpa_backend {variant['sdpa_backend']!r}; use math or omit it")
    try:
        with attention_kernel:
            generated = evaluator._run_once(policy, collect_history=False, collect_diagnostics=False)
        record = generated.record
        record["generation_settings"] = dict(config.generation.__dict__)
    finally:
        config.generation = base_generation

    record["variant"] = variant
    record["code_execution"] = evaluator._evalplus(
        record["task_outputs"], config.run.run_dir / "code_execution" / variant["name"]
    )
    return record


def compare_to_reference(reference: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    reference_code = {item["task_id"]: item["executable_code"] for item in reference["task_outputs"]}
    candidate_code = {item["task_id"]: item["executable_code"] for item in candidate["task_outputs"]}
    changed = sorted(task for task in reference_code if reference_code[task] != candidate_code[task])
    flips = compare_base_correctness(reference["code_execution"], candidate["code_execution"])
    return {
        "identical_output": reference["output_hash"] == candidate["output_hash"],
        "tasks_with_changed_code": len(changed),
        "pass_to_fail": flips["base_regression_count"],
        "fail_to_pass": flips["base_recovery_count"],
        "pass_at_1": candidate["code_execution"]["aggregate"]["base_pass_at_1"],
        "changed_task_ids": changed,
    }


def print_table(rows: list[dict[str, Any]], task_count: int) -> None:
    print(f"{'variant':<34} {'identical':>9} {'changed':>9} {'P->F':>5} {'F->P':>5} {'pass@1':>7}")
    for row in rows:
        print(
            f"{row['variant']:<34} {str(row['identical_output']):>9} "
            f"{row['tasks_with_changed_code']:>4}/{task_count:<4} {row['pass_to_fail']:>5} "
            f"{row['fail_to_pass']:>5} {row['pass_at_1']:>7.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--only", help="Comma-separated variant names to run; the report uses all saved ones")
    args = parser.parse_args()

    config = load_kv_policy_experiment_config(args.config)
    reference_name, variants = load_study(args.config)
    # Results go to <run.name>/<GPU type>/ so GPU types never mix.
    config.run.name = f"{config.run.name}/{gpu_folder_name()}"
    print(f"Results: {config.run.run_dir}", flush=True)
    evaluator = None  # the model is loaded only if a variant still needs to run
    variant_dir = config.run.run_dir / "variants"
    variant_dir.mkdir(parents=True, exist_ok=True)

    selected = set(args.only.split(",")) if args.only else {variant["name"] for variant in variants}
    for variant in variants:
        path = variant_dir / f"{variant['name']}.json"
        if variant["name"] not in selected:
            continue
        if path.is_file() and json.loads(path.read_text())["variant"] == variant:
            print(f"{variant['name']}: already done", flush=True)
            continue
        print(f"{variant['name']}: generating", flush=True)
        if evaluator is None:
            evaluator = DiffuCoderKVPolicyEvaluator(config)
        record = run_variant(evaluator, variant)
        path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")

    reference_path = variant_dir / f"{reference_name}.json"
    if not reference_path.is_file():
        print(f"Reference {reference_name} has not run yet; no report written.")
        return
    reference = json.loads(reference_path.read_text())
    rows = []
    for variant in variants:
        path = variant_dir / f"{variant['name']}.json"
        if path.is_file():
            rows.append({"variant": variant["name"], **compare_to_reference(reference, json.loads(path.read_text()))})
    report = {"reference": reference_name, "comparisons": rows}
    (config.run.run_dir / "variance_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_table(rows, len(reference["task_outputs"]))


if __name__ == "__main__":
    main()
