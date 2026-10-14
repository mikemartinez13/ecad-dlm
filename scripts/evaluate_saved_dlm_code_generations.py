#!/usr/bin/env python3
"""Score saved DLM code generations with sandboxed EvalPlus tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecad.language_models.code_tasks_and_response_processing import (  # noqa: E402
    CODE_POSTPROCESSOR_VERSION,
    load_code_task_dataset,
    make_code_task_output,
)
from ecad.language_models.code_execution import (  # noqa: E402
    run_evalplus_apptainer,
)
from ecad.language_models.experiment_artifact_paths import (  # noqa: E402
    resolve_experiment_result_path,
)
from ecad.language_models.residual_cache_experiment_config import (  # noqa: E402
    load_dlm_residual_cache_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Read run directory, container, and benchmark paths from an experiment YAML")
    parser.add_argument("--run-dir", type=Path, help="Override the run directory from --config")
    parser.add_argument("--artifact", type=Path, help="Saved response artifact; with --config, relative paths are resolved within the run (default: baseline.json)")
    parser.add_argument("--image", help="Override the EvalPlus SIF image from --config")
    parser.add_argument(
        "--humaneval-data",
        type=Path,
        help="Absolute, pinned HumanEval+ JSONL file to bind read-only",
    )
    parser.add_argument(
        "--mbpp-data",
        type=Path,
        help="Absolute, pinned MBPP+ JSONL file to bind read-only",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--parallel", type=int)
    parser.add_argument("--apptainer-bin")
    parser.add_argument(
        "--use-saved-task-outputs",
        action="store_true",
        help=(
            "Use executable_code embedded in the artifact instead of rebuilding "
            "it from raw responses. Intended only for reproducing historical runs."
        ),
    )
    args = parser.parse_args(argv)
    config = load_dlm_residual_cache_config(args.config) if args.config is not None else None
    if config is not None:
        if args.run_dir is None:
            args.run_dir = config.run_dir
        if args.image is None:
            args.image = config.evaluation.evalplus_image
        if args.humaneval_data is None and config.evaluation.evalplus_humaneval_path:
            args.humaneval_data = Path(config.evaluation.evalplus_humaneval_path)
        if args.mbpp_data is None and config.evaluation.evalplus_mbpp_path:
            args.mbpp_data = Path(config.evaluation.evalplus_mbpp_path)
        if args.parallel is None:
            args.parallel = config.evaluation.evalplus_parallel
        if args.apptainer_bin is None:
            args.apptainer_bin = config.evaluation.evalplus_apptainer_bin
    for field in ("run_dir", "image", "humaneval_data", "mbpp_data"):
        if not getattr(args, field):
            parser.error(f"--{field.replace('_', '-')} is required unless provided by --config")
    args.run_dir = resolve_experiment_result_path(args.run_dir).resolve()
    if args.artifact is None:
        args.artifact = args.run_dir / "baseline.json"
    elif config is not None and not args.artifact.is_absolute():
        args.artifact = args.run_dir / args.artifact
    args.artifact = resolve_experiment_result_path(args.artifact).resolve()
    if args.output is not None and config is not None and not args.output.is_absolute():
        args.output = args.run_dir / args.output
    if args.parallel is None:
        args.parallel = 8
    if args.apptainer_bin is None:
        args.apptainer_bin = "apptainer"
    return args


def load_saved_code_task_outputs(
    run_dir: Path,
    artifact: Path,
    *,
    use_saved_task_outputs: bool = False,
) -> tuple[list[dict], str]:
    """Load task outputs, rebuilding derived code from raw text by default."""

    run_dir = resolve_experiment_result_path(run_dir).resolve()
    artifact = resolve_experiment_result_path(artifact).resolve()
    dataset_path = run_dir / "code_tasks.json"
    if not dataset_path.is_file():
        # Historical runs stored the identical task records under this name.
        dataset_path = run_dir / "calibration_manifest.json"
    tasks = load_code_task_dataset(dataset_path)
    data = json.loads(artifact.read_text(encoding="utf-8"))
    data = data.get("metrics", data)
    saved_outputs = data.get("task_outputs")
    if use_saved_task_outputs:
        if not saved_outputs:
            raise ValueError(f"No saved task_outputs in {artifact}")
        return saved_outputs, "saved_task_outputs"

    responses = data.get("responses")
    if responses is None:
        responses = data.get("decoded")
    if responses is None:
        if saved_outputs:
            return saved_outputs, "saved_task_outputs_fallback_no_raw_responses"
        raise ValueError(f"No raw responses or saved task_outputs in {artifact}")
    if len(responses) != len(tasks):
        raise ValueError(
            f"Response/task count mismatch in {artifact}: "
            f"{len(responses)} responses for {len(tasks)} tasks"
        )
    return (
        [
            make_code_task_output(task, response)
            for task, response in zip(tasks, responses)
        ],
        "rebuilt_from_raw_responses",
    )


def summarize_postprocessing(task_outputs: list[dict], source: str) -> dict:
    syntax_checked = [
        output for output in task_outputs if output.get("syntax_valid") is not None
    ]
    invalid = [
        output["task_id"]
        for output in syntax_checked
        if output["syntax_valid"] is False
    ]
    versions = sorted(
        {
            str(output.get("postprocessing", {}).get("version"))
            for output in task_outputs
            if output.get("postprocessing", {}).get("version")
        }
    )
    extractors = sorted(
        {
            str(output.get("postprocessing", {}).get("extractor"))
            for output in task_outputs
            if output.get("postprocessing", {}).get("extractor")
        }
    )
    sources = sorted(
        {
            str(output.get("postprocessing", {}).get("source"))
            for output in task_outputs
            if output.get("postprocessing", {}).get("source")
        }
    )
    return {
        "source": source,
        "current_version": CODE_POSTPROCESSOR_VERSION,
        "versions_present": versions,
        "extractors_present": extractors,
        "sources": sources,
        "task_count": len(task_outputs),
        "syntax_checked_count": len(syntax_checked),
        "syntax_valid_count": sum(
            output["syntax_valid"] is True for output in syntax_checked
        ),
        "syntax_invalid_count": len(invalid),
        "syntax_invalid_task_ids": invalid,
        "tasks": [
            {
                "task_id": output["task_id"],
                "dataset": output["dataset"],
                "syntax_valid": output.get("syntax_valid"),
                "extraction_strategy": output.get("postprocessing", {}).get(
                    "extraction_strategy"
                ),
                "reconstruction_strategy": output.get("postprocessing", {}).get(
                    "reconstruction_strategy"
                ),
            }
            for output in task_outputs
        ],
    }


def main() -> None:
    args = parse_args()
    run_dir = resolve_experiment_result_path(args.run_dir).resolve()
    artifact = resolve_experiment_result_path(args.artifact).resolve()
    output = (
        resolve_experiment_result_path(args.output).resolve()
        if args.output is not None
        else artifact.with_suffix(".code_eval.json")
    )
    task_outputs, task_output_source = load_saved_code_task_outputs(
        run_dir,
        artifact,
        use_saved_task_outputs=args.use_saved_task_outputs,
    )
    postprocessing = summarize_postprocessing(task_outputs, task_output_source)
    print(
        "Postprocessing: "
        f"source={task_output_source}, "
        f"version={CODE_POSTPROCESSOR_VERSION}, "
        f"syntax_valid={postprocessing['syntax_valid_count']}/"
        f"{postprocessing['syntax_checked_count']}"
    )
    if postprocessing["syntax_invalid_task_ids"]:
        print(
            "Syntax-invalid tasks: "
            + ", ".join(postprocessing["syntax_invalid_task_ids"])
        )
    result = run_evalplus_apptainer(
        task_outputs,
        image=args.image,
        work_dir=run_dir / "code_execution" / artifact.stem,
        parallel=args.parallel,
        apptainer_bin=args.apptainer_bin,
        dataset_paths={
            "humaneval": args.humaneval_data,
            "mbpp": args.mbpp_data,
        },
    )
    result.update(
        {
            "artifact": str(artifact),
            "run_dir": str(run_dir),
            "postprocessing": postprocessing,
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for dataset, dataset_result in result["datasets"].items():
        aggregate = dataset_result["aggregate"]
        print(
            f"{dataset}: {len(dataset_result['tasks'])} tasks, "
            f"base_pass@1={aggregate['base_pass_at_1']:.4f}, "
            f"plus_pass@1={aggregate['plus_pass_at_1']:.4f}, "
            f"base_test_fraction={aggregate['base_pass_fraction']:.4f}, "
            f"extra_test_fraction={aggregate['plus_pass_fraction']:.4f}"
        )
    print(json.dumps(result["aggregate"], indent=2))
    print(f"Saved EvalPlus results to {output}")


if __name__ == "__main__":
    main()
