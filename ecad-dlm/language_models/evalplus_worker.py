"""Run selected EvalPlus tasks inside the Apptainer image."""

from __future__ import annotations

import argparse
import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from concurrent.futures import ProcessPoolExecutor
import os
from pathlib import Path
import tempfile

from evalplus.data import (
    get_human_eval_plus,
    get_human_eval_plus_hash,
    get_mbpp_plus,
    get_mbpp_plus_hash,
)
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.eval import PASS
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS


def _check_one(arguments):
    return check_correctness(*arguments)


def _counts(status_details, fallback_total: int) -> tuple[str, int, int]:
    status, details = status_details
    details = list(details or [])
    total = len(details) or fallback_total
    passed = sum(bool(value) for value in details)
    if not details and status == PASS:
        passed = total
    return str(status), int(passed), int(total)


def _evalplus_version() -> str | None:
    try:
        return version("evalplus")
    except PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["humaneval", "mbpp"], required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--scope", choices=["base", "full"], default="full")
    args = parser.parse_args()

    samples = [
        json.loads(line)
        for line in args.samples.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not samples:
        raise ValueError("EvalPlus sample file is empty")
    task_ids = [str(sample["task_id"]) for sample in samples]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("EvalPlus samples contain duplicate task IDs")
    loaders = {
        "humaneval": (get_human_eval_plus, get_human_eval_plus_hash, set()),
        "mbpp": (get_mbpp_plus, get_mbpp_plus_hash, MBPP_OUTPUT_NOT_NONE_TASKS),
    }
    load_problems, load_hash, output_not_none = loaders[args.dataset]
    all_problems = load_problems()
    problems = {
        sample["task_id"]: dict(all_problems[sample["task_id"]])
        for sample in samples
    }
    del all_problems
    if args.scope == "base":
        # Do not place Plus inputs or outputs in the candidate execution graph.
        # get_groundtruth still expects the field, so retain it as an empty list.
        for problem in problems.values():
            problem["plus_input"] = []
    dataset_hash = load_hash()
    subset_key = hashlib.sha256(
        (dataset_hash + json.dumps(sorted(problems))).encode("utf-8")
    ).hexdigest()
    groundtruth_key = (
        subset_key
        if args.scope == "full"
        else hashlib.sha256((subset_key + ":base-only-v1").encode("utf-8")).hexdigest()
    )
    expected = get_groundtruth(problems, groundtruth_key, output_not_none)
    if args.scope == "base":
        for problem in problems.values():
            problem.pop("canonical_solution", None)
            problem.pop("contract", None)
        for task_expected in expected.values():
            task_expected.pop("plus", None)
            task_expected.pop("plus_time", None)
        gc.collect()

    arguments = [
        (
            args.dataset,
            index,
            problems[sample["task_id"]],
            sample["solution"],
            expected[sample["task_id"]],
            args.scope == "base",
            False,
            sample["task_id"],
        )
        for index, sample in enumerate(samples)
    ]
    workers = max(1, min(args.parallel, len(samples)))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        checked = list(executor.map(_check_one, arguments))

    task_results = []
    for result in checked:
        problem = problems[result["task_id"]]
        base_status, base_passed, base_total = _counts(
            result["base"], len(problem["base_input"])
        )
        task_results.append(
            {
                "task_id": result["task_id"],
                "base_status": base_status,
                "base_passed": base_passed,
                "base_total": base_total,
            }
        )
        if args.scope == "full":
            plus_status, plus_passed, plus_total = _counts(
                result["plus"], len(problem["plus_input"])
            )
            task_results[-1].update(
                {
                    "plus_status": plus_status,
                    "plus_passed": plus_passed,
                    "plus_total": plus_total,
                }
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": args.dataset,
        "scope": args.scope,
        "evalplus_version": _evalplus_version(),
        "dataset_hash": dataset_hash,
        "subset_key": subset_key,
        "groundtruth_cache_key": groundtruth_key,
        "tasks": task_results,
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=args.output.name + ".", suffix=".tmp", dir=args.output.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        os.chmod(temporary_name, 0o600)
        os.replace(temporary_name, args.output)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


if __name__ == "__main__":
    main()
