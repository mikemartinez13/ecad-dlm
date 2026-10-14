#!/usr/bin/env python3
"""Select EvalPlus coding tasks, render prompts, and save a reusable JSONL dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ecad.language_models.code_tasks_and_response_processing import (
    select_evalplus_code_tasks,
    write_code_task_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/code/humaneval_mbpp_64_strict_python_seed0.jsonl"),
        help="JSONL destination for selected tasks and formatted prompts (relative paths are accepted).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["humaneval", "mbpp"],
        default=["humaneval", "mbpp"],
    )
    parser.add_argument("--max-tasks", type=int, default=64)
    parser.add_argument("--max-tasks-per-dataset", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt-style",
        choices=["strict_python", "opencompass"],
        default="strict_python",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output = args.output.expanduser().resolve()
    tasks = select_evalplus_code_tasks(
        args.datasets,
        max_tasks=args.max_tasks,
        max_tasks_per_dataset=args.max_tasks_per_dataset,
        seed=args.seed,
        prompt_style=args.prompt_style,
    )
    output = write_code_task_dataset(tasks, args.output)
    print(f"Wrote {len(tasks)} coding tasks to {output}")
    for dataset in args.datasets:
        count = sum(task.dataset == dataset for task in tasks)
        print(f"  {dataset}: {count}")


if __name__ == "__main__":
    main()
