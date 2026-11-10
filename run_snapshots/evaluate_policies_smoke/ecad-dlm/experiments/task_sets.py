"""Build three disjoint 96-task sets from the tasks no study has seen.

The sets are tau confirmation, search and audit (CONSTRAINED_SEARCH.md). Each gets the
same dataset mix, balanced across prompt-length tertiles. Run once, on a CPU; it refuses
to overwrite existing sets, so a set's tasks cannot change after anything has used it.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import yaml

SEED = 20261107
SET_NAMES = ("tau_confirmation", "search", "audit")
PER_SET = {"humaneval": 28, "mbpp": 68}
# The cost table stops at 768 positions, and 256 of those are generated.
MAX_PROMPT_TOKENS = 512


def deal(items: list[Any], lengths: list[int], per_set: int, rng: random.Random) -> list[list[Any]]:
    """Deal items into len(SET_NAMES) equal sets, balanced across length tertiles."""

    order = sorted(range(len(items)), key=lambda index: (lengths[index], str(items[index])))
    sets: list[list[Any]] = [[] for _ in SET_NAMES]
    turn = 0
    for tertile in np.array_split(np.array(order), 3):
        shuffled = tertile.tolist()
        rng.shuffle(shuffled)
        for index in shuffled:
            if min(map(len, sets)) >= per_set:
                break
            while len(sets[turn % len(sets)]) >= per_set:
                turn += 1
            sets[turn % len(sets)].append(items[index])
            turn += 1
    if any(len(chosen) != per_set for chosen in sets):
        raise ValueError(f"Not enough tasks for {len(sets)} sets of {per_set}")
    return sets


def build(config_path: Path, selection_path: Path, output: Path) -> None:
    from transformers import AutoTokenizer

    from language_models.code_tasks_and_response_processing import CodeGenerationTask, render_strict_python_prompt
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    raw = yaml.safe_load(config_path.read_text())
    selection = json.loads(selection_path.read_text())
    unseen = set(selection["remaining_unexposed_ids"])
    paths = {"humaneval": raw["evaluation"]["evalplus_humaneval_path"], "mbpp": raw["evaluation"]["evalplus_mbpp_path"]}
    tokenizer = AutoTokenizer.from_pretrained(raw["model"]["path"], trust_remote_code=True, local_files_only=True)
    tokenizer.padding_side = "left"
    rng = random.Random(SEED)
    sets: dict[str, list[Any]] = {name: [] for name in SET_NAMES}
    details: dict[str, dict[str, Any]] = {}
    for dataset, path in paths.items():
        tasks, lengths = [], []
        for line in Path(path).read_text().splitlines():
            record = json.loads(line)
            task_id = record["task_id"]
            if task_id not in unseen:
                continue
            original = record["prompt"].strip()
            prompt, mode = render_strict_python_prompt(dataset, original, task_id)
            task = CodeGenerationTask(task_id, dataset, prompt, original, mode, entry_point=record.get("entry_point"))
            tokens = prepare_diffucoder_prompt_batch(tokenizer, [prompt], "cpu")[0].shape[1]
            if tokens <= MAX_PROMPT_TOKENS:
                tasks.append(task)
                lengths.append(tokens)
                details[task_id] = {"prompt_tokens": tokens, "dataset": dataset}
        ids = [task.task_id for task in tasks]
        for name, chosen in zip(SET_NAMES, deal(ids, lengths, PER_SET[dataset], rng)):
            sets[name].extend(task for task in tasks if task.task_id in set(chosen))
    if sum(len(tasks) for tasks in sets.values()) != len({task.task_id for tasks in sets.values() for task in tasks}):
        raise ValueError("The sets overlap")
    output.mkdir(parents=True, exist_ok=True)
    for name, tasks in sets.items():
        tasks.sort(key=lambda task: (details[task.task_id]["prompt_tokens"], task.task_id))
        path = output / f"{name}_96_seed{SEED}.jsonl"
        if path.exists():
            raise SystemExit(f"{path} exists; task sets are fixed once built")
        path.write_text("".join(json.dumps(task.to_dict()) + "\n" for task in tasks))
        lengths = [details[task.task_id]["prompt_tokens"] for task in tasks]
        path.with_suffix(".selection.json").write_text(json.dumps({
            "seed": SEED, "set": name, "tasks": len(tasks), "manifest_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "selected": {task.task_id: details[task.task_id] for task in tasks}}, indent=2))
        print(name, len(tasks), Counter(task.dataset for task in tasks), "prompt tokens", min(lengths), int(np.median(lengths)), max(lengths))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Any config with the model and EvalPlus paths")
    parser.add_argument("--selection", type=Path, required=True, help="The proxy recheck's .selection.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.config, args.selection, args.output)


if __name__ == "__main__":
    main()
