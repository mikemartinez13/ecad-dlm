"""Does batched teacher forcing make the search's decisions batch one makes? (BATCHED_TEACHER_FORCING_CHECK.md)

Runs 1 and 2 saved batch-one teacher forcing of every genome on the 96 search tasks,
on an A100 80GB, with the search set and generation settings v3 uses. This check
teacher-forces two of those generations again with dlm_kv_policy/batched_teacher_forcing.py
and compares, under v3's thresholds and FLOPs objective, every number the search reads.

  trajectories  GPU. Dense at batch one on the 96 tasks, keeping the step each position
                was committed. Its tokens must equal the search's dense reference.
  control       GPU. The search's own batch-one evaluate on tasks 0 and 48 of each
                panel generation: do the saved records still reproduce exactly?
  force         GPU. Batched teacher forcing of one panel generation, BATCH tasks per
                forward, in the search's task-record format, with seconds per batch.
                --max-batches limits it (the batch-size timing runs).
  report        No model. EAR and FLOPs differences, feasibility flips, survivors, and
                GPU-hours per generation; applies the pass bar.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.batched_teacher_forcing import record_trajectory, teacher_force_batch
from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline, DenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
from dlm_kv_policy.teacher_forcing import SAVED_KEY, load_top_p_logits
from evolution.constrained_kv_policy_search import (
    GPU, cost_of, evaluate, generation_dir, read_genomes, score, stop_ids)
from evolution.nsga2_kv_policy_search import EvaluatedKVPolicy, environmental_selection
from experiments.ear_validity import answer_length, task_metrics
from utils.results_io import save

RESULTS = Path("/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy")
# Run 1's last generation and run 2's survivors, which v3 starts from.
PANEL = {"v1_generation_007": RESULTS / "constrained_kv_search_v1" / "generation_007",
         "v2_generation_001": RESULTS / "constrained_kv_search_v2" / "generation_001"}
CONTROL_PARTS = 48  # part 0 of 48 is tasks 0 and 48
# Pass bar (BATCHED_TEACHER_FORCING_CHECK.md).
MAX_EAR_DIFFERENCE = 0.001
FLIP_ALLOWANCE = 0.001
BENCH = RESULTS / "search_concurrency_bench" / "report.json"


def load_model(config_path: Path):
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"The check compares with records made on {GPU}")
    return config, evaluator


def trajectories(config_path: Path, output: Path) -> None:
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config, evaluator = load_model(config_path)
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(evaluator.model))
    reference = Path(config.evolution.dense_reference) / "tasks"
    for index, task in enumerate(evaluator.tasks):
        path = output / "trajectories" / f"task_{index:03d}.json"
        if path.exists():
            continue
        prompt, mask = prepare_diffucoder_prompt_batch(evaluator.tokenizer, [task.prompt], config.model.device)
        trajectory = record_trajectory(sampler, prompt, mask, config.generation)
        saved = json.loads((reference / f"task_{index:03d}.json").read_text())
        if saved["task_id"] != task.task_id or saved[SAVED_KEY]["dense_token_ids"] != trajectory["dense_token_ids"]:
            raise RuntimeError(f"Task {index}: dense differs from the search's dense reference")
        save(path, {"task_id": task.task_id, **trajectory})
        print(f"trajectory {index} {task.task_id}", flush=True)


def control(config_path: Path, output: Path) -> None:
    for name, source in PANEL.items():
        root = output / "control" / name
        (root / "generation_000").mkdir(parents=True, exist_ok=True)
        shutil.copy(source / "genomes.json", root / "generation_000" / "genomes.json")
        evaluate(config_path, root, 0, 0, CONTROL_PARTS)


def force(config_path: Path, output: Path, name: str, batch_size: int, max_batches: int | None, folder: str) -> None:
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config, evaluator = load_model(config_path)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    stops = stop_ids(tokenizer)
    genomes = read_genomes(PANEL[name])
    top_p_logits = load_top_p_logits(model)
    root = output / folder
    trajectory = {index: json.loads((output / "trajectories" / f"task_{index:03d}.json").read_text())
                  for index in range(len(evaluator.tasks))}
    # Similar prompt lengths share a batch, so little of each forward is padding.
    lengths = {index: len(tokenizer.apply_chat_template(
        [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": task.prompt}],
        add_generation_prompt=True, tokenize=True)) for index, task in enumerate(evaluator.tasks)}
    order = sorted(lengths, key=lambda index: (lengths[index], index))
    batches = [order[start:start + batch_size] for start in range(0, len(order), batch_size)][:max_batches]
    timing_path = root / "timing.json"
    timing = json.loads(timing_path.read_text()) if timing_path.exists() else []
    for batch in batches:
        paths = {index: root / "tasks" / f"task_{index:03d}.json" for index in batch}
        if all(path.exists() for path in paths.values()):
            continue
        tasks = [evaluator.tasks[index] for index in batch]
        if any(trajectory[index]["task_id"] != task.task_id for index, task in zip(batch, tasks)):
            raise RuntimeError("Trajectories are not the config's tasks")
        prompts, mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt for task in tasks], config.model.device)
        answer_ends = [answer_length(trajectory[index]["dense_token_ids"], stops) for index in batch]
        backends = {genome.name: BatchedDiffuCoderKVBackend(model, genome.policy(stops)) for genome in genomes}
        torch.cuda.synchronize()
        started = time.perf_counter()
        results = teacher_force_batch(model, BatchedDenseDiffuCoderBaseline(model), backends, prompts, mask,
                                      [trajectory[index] for index in batch], config.generation, top_p_logits,
                                      answer_ends=answer_ends)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        for index, task, answer_end, result, length in zip(batch, tasks, answer_ends, results, mask.sum(-1).tolist()):
            policies = {}
            for policy, steps in result["policies"].items():
                outside = sum(position < answer_end for row in steps for position in row.get("outside_positions", []))
                policies[policy] = {**task_metrics(steps, answer_end), "answer_outside": outside,
                                    "rows": result["rows"][policy], "flops": result["flops"][policy]}
            save(paths[index], {"task_id": task.task_id, "index": index, "prompt_tokens": length,
                                "answer_end": answer_end, "dense_flops": result["dense_flops"], "policies": policies})
        timing.append({"tasks": batch, "batch_size": len(batch), "padded_prompt": prompts.shape[1],
                       "seconds": seconds, "peak_gb": torch.cuda.max_memory_allocated() / 2**30})
        save(timing_path, timing)
        print(f"{name}: batch of {len(batch)} (prompt {prompts.shape[1]}) in {seconds:.0f} s, "
              f"{seconds / len(batch):.1f} s per task", flush=True)
        del backends, results
        torch.cuda.empty_cache()


def records(directory: Path) -> list[dict[str, Any]]:
    return [json.loads(path.read_text()) for path in sorted(directory.glob("task_*.json"))]


def gpu_hours_per_generation(timing: list[dict[str, Any]], tasks: int) -> float | None:
    if not timing:
        return None
    return tasks * sum(row["seconds"] for row in timing) / sum(row["batch_size"] for row in timing) / 3600


def compare_generation(name: str, source: Path, batched: list[dict[str, Any]], settings: Any) -> dict[str, Any]:
    single = records(source / "tasks")
    if [row["task_id"] for row in single] != [row["task_id"] for row in batched]:
        raise RuntimeError(f"{name}: batched records do not cover the saved tasks")
    genomes = read_genomes(source)
    per_policy, flips = {}, []
    one = {genome: score(genome, single, settings) for genome in genomes}
    many = {genome: score(genome, batched, settings) for genome in genomes}
    for genome in genomes:
        policy = genome.name
        task_ear = np.array([[s["policies"][policy]["ear"], b["policies"][policy]["ear"]] for s, b in zip(single, batched)])
        a, b = one[genome], many[genome]
        per_policy[policy] = {
            "ear_difference": b.constraints["ear"] - a.constraints["ear"],
            "ear_lower_difference": b.constraints["ear_lower_95"] - a.constraints["ear_lower_95"],
            "tail_ear_difference": b.constraints["tail_ear"] - a.constraints["tail_ear"],
            "coverage_difference": b.constraints["answer_coverage"] - a.constraints["answer_coverage"],
            "max_task_ear_difference": float(np.nanmax(np.abs(task_ear[:, 1] - task_ear[:, 0]))),
            "flops_identical": all(s["policies"][policy]["flops"] == m["policies"][policy]["flops"]
                                   for s, m in zip(single, batched)),
            "rows_identical": all(s["policies"][policy]["rows"] == m["policies"][policy]["rows"]
                                  for s, m in zip(single, batched)),
            "feasible_batch_one": not a.violation, "feasible_batched": not b.violation,
        }
        if bool(a.violation) != bool(b.violation):
            # Distance of the batch-one values from the nearest threshold.
            margin = min(abs(a.constraints["ear_lower_95"] - settings.min_ear),
                         abs(a.constraints["tail_ear"] - settings.min_tail_ear),
                         abs(a.constraints["answer_coverage"] - settings.min_answer_coverage))
            flips.append({"policy": policy, "batch_one_violation": a.violation, "batched_violation": b.violation,
                          "threshold_margin": margin, "allowed": margin <= FLIP_ALLOWANCE})
    previous = generation_dir(source.parent, int(source.name.split("_")[1]) - 1) / "summary.json"
    survivors = [replace(item, cost=cost_of(item, settings)) for item in
                 map(EvaluatedKVPolicy.from_dict, json.loads(previous.read_text())["population"])]
    selected = {label: sorted(item.genome.name for item in environmental_selection(survivors + list(scored.values()),
                                                                                   settings.population_size))
                for label, scored in (("batch_one", one), ("batched", many))}
    return {"policies": per_policy, "feasibility_flips": flips, "survivors": selected,
            "same_survivors": selected["batch_one"] == selected["batched"],
            "max_ear_difference": max(abs(row["ear_difference"]) for row in per_policy.values()),
            "max_ear_lower_difference": max(abs(row["ear_lower_difference"]) for row in per_policy.values()),
            "max_tail_ear_difference": max(abs(row["tail_ear_difference"]) for row in per_policy.values()),
            "flops_identical": all(row["flops_identical"] for row in per_policy.values())}


def report(config_path: Path, output: Path) -> None:
    config = load_kv_policy_experiment_config(config_path)
    settings, tasks = config.evolution, config.evaluation.max_prompts
    result: dict[str, Any] = {"generations": {}, "control": {}, "timing": {}}
    for name, source in PANEL.items():
        batched = records(output / name / "tasks")
        if len(batched) != tasks:
            raise SystemExit(f"{name}: {len(batched)} of {tasks} tasks forced")
        result["generations"][name] = compare_generation(name, source, batched, settings)
        reproduced = records(output / "control" / name / "generation_000" / "tasks")
        saved = {row["index"]: row for row in records(source / "tasks")}
        result["control"][name] = {
            "tasks": [row["index"] for row in reproduced],
            # As JSON text, since a task without answer commits has NaN EAR.
            "identical": bool(reproduced) and all(
                json.dumps(row["policies"]) == json.dumps(saved[row["index"]]["policies"]) for row in reproduced)}
    for folder in sorted(path.name for path in output.iterdir() if path.is_dir()):
        timing_path = output / folder / "timing.json"
        if timing_path.exists():
            timing = json.loads(timing_path.read_text())
            result["timing"][folder] = {
                "batch_size": max(row["batch_size"] for row in timing),
                "seconds_per_task": sum(row["seconds"] for row in timing) / sum(row["batch_size"] for row in timing),
                "gpu_hours_per_generation": gpu_hours_per_generation(timing, tasks),
                "peak_gb": max(row["peak_gb"] for row in timing)}
    if BENCH.exists():
        bench = json.loads(BENCH.read_text())
        best = max(bench.values(), key=lambda row: row["tasks_per_gpu_hour"])
        result["timing"]["batch_one_benchmark"] = {"gpu_hours_per_generation": tasks / best["tasks_per_gpu_hour"],
                                                   "source": str(BENCH)}
    generations = result["generations"].values()
    result["pass"] = {
        "ear_within": all(row["max_ear_difference"] <= MAX_EAR_DIFFERENCE and row["max_ear_lower_difference"]
                          <= MAX_EAR_DIFFERENCE and row["max_tail_ear_difference"] <= MAX_EAR_DIFFERENCE
                          for row in generations),
        "feasibility_flips_only_at_thresholds": all(flip["allowed"] for row in generations
                                                    for flip in row["feasibility_flips"]),
        "same_survivors": all(row["same_survivors"] for row in generations),
        "flops_identical": all(row["flops_identical"] for row in generations),
        "control_reproduces": all(row["identical"] for row in result["control"].values()),
    }
    result["pass"]["all"] = all(result["pass"].values())
    save(output / "report.json", result)
    for name, row in result["generations"].items():
        print(f"{name}: max |dEAR| {row['max_ear_difference']:.2e}, lower {row['max_ear_lower_difference']:.2e}, "
              f"tail {row['max_tail_ear_difference']:.2e}; {len(row['feasibility_flips'])} feasibility flips; "
              f"survivors {'same' if row['same_survivors'] else 'DIFFER'}; FLOPs "
              f"{'identical' if row['flops_identical'] else 'DIFFER'}", flush=True)
    for folder, row in result["timing"].items():
        print(f"{folder}: {row['gpu_hours_per_generation']:.2f} GPU-h per generation", flush=True)
    print("Pass:", result["pass"], flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["trajectories", "control", "force", "report"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", choices=sorted(PANEL), help="Panel generation to force")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--folder", help="Output folder; default is the generation's name")
    args = parser.parse_args()
    if args.mode == "trajectories":
        trajectories(args.config, args.output)
    elif args.mode == "control":
        control(args.config, args.output)
    elif args.mode == "force":
        if args.generation is None:
            parser.error("force needs --generation")
        force(args.config, args.output, args.generation, args.batch_size, args.max_batches,
              args.folder or args.generation)
    else:
        report(args.config, args.output)


if __name__ == "__main__":
    main()
