"""NSGA-II over cache policies under a fidelity constraint, minimizing predicted batched time.

Fidelity is a bound, not an objective (EAR_VALIDITY_STUDY.md): every policy meeting
it counts as equally faithful, and Deb's constrained domination then ranks policies
on predicted time alone. A policy is feasible if the 95% lower bound of its mean EAR,
its tail EAR and its live answer coverage all reach the thresholds in the config.
Infeasible ones rank by total shortfall, so the search can still climb toward the
feasible region. Predicted time is the cost model's model-compute estimate, which
failed its absolute calibration; only its ranking is relied on.

A generation is one sharded replay of every genome over the tasks:

  advance    CPU. Writes generation 0's genomes, or scores the finished generation,
             selects survivors and writes the next genomes. Safe to rerun.
  evaluate   GPU, A100 80GB. Replays one shard of the current generation.
  check      GPU, A100 80GB. Whether each genome of a generation decodes token for
             token the same in the batched backend, at batch one, on one task.
             Predicted time assumes that backend can run these policies.
  audit      GPU, A100 80GB. Free-running pass@1 against dense for the fastest
             feasible policies. EAR was validated only on depth, refresh and
             masked-reuse policies, so the policies using other genes are chosen
             here. It generates dense itself, so a config with other tasks works.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dlm_kv_policy.dense_diffucoder_baseline import DenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.forced_trajectory_replay import load_top_p_logits, replay_task
from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
from experiments.batched_cost_model import predicted_seconds
from experiments.ear_validity import TAIL_QUANTILE, answer_length, task_metrics
from experiments.proxy_recheck import assert_same, backend, generation_settings, save

from .nsga2_kv_policy_search import (
    EvaluatedKVPolicy,
    KVRefreshPolicyGenome,
    environmental_selection,
    initial_population,
    offspring_population,
)

GPU = "NVIDIA A100-SXM4-80GB"
SEED = 20261107
BOOTSTRAPS = 2000
AUDIT_NET_LOSS_MARGIN = 2


def generation_dir(output: Path, generation: int) -> Path:
    return output / f"generation_{generation:03d}"


def stop_ids(tokenizer: Any) -> set[int]:
    return {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}


def read_genomes(directory: Path) -> list[KVRefreshPolicyGenome]:
    return [KVRefreshPolicyGenome.from_dict(raw) for raw in json.loads((directory / "genomes.json").read_text())]


def evaluate(config_path: Path, output: Path, generation: int, part: int, parts: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, seed_everything
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Search requires {GPU}: thresholds and costs were measured there")
    stops = stop_ids(tokenizer)
    directory = generation_dir(output, generation)
    genomes = read_genomes(directory)
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)
    reference = Path(config.evolution.dense_reference)
    for index in range(part, len(evaluator.tasks), parts):
        path = directory / "tasks" / f"task_{index:03d}.json"
        if path.exists():
            continue
        task = evaluator.tasks[index]
        saved = json.loads((reference / "tasks" / f"task_{index:03d}.json").read_text())
        if saved["task_id"] != task.task_id:
            raise RuntimeError(f"Task {index} is not the dense reference's task")
        dense_ids = saved["replay"]["dense_token_ids"]
        answer_end = answer_length(dense_ids, stops)
        seed_everything(config.generation.seed + index)
        prompt, mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], config.model.device)
        backends = {genome.name: backend(model, genome.policy(stops)) for genome in genomes}
        rows: dict[str, list[dict[str, int]]] = {name: [] for name in backends}
        for name, instance in backends.items():
            forward = instance.forward_mask_logits

            def record(_forward=forward, _name=name, **inputs):
                result = _forward(**inputs)
                decision = result.decision
                rows[_name].append({"active": int(decision.active.sum()), "masked": result.mask_logits.shape[0],
                                    "full_depth": int((decision.active & decision.refresh_depth.le(1)).sum())})
                return result

            instance.forward_mask_logits = record
        replay = replay_task(model, sampler, backends, prompt, mask, config.generation, top_p_logits,
                             answer_end=answer_end)
        if replay["dense_token_ids"] != dense_ids:
            raise RuntimeError(f"Task {index}: dense differs from the dense reference")
        policies = {}
        for name, steps in replay["policies"].items():
            outside = sum(position < answer_end for row in steps for position in row.get("outside_positions", []))
            policies[name] = {**task_metrics(steps, answer_end), "answer_outside": outside, "rows": rows[name],
                              "flops": replay["flops"][name]}
        save(path, {"task_id": task.task_id, "index": index, "prompt_tokens": prompt.shape[1], "answer_end": answer_end,
                    "dense_flops": replay["dense_flops"], "policies": policies})
        print(f"generation {generation} part {part}: {index} {task.task_id} complete", flush=True)


def check(config_path: Path, output: Path, generation: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    stops = stop_ids(tokenizer)
    settings = {**generation_settings(config), "collect_history": True}
    prompt, mask = prepare_diffucoder_prompt_batch(tokenizer, [evaluator.tasks[0].prompt], config.model.device)
    results = {}
    for genome in read_genomes(generation_dir(output, generation)):
        expected = DiffuCoderDenoisingSampler(backend(model, genome.policy(stops))).generate(prompt, attention_mask=mask, **settings)
        actual = DiffuCoderDenoisingSampler(backend(model, genome.policy(stops), batched=True)).generate(prompt, attention_mask=mask, **settings)
        try:
            assert_same(actual, expected, genome.name)
            results[genome.name] = "identical"
        except RuntimeError:
            results[genome.name] = "DIFFERENT"
        print(f"{results[genome.name]} {genome.name}", flush=True)
    save(generation_dir(output, generation) / "batched_check.json", results)


def score(genome: KVRefreshPolicyGenome, records: list[dict[str, Any]], settings: Any, table: dict[str, Any]) -> EvaluatedKVPolicy:
    name = genome.name
    ear = np.array([record["policies"][name]["ear"] for record in records])
    samples = np.random.default_rng(SEED).integers(0, len(ear), size=(BOOTSTRAPS, len(ear)))
    ear_lower = float(np.nanquantile(np.nanmean(ear[samples], axis=1), 0.05))
    tail_ear = float(np.nanquantile(ear, TAIL_QUANTILE))
    answer_commits = sum(record["answer_end"] for record in records)
    coverage = 1 - sum(record["policies"][name]["answer_outside"] for record in records) / answer_commits
    violation = (max(0.0, settings.min_ear - ear_lower) + max(0.0, settings.min_tail_ear - tail_ear)
                 + max(0.0, settings.min_answer_coverage - coverage))
    tasks = [{"task_id": record["task_id"], "prompt_tokens": record["prompt_tokens"], "rows": record["policies"][name]["rows"]}
             for record in records]
    seconds, dense_seconds = predicted_seconds(table, tasks, settings.objective_batch_size, genome.refresh_depth)
    compute = sum(record["policies"][name]["flops"] for record in records) / sum(record["dense_flops"] for record in records)
    return EvaluatedKVPolicy(
        genome=genome, quality_loss=1 - float(np.nanmean(ear)), compute_fraction=compute, predicted_seconds=seconds,
        violation=violation,
        constraints={"ear": float(np.nanmean(ear)), "ear_lower_95": ear_lower, "tail_ear": tail_ear,
                     "answer_coverage": coverage, "predicted_speedup": dense_seconds / seconds,
                     "position_flip_rate": float(np.nanmean([record["policies"][name]["position_flip_rate"] for record in records])),
                     "masked_ear": float(np.nanmean([record["policies"][name]["masked_ear"] for record in records]))},
    )


def advance(config_path: Path, output: Path) -> None:
    config = load_kv_policy_experiment_config(config_path)
    settings = config.evolution
    if settings.quality_objective != "constrained_batched_time":
        raise ValueError("This search needs evolution.quality_objective: constrained_batched_time")
    if not generation_dir(output, 0).exists():
        save(generation_dir(output, 0) / "genomes.json", [genome.to_dict() for genome in initial_population(settings)])
        print("Wrote generation 0", flush=True)
        return
    generation = max(int(path.name.split("_")[1]) for path in output.glob("generation_*"))
    directory = generation_dir(output, generation)
    if (directory / "summary.json").exists():
        print(f"Generation {generation} is already scored; the search is finished", flush=True)
        return
    records = [json.loads(path.read_text()) for path in sorted((directory / "tasks").glob("task_*.json"))]
    if len(records) != config.evaluation.max_prompts:
        raise SystemExit(f"Generation {generation}: {len(records)} of {config.evaluation.max_prompts} tasks finished")
    table = json.loads(Path(settings.cost_table).read_text())
    evaluated = [score(genome, records, settings, table) for genome in read_genomes(directory)]
    population = evaluated
    seen = {item.genome for item in evaluated}
    if generation:
        previous = json.loads((generation_dir(output, generation - 1) / "summary.json").read_text())
        population = environmental_selection([EvaluatedKVPolicy.from_dict(raw) for raw in previous["population"]] + evaluated,
                                             settings.population_size)
        seen |= {KVRefreshPolicyGenome.from_dict(raw) for raw in previous["seen"]}
    feasible = sorted((item for item in population if not item.violation), key=lambda item: item.predicted_seconds)
    save(directory / "summary.json", {
        "generation": generation, "evaluated": [item.to_dict() for item in evaluated],
        "population": [item.to_dict() for item in population], "feasible": len(feasible),
        "seen": [genome.to_dict() for genome in sorted(seen, key=lambda item: item.name)],
    })
    print(f"Generation {generation}: {sum(not item.violation for item in evaluated)}/{len(evaluated)} feasible", flush=True)
    for item in feasible[:5]:
        print(f"  {item.constraints['predicted_speedup']:.2f}x {item.genome.name}", flush=True)
    if generation + 1 < settings.generations:
        genomes = offspring_population(population, settings, generation + 1, forbidden=seen)
        save(generation_dir(output, generation + 1) / "genomes.json", [genome.to_dict() for genome in genomes])


def outside_validation(genome: KVRefreshPolicyGenome) -> bool:
    """Whether a policy uses a gene EAR was not validated on."""

    return bool(genome.prompt_refresh_interval or genome.local_refresh_radius or genome.padding_window or genome.entropy_threshold)


def audit_choices(evaluated: list[EvaluatedKVPolicy], top: int) -> list[EvaluatedKVPolicy]:
    """The fastest feasible policies that use unvalidated genes, plus the fastest overall."""

    feasible = sorted((item for item in evaluated if not item.violation), key=lambda item: item.predicted_seconds)
    chosen = [item for item in feasible if outside_validation(item.genome)][:top]
    chosen += [item for item in feasible[:1] if item not in chosen]
    return chosen


def audit(config_path: Path, output: Path, top: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    from experiments.output_variance_study import compare_to_reference

    config = load_kv_policy_experiment_config(config_path)
    evaluated: dict[KVRefreshPolicyGenome, EvaluatedKVPolicy] = {}
    for path in sorted(output.glob("generation_*/summary.json")):
        for raw in json.loads(path.read_text())["evaluated"]:
            item = EvaluatedKVPolicy.from_dict(raw)
            evaluated[item.genome] = item
    chosen = audit_choices(list(evaluated.values()), top)
    if not chosen:
        raise SystemExit("No feasible policy to audit")
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Audit requires {GPU}")
    stops = stop_ids(evaluator.tokenizer)

    def generate(name: str, policy: Any) -> dict[str, Any]:
        print(f"audit: generating {name}", flush=True)
        record = evaluator._run_once(policy, collect_history=False, collect_diagnostics=False).record
        record["code_execution"] = evaluator._evalplus(record["task_outputs"], output / "audit" / "execution" / name)
        return record

    dense = generate("dense", config.policies["dense"])
    rows = []
    for item in chosen:
        record = generate(item.genome.name, item.genome.policy(stops))
        result = compare_to_reference(dense, record)
        net = result["pass_to_fail"] - result["fail_to_pass"]
        rows.append({"policy": item.genome.name, "genome": item.genome.to_dict(), "outside_validation": outside_validation(item.genome),
                     "predicted_speedup": item.constraints["predicted_speedup"], "ear": item.constraints["ear"],
                     "compute_fraction": record["compute_fraction"], "net_loss": net, "within_margin": net <= AUDIT_NET_LOSS_MARGIN,
                     **{key: result[key] for key in ("pass_at_1", "pass_to_fail", "fail_to_pass", "tasks_with_changed_code")}})
        save(output / "audit" / "audit.json", {"dense_pass_at_1": dense["code_execution"]["aggregate"]["base_pass_at_1"], "policies": rows})
    for row in rows:
        print(f"{row['policy']}: {row['predicted_speedup']:.2f}x, pass@1 {row['pass_at_1']:.3f}, net loss {row['net_loss']}, "
              f"{'within' if row['within_margin'] else 'OUTSIDE'} margin", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["advance", "evaluate", "check", "audit"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "advance":
        advance(args.config, args.output)
    elif args.mode == "evaluate":
        evaluate(args.config, args.output, args.generation, args.part, args.parts)
    elif args.mode == "check":
        check(args.config, args.output, args.generation)
    else:
        audit(args.config, args.output, args.top)


if __name__ == "__main__":
    main()
