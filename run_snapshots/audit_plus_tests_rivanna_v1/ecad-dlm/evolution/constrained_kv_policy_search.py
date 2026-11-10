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
  dense      GPU, A100 80GB. Generates dense once, for the answer lengths replay needs.
  evaluate   GPU, A100 80GB. Replays one shard of the current generation.
  check      GPU, A100 80GB. Whether each genome of a generation decodes token for
             token the same in the batched backend, at batch one, on one task.
             Predicted time assumes that backend can run these policies.
  audit      GPU, A100 80GB, on the audit set. Free-running pass@1 against dense and
             measured batch-8 time, for the fastest feasible policies, the fastest
             feasible seed, and the policies using genes EAR was not validated on.
             Applies the success criterion in CONSTRAINED_SEARCH.md.
  plus       No model. Reruns the audited programs under EvalPlus's full scope, base
             plus extra tests, and compares each policy with dense.
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
from experiments.ear_validity import TAIL_QUANTILE, answer_length, pass_fraction, task_metrics
from experiments.proxy_recheck import assert_same, backend, generation_settings, save

from .nsga2_kv_policy_search import (
    SEED_GENOMES,
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
# Success criterion: measured batch-8 speedup over batched dense, and over the best seed.
MIN_SPEEDUP_OVER_DENSE = 2.0
MIN_SPEEDUP_OVER_SEED = 1.10


def generation_dir(output: Path, generation: int) -> Path:
    return output / f"generation_{generation:03d}"


def stop_ids(tokenizer: Any) -> set[int]:
    return {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}


def read_genomes(directory: Path) -> list[KVRefreshPolicyGenome]:
    return [KVRefreshPolicyGenome.from_dict(raw) for raw in json.loads((directory / "genomes.json").read_text())]


def dense_reference(config_path: Path) -> None:
    """Write dense's tokens for the search tasks, where evaluate reads them."""

    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, seed_everything
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Search requires {GPU}")
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(evaluator.model))
    settings = generation_settings(config)
    for index, task in enumerate(evaluator.tasks):
        path = Path(config.evolution.dense_reference) / "tasks" / f"task_{index:03d}.json"
        if path.exists():
            continue
        seed_everything(config.generation.seed + index)
        prompt, mask = prepare_diffucoder_prompt_batch(evaluator.tokenizer, [task.prompt], config.model.device)
        result = sampler.generate(prompt, attention_mask=mask, **settings)
        save(path, {"task_id": task.task_id, "replay": {"dense_token_ids": result.sequences[0, prompt.shape[1]:].cpu().tolist()}})
        print(f"dense {index} {task.task_id}", flush=True)


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
    """The fastest feasible policies that use unvalidated genes, plus the fastest overall and the fastest seed."""

    feasible = sorted((item for item in evaluated if not item.violation), key=lambda item: item.predicted_seconds)
    chosen = [item for item in feasible if outside_validation(item.genome)][:top]
    seeds = [item for item in feasible if item.genome in SEED_GENOMES]
    for item in (*feasible[:1], *seeds[:1]):
        if item not in chosen:
            chosen.append(item)
    return chosen


def success(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the criterion to audited policies: faithful, 2x over dense, and 10% over the best seed."""

    faithful = [row for row in rows if row["within_margin"]]
    seeds = [row["measured_speedup"] for row in faithful if row["seed"]]
    best_seed = max(seeds, default=None)
    winners = [row["policy"] for row in faithful if not row["seed"] and row["measured_speedup"] >= MIN_SPEEDUP_OVER_DENSE
               and (best_seed is None or row["measured_speedup"] >= MIN_SPEEDUP_OVER_SEED * best_seed)]
    return {"success": bool(winners), "winners": winners, "best_faithful_seed_speedup": best_seed}


def audit(config_path: Path, output: Path, top: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator
    from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline
    from experiments.output_variance_study import compare_to_reference
    from experiments.tau_confirmation import measured_batched_seconds

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
    settings = generation_settings(config)

    def seconds(make_backend: Any) -> float:
        return measured_batched_seconds(evaluator.model, evaluator.tokenizer, evaluator.tasks, make_backend, settings, config.model.device)

    dense_seconds = seconds(lambda: BatchedDenseDiffuCoderBaseline(evaluator.model))
    rows = []
    for item in chosen:
        record = generate(item.genome.name, item.genome.policy(stops))
        result = compare_to_reference(dense, record)
        net = result["pass_to_fail"] - result["fail_to_pass"]
        measured = dense_seconds / seconds(lambda: backend(evaluator.model, item.genome.policy(stops), batched=True))
        rows.append({"policy": item.genome.name, "genome": item.genome.to_dict(), "outside_validation": outside_validation(item.genome),
                     "seed": item.genome in SEED_GENOMES, "measured_speedup": measured,
                     "predicted_speedup": item.constraints["predicted_speedup"], "ear": item.constraints["ear"],
                     "compute_fraction": record["compute_fraction"], "net_loss": net, "within_margin": net <= AUDIT_NET_LOSS_MARGIN,
                     **{key: result[key] for key in ("pass_at_1", "pass_to_fail", "fail_to_pass", "tasks_with_changed_code")}})
        save(output / "audit" / "audit.json", {"dense_pass_at_1": dense["code_execution"]["aggregate"]["base_pass_at_1"],
                                               "policies": rows, "criterion": success(rows)})
    for row in rows:
        print(f"{row['policy']}: predicted {row['predicted_speedup']:.2f}x, measured {row['measured_speedup']:.2f}x, "
              f"pass@1 {row['pass_at_1']:.3f}, net loss {row['net_loss']}, {'within' if row['within_margin'] else 'OUTSIDE'} margin", flush=True)
    print("Success criterion:", success(rows), flush=True)


def plus_tests(config_path: Path, output: Path) -> None:
    from language_models.code_execution import run_evalplus_apptainer

    settings = load_kv_policy_experiment_config(config_path).evaluation
    audited = json.loads((output / "audit" / "audit.json").read_text())["policies"]
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for name in ["dense", *(row["policy"] for row in audited)]:
        path = output / "audit" / "plus" / f"{name}.json"
        if not path.exists():
            # The exact programs the audit's base tests ran.
            samples = output / "audit" / "execution" / name
            programs = [{"task_id": sample["task_id"], "dataset": dataset, "executable_code": sample["solution"]}
                        for dataset in ("humaneval", "mbpp")
                        for sample in map(json.loads, (samples / f"samples_{dataset}.jsonl").read_text().splitlines())]
            save(path, run_evalplus_apptainer(
                programs, image=settings.evalplus_image, work_dir=output / "audit" / "plus_execution" / name,
                parallel=settings.evalplus_parallel, apptainer_bin=settings.evalplus_apptainer_bin, scope="full",
                timeout_seconds=settings.evalplus_timeout_seconds,
                dataset_paths={"humaneval": settings.evalplus_humaneval_path, "mbpp": settings.evalplus_mbpp_path}))
        result = json.loads(path.read_text())
        scored[name] = {task["task_id"]: task for dataset in result["datasets"].values() for task in dataset["tasks"]}
    ids = sorted(scored["dense"])

    def passes(name: str, kind: str) -> np.ndarray:
        if kind == "base":
            return np.array([scored[name][task]["base_status"] == "pass" for task in ids])
        return np.array([scored[name][task]["base_status"] == scored[name][task]["plus_status"] == "pass" for task in ids])

    rows = []
    for name in scored:
        row = {"policy": name, "pass_fraction": float(np.mean([pass_fraction(scored[name][task]) for task in ids]))}
        for kind in ("base", "plus"):
            reference, policy = passes("dense", kind), passes(name, kind)
            row.update({f"{kind}_pass_at_1": float(policy.mean()), f"{kind}_lost": int((reference & ~policy).sum()),
                        f"{kind}_gained": int((~reference & policy).sum())})
        rows.append(row)
    save(output / "audit" / "plus.json", {"tasks": len(ids), "policies": rows})
    for row in rows:
        print(f"{row['policy']}: base pass@1 {row['base_pass_at_1']:.3f} ({row['base_lost']} lost, {row['base_gained']} gained), "
              f"plus pass@1 {row['plus_pass_at_1']:.3f} ({row['plus_lost']} lost, {row['plus_gained']} gained), "
              f"pass fraction {row['pass_fraction']:.3f}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["advance", "dense", "evaluate", "check", "audit", "plus"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generation", type=int, default=0)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=16)
    parser.add_argument("--top", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "advance":
        advance(args.config, args.output)
    elif args.mode == "dense":
        dense_reference(args.config)
    elif args.mode == "evaluate":
        evaluate(args.config, args.output, args.generation, args.part, args.parts)
    elif args.mode == "check":
        check(args.config, args.output, args.generation)
    elif args.mode == "plus":
        plus_tests(args.config, args.output)
    else:
        audit(args.config, args.output, args.top)


if __name__ == "__main__":
    main()
