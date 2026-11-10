"""Confirm the search's fidelity thresholds and the cost model's ranking on unseen tasks.

The thresholds came from the 96 tasks of the EAR validity study. This applies the
search's exact feasibility rule, unchanged, to the tau confirmation set, and checks
that predicted batched time ranks policies the way real batched decoding does. The
rules are fixed in CONSTRAINED_SEARCH.md.

  run      GPU, A100 80GB. One shard of dense generation, replay with EAR, and
           free-running programs for every panel policy.
  time     GPU, A100 80GB. Real batched decoding of every policy at batch 8, on all
           tasks, in one job so that all timings share a GPU.
  report   No model. Scores the programs, then writes both verdicts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from dlm_kv_policy.dense_diffucoder_baseline import BatchedDenseDiffuCoderBaseline, DenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.forced_trajectory_replay import load_top_p_logits, replay_task
from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
from evolution.nsga2_kv_policy_search import KVRefreshPolicyGenome as Genome

from .batched_cost_model import predicted_seconds
from .batched_decoding_study import fixed_batches, source_provenance
from .ear_validity import (
    NET_LOSS_MARGIN,
    NOISE_FLOOR,
    PANEL,
    REGISTERED,
    TAIL_QUANTILE,
    MathKernelDense,
    answer_length,
    pass_fraction,
    task_metrics,
)
from .proxy_recheck import backend, generation_settings, policies, rank_score, save

GPU = "NVIDIA A100-SXM4-80GB"
SEED = 20261108
BOOTSTRAPS = 2000
BATCH = 8
# Reported, not in the verdict: one new gene each, which EAR was never validated on.
GENE_PANEL = {
    "gene_prompt_32": Genome(1, 8, 22, 0, prompt_refresh_interval=32),
    "gene_local_2": Genome(1, 8, 22, 0, local_refresh_radius=2),
    "gene_padding_16": Genome(1, 8, 22, 0, 4, padding_window=16),
    "gene_entropy_local": Genome(1, 8, 22, 0, entropy_threshold=0.5, entropy_scope="local"),
    "gene_entropy_full": Genome(1, 8, 22, 0, entropy_threshold=0.5, entropy_scope="full"),
}
DEPTHS = {**{row[0]: row[2] for row in PANEL}, **{name: genome.refresh_depth for name, genome in GENE_PANEL.items()}}
RANKING_MIN_SPEARMAN = 0.90


def panel(stops: set[int]) -> dict[str, Any]:
    return {**policies(stops, PANEL), **{name: genome.policy(stops) for name, genome in GENE_PANEL.items()}}


def stop_ids(tokenizer: Any) -> set[int]:
    return {token for token in (tokenizer.eos_token_id, tokenizer.pad_token_id) if token is not None}


def load(config_path: Path) -> tuple[Any, Any]:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator

    config = load_kv_policy_experiment_config(config_path)
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    if torch.cuda.get_device_name() != GPU:
        raise RuntimeError(f"Requires {GPU}: thresholds and costs were measured there")
    return config, evaluator


def run(config_path: Path, output: Path, part: int, parts: int) -> None:
    from dlm_kv_policy.code_correctness_and_latency_evaluator import seed_everything
    from language_models.code_tasks_and_response_processing import make_code_task_output
    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    config, evaluator = load(config_path)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    stops = stop_ids(tokenizer)
    save(output / f"part_{part}_provenance.json", {**source_provenance(), "gpu": GPU, "part": part, "parts": parts})
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)
    settings = generation_settings(config)

    def program(task: Any, result: Any, width: int) -> dict[str, Any]:
        ids = result.sequences[0, width:].cpu().tolist()
        return {"token_ids": ids, "output": make_code_task_output(task, tokenizer.decode(ids, skip_special_tokens=True))}

    for index in range(part, len(evaluator.tasks), parts):
        path = output / "tasks" / f"task_{index:03d}.json"
        if path.exists():
            continue
        task = evaluator.tasks[index]
        seed_everything(config.generation.seed + index)
        prompt, mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], config.model.device)
        width = prompt.shape[1]
        print(f"part {part}: task {index} dense", flush=True)
        dense = sampler.generate(prompt, attention_mask=mask, **settings)
        generations = {"dense": program(task, dense, width)}
        answer_end = answer_length(generations["dense"]["token_ids"], stops)
        backends = {name: backend(model, policy) for name, policy in panel(stops).items()}
        backends[NOISE_FLOOR] = MathKernelDense(model)
        rows: dict[str, list[dict[str, int]]] = {name: [] for name in backends if name != NOISE_FLOOR}
        for name in rows:
            forward = backends[name].forward_mask_logits

            def record(_forward=forward, _name=name, **inputs):
                result = _forward(**inputs)
                decision = result.decision
                rows[_name].append({"active": int(decision.active.sum()), "masked": result.mask_logits.shape[0],
                                    "full_depth": int((decision.active & decision.refresh_depth.le(1)).sum())})
                return result

            backends[name].forward_mask_logits = record
        print(f"part {part}: task {index} replay", flush=True)
        replay = replay_task(model, sampler, backends, prompt, mask, config.generation, top_p_logits, answer_end=answer_end)
        if replay["dense_token_ids"] != generations["dense"]["token_ids"]:
            raise RuntimeError(f"Task {index}: replay and generation disagree on dense")
        del backends
        with sdpa_kernel(SDPBackend.MATH):
            generations[NOISE_FLOOR] = program(task, sampler.generate(prompt, attention_mask=mask, **settings), width)
        for name, policy in panel(stops).items():
            print(f"part {part}: task {index} generation {name}", flush=True)
            result = DiffuCoderDenoisingSampler(backend(model, policy)).generate(prompt, attention_mask=mask, **settings)
            generations[name] = program(task, result, width)
        metrics = {}
        for name, steps in replay["policies"].items():
            outside = sum(position < answer_end for row in steps for position in row.get("outside_positions", []))
            metrics[name] = {**task_metrics(steps, answer_end), "answer_outside": outside, "flops": replay["flops"][name],
                             "rows": rows.get(name)}
        save(path, {"task_id": task.task_id, "index": index, "prompt_tokens": width, "answer_end": answer_end,
                    "dense_flops": replay["dense_flops"], "metrics": metrics, "generations": generations})
        print(f"part {part}: {index} {task.task_id} complete", flush=True)


def measured_batched_seconds(model: Any, tokenizer: Any, tasks: list[Any], make_backend: Any, settings: dict[str, Any],
                             device: str, batch: int = BATCH) -> float:
    """Real decoding time over all tasks in batches of similar length, after one warm-up batch."""

    from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    sampler = DiffuCoderDenoisingSampler(make_backend())
    batches, _ = fixed_batches(tasks, tokenizer, batch)
    inputs = [prepare_diffucoder_prompt_batch(tokenizer, [tasks[index].prompt for index in group], device) for group in batches]
    sampler.generate(inputs[0][0], attention_mask=inputs[0][1], **settings)
    return sum(sampler.generate(prompt, attention_mask=mask, **settings).elapsed_seconds for prompt, mask in inputs)


def time_policies(config_path: Path, output: Path) -> None:
    config, evaluator = load(config_path)
    model, tokenizer = evaluator.model, evaluator.tokenizer
    stops = stop_ids(tokenizer)
    settings = generation_settings(config)
    path = output / "measured_batch8.json"
    seconds: dict[str, float] = json.loads(path.read_text()) if path.exists() else {}
    makers = {"dense": lambda: BatchedDenseDiffuCoderBaseline(model)}
    for name in panel(stops):
        makers[name] = lambda _name=name: backend(model, panel(stops)[_name], batched=True)
    for name, make in makers.items():
        if name not in seconds:
            seconds[name] = measured_batched_seconds(model, tokenizer, evaluator.tasks, make, settings, config.model.device)
            save(path, seconds)
            print(f"{name}: {seconds[name]:.1f} s", flush=True)


# -- report ------------------------------------------------------------------


def feasibility(ear: np.ndarray, coverage: float, settings: Any) -> dict[str, Any]:
    """The search's rule, from evolution/constrained_kv_policy_search.py."""

    samples = np.random.default_rng(SEED).integers(0, len(ear), size=(BOOTSTRAPS, len(ear)))
    lower = float(np.nanquantile(np.nanmean(ear[samples], axis=1), 0.05))
    tail = float(np.nanquantile(ear, TAIL_QUANTILE))
    return {"ear": float(np.nanmean(ear)), "ear_lower_95": lower, "tail_ear": tail, "answer_coverage": coverage,
            "feasible": lower >= settings.min_ear and tail >= settings.min_tail_ear and coverage >= settings.min_answer_coverage}


def decide(names: list[str], feasible: dict[str, bool], within: dict[str, bool]) -> dict[str, Any]:
    """CONFIRMED needs no lossy policy feasible and at most one near-lossless policy rejected."""

    false_safe = [name for name in names if feasible[name] and not within[name]]
    false_reject = [name for name in names if not feasible[name] and within[name]]
    verdict = "UNSAFE" if false_safe else "TOO_STRICT" if len(false_reject) > 1 else "CONFIRMED"
    return {"verdict": verdict, "false_safe": false_safe, "false_reject": false_reject}


def ranking(predicted: dict[str, float], measured: dict[str, float]) -> dict[str, Any]:
    """Whether predicted speedup over dense orders policies as measured speedup does."""

    names = [name for name in predicted if name in measured]
    observed = {name: measured["dense"] / measured[name] for name in names}
    spearman = rank_score([predicted[name] for name in names], [observed[name] for name in names])
    return {"spearman": spearman, "passed": spearman is not None and spearman >= RANKING_MIN_SPEARMAN, "policies": len(names),
            "predicted_speedup": predicted, "measured_speedup": observed}


def report(config_path: Path, search_config: Path, output: Path) -> dict[str, Any]:
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config as load_config
    from language_models.code_execution import run_evalplus_apptainer

    settings = load_config(config_path).evaluation
    limits = load_config(search_config).evolution
    records = [json.loads(path.read_text()) for path in sorted((output / "tasks").glob("task_*.json"))]
    if len(records) != 96:
        raise SystemExit(f"{len(records)} tasks finished, expected 96")
    names = ["dense", NOISE_FLOOR, *(row[0] for row in PANEL), *GENE_PANEL]
    scored: dict[str, dict[str, dict[str, Any]]] = {}
    for name in names:
        path = output / "scores" / f"{name}_full.json"
        if not path.exists():
            save(path, run_evalplus_apptainer(
                [record["generations"][name]["output"] for record in records], image=settings.evalplus_image,
                work_dir=output / "execution" / name, parallel=settings.evalplus_parallel,
                apptainer_bin=settings.evalplus_apptainer_bin, scope="full", timeout_seconds=settings.evalplus_timeout_seconds,
                dataset_paths={"humaneval": settings.evalplus_humaneval_path, "mbpp": settings.evalplus_mbpp_path}))
        result = json.loads(path.read_text())
        scored[name] = {task["task_id"]: task for dataset in result["datasets"].values() for task in dataset["tasks"]}
    ids = [record["task_id"] for record in records]
    reference = np.array([scored["dense"][task]["base_status"] == "pass" for task in ids])
    answer_commits = sum(record["answer_end"] for record in records)
    dense_flops = sum(record["dense_flops"] for record in records)
    table: dict[str, dict[str, Any]] = {}
    for name in names[1:]:
        passes = np.array([scored[name][task]["base_status"] == "pass" for task in ids])
        lost, gained = int((reference & ~passes).sum()), int((~reference & passes).sum())
        ear = np.array([record["metrics"][name]["ear"] for record in records])
        coverage = 1 - sum(record["metrics"][name]["answer_outside"] for record in records) / answer_commits
        table[name] = {
            **feasibility(ear, coverage, limits), "lost": lost, "gained": gained, "net_loss": lost - gained,
            "within_margin": lost - gained <= NET_LOSS_MARGIN,
            "pass_at_1": float(passes.mean()),
            "pass_fraction_drop": float(np.mean([pass_fraction(scored["dense"][task]) - pass_fraction(scored[name][task]) for task in ids])),
            "position_flip_rate": float(np.nanmean([record["metrics"][name]["position_flip_rate"] for record in records])),
            "compute_fraction": sum(record["metrics"][name]["flops"] for record in records) / dense_flops,
        }
    feasible = {name: row["feasible"] for name, row in table.items()}
    within = {name: row["within_margin"] for name, row in table.items()}
    tracked = [name for name in table if name != NOISE_FLOOR]
    summary: dict[str, Any] = {
        "dense_pass_at_1": float(reference.mean()), "table": table,
        "registered": decide(list(REGISTERED), feasible, within),
        "all_old_policies": decide([name for name in table if name not in GENE_PANEL], feasible, within),
        "new_genes": decide(list(GENE_PANEL), feasible, within),
        "spearman_one_minus_ear": {
            "net_loss": rank_score([1 - table[name]["ear"] for name in tracked], [table[name]["net_loss"] for name in tracked]),
            "pass_fraction_drop": rank_score([1 - table[name]["ear"] for name in tracked], [table[name]["pass_fraction_drop"] for name in tracked]),
        },
    }
    measured_path = output / "measured_batch8.json"
    if measured_path.exists():
        cost = json.loads(Path(limits.cost_table).read_text())
        predicted = {}
        for name in tracked:
            tasks = [{"task_id": record["task_id"], "prompt_tokens": record["prompt_tokens"], "rows": record["metrics"][name]["rows"]}
                     for record in records]
            seconds, dense_seconds = predicted_seconds(cost, tasks, BATCH, DEPTHS[name])
            predicted[name] = dense_seconds / seconds
        summary["cost_ranking"] = ranking(predicted, json.loads(measured_path.read_text()))
    summary["launch"] = summary["registered"]["verdict"] == "CONFIRMED" and summary.get("cost_ranking", {}).get("passed", False)
    save(output / "tau_confirmation_report.json", summary)

    speed = summary.get("cost_ranking", {})
    print(f"dense pass@1 {summary['dense_pass_at_1']:.3f}")
    print(f"{'policy':<22}{'EAR':>8}{'lower':>8}{'tail':>8}{'cover':>8}{'net':>4}{'pass@1':>7}{'drop':>8}{'pred':>7}{'meas':>7}  feasible / margin")
    for name, row in sorted(table.items(), key=lambda item: -item[1]["ear"]):
        print(f"{name:<22}{row['ear']:>8.4f}{row['ear_lower_95']:>8.4f}{row['tail_ear']:>8.4f}{row['answer_coverage']:>8.4f}"
              f"{row['net_loss']:>4}{row['pass_at_1']:>7.3f}{row['pass_fraction_drop']:>8.4f}"
              f"{speed.get('predicted_speedup', {}).get(name, float('nan')):>7.2f}{speed.get('measured_speedup', {}).get(name, float('nan')):>7.2f}"
              f"  {'feasible' if row['feasible'] else 'infeasible'} / {'within' if row['within_margin'] else 'OUTSIDE'}")
    for label in ("registered", "all_old_policies", "new_genes"):
        print(label, summary[label])
    print("Spearman 1-EAR:", summary["spearman_one_minus_ear"])
    print("cost ranking Spearman:", speed.get("spearman"), "passed:", speed.get("passed"))
    print("LAUNCH:", summary["launch"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["run", "time", "report"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--search-config", type=Path, help="report: the thresholds to confirm")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--part", type=int, default=0)
    parser.add_argument("--parts", type=int, default=16)
    args = parser.parse_args()
    if args.mode == "run":
        run(args.config, args.output, args.part, args.parts)
    elif args.mode == "time":
        time_policies(args.config, args.output)
    else:
        report(args.config, args.search_config, args.output)


if __name__ == "__main__":
    main()
