"""Search attention/MLP residual-output-cache schedules with one resident DLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import dill
import numpy as np
import yaml
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.population import Population
from pymoo.core.termination import NoTermination
from pymoo.operators.crossover.pntx import PointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation

from ecad.genetic.dlm_problem import (
    DLMCachingScheduleProblem,
    DLMGroupedRandomSampling,
)
from ecad.language_models.residual_cache_model_backend import load_dlm_backend
from ecad.language_models.code_tasks_and_response_processing import CODE_POSTPROCESSOR_VERSION
from ecad.language_models.residual_cache_experiment_config import DLMResidualCacheExperimentConfig, load_dlm_residual_cache_config
from ecad.language_models.residual_cache_schedule_evaluator import DLMResidualCacheScheduleEvaluator, seed_everything
from ecad.language_models.generation_task_loader import load_generation_tasks
from ecad.language_models.task_dataset_path_compatibility import legacy_task_source_path
from ecad.language_models.experiment_artifact_paths import legacy_checkpoint_result_path
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


CHECKPOINT_NAME = "algorithm_checkpoint.pkl"
STATE_NAME = "run_state.json"
GENERATION_ARTIFACT_VERSION = "dlm-candidate-generation-v1"


def _objective_version(config: DLMResidualCacheExperimentConfig) -> str:
    if config.evaluation.quality_objective == "base_regression_rate":
        return "evalplus-base-regression-v1"
    return "token-agreement-v1"


def _objective_names(config: DLMResidualCacheExperimentConfig) -> list[str]:
    quality_name = (
        "base_regression_rate"
        if config.evaluation.quality_objective == "base_regression_rate"
        else "token_agreement_loss"
    )
    return [quality_name, "compute_fraction"]


def _json_write(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)


def _config_signature(config: DLMResidualCacheExperimentConfig) -> str:
    config_data = config.to_dict()
    # Artifact relocation changes storage only, not the experiment definition.
    config_data["run"]["output_dir"] = legacy_checkpoint_result_path(
        config_data["run"]["output_dir"]
    )
    baseline_artifact = config_data["evaluation"].get("baseline_artifact")
    if baseline_artifact is not None:
        config_data["evaluation"]["baseline_artifact"] = legacy_checkpoint_result_path(
            baseline_artifact
        )
    # Filename-only changes preserve the signatures of historical checkpoints.
    for field in ("code_tasks_file", "prompts_file"):
        value = config_data["evaluation"].get(field)
        if value is not None:
            config_data["evaluation"][field] = legacy_task_source_path(value)
    # Resume is an invocation policy, not part of the experiment definition.
    config_data["run"].pop("resume", None)
    # Preserve checkpoints created before correctness-aware execution fields
    # were added. These defaults do not affect legacy token-agreement runs.
    if config.evaluation.quality_objective == "token_agreement":
        for key in (
            "quality_objective",
            "evalplus_image",
            "evalplus_humaneval_path",
            "evalplus_mbpp_path",
            "evalplus_parallel",
            "evalplus_apptainer_bin",
            "evalplus_timeout_seconds",
            "baseline_artifact",
            "max_syntax_invalid",
        ):
            config_data["evaluation"].pop(key, None)
    else:
        config_data["code_postprocessor_version"] = CODE_POSTPROCESSOR_VERSION
    # Preserve signatures made before the backend selector was introduced.
    # These fields do not affect LLaDA generation.
    if config.model.family == "llada":
        config_data["model"].pop("family", None)
        for key in ("top_p", "top_k", "algorithm", "algorithm_temperature"):
            config_data["generation"].pop(key, None)
    payload = json.dumps(config_data, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _save_checkpoint(
    run_dir: Path,
    algorithm: NSGA2,
    completed_generations: int,
    config_signature: str,
    pending_generation: int | None = None,
    pending_genomes: np.ndarray | None = None,
) -> None:
    pending_genomes_list = (
        pending_genomes.astype(bool).tolist()
        if pending_genomes is not None
        else None
    )
    checkpoint_path = run_dir / CHECKPOINT_NAME
    temporary = checkpoint_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        dill.dump(
            {
                "algorithm": algorithm,
                "completed_generations": completed_generations,
                "config_signature": config_signature,
                "pending_generation": pending_generation,
                "pending_genomes": pending_genomes_list,
            },
            handle,
        )
    temporary.replace(checkpoint_path)
    state = {
        "completed_generations": completed_generations,
        "config_signature": config_signature,
        "checkpoint": CHECKPOINT_NAME,
        "phase": "pending_evaluation" if pending_genomes is not None else "ready",
        "pending_generation": pending_generation,
        "pending_genomes": pending_genomes_list,
    }
    _json_write(run_dir / STATE_NAME, state)


def _load_checkpoint(
    run_dir: Path, config_signature: str
) -> tuple[NSGA2, int, tuple[int, np.ndarray] | None] | None:
    checkpoint_path = run_dir / CHECKPOINT_NAME
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        checkpoint = dill.load(handle)
    if checkpoint.get("config_signature") != config_signature:
        raise ValueError(
            "Refusing to resume because the resolved configuration changed. "
            "Use a new run.name or remove the old run directory intentionally."
        )
    algorithm = checkpoint["algorithm"]
    pending = None
    if checkpoint.get("pending_genomes") is not None:
        if checkpoint.get("pending_generation") is None:
            raise ValueError("Incomplete pending-evaluation checkpoint state")
        pending = (
            int(checkpoint["pending_generation"]),
            np.asarray(checkpoint["pending_genomes"], dtype=np.bool_),
        )
    return algorithm, int(checkpoint["completed_generations"]), pending


def _build_algorithm(
    problem: DLMCachingScheduleProblem, config: DLMResidualCacheExperimentConfig
) -> NSGA2:
    evolution = config.evolution
    mutation: BitflipMutation
    if evolution.mutation_expected_flips is not None:
        mutation_probability = min(
            1.0, evolution.mutation_expected_flips / problem.n_var
        )
        mutation = BitflipMutation(prob=1.0, prob_var=mutation_probability)
    else:
        # Preserve legacy configs, where this controls whether an offspring
        # receives pymoo's default one-bit mutation.
        mutation = BitflipMutation(prob=evolution.mutation_probability)
    algorithm = NSGA2(
        pop_size=evolution.population_size,
        sampling=DLMGroupedRandomSampling(
            recompute_probability=evolution.initial_recompute_probability,
            min_cached_groups=evolution.min_cached_groups,
            cached_groups=evolution.initial_cached_groups,
        ),
        crossover=PointCrossover(
            prob=evolution.crossover_probability,
            n_points=min(evolution.crossover_points, problem.n_var - 1),
        ),
        mutation=mutation,
        eliminate_duplicates=True,
    )
    algorithm.setup(
        problem,
        termination=NoTermination(),
        seed=evolution.seed,
        verbose=True,
    )
    return algorithm


def _schedule_for_candidate(
    genome: np.ndarray,
    index: int,
    generation: int,
    num_layers: int,
    config: DLMResidualCacheExperimentConfig,
) -> DLMCacheSchedule:
    return DLMCacheSchedule.from_genome(
        genome,
        num_layers=num_layers,
        num_inference_steps=config.num_schedule_steps,
        step_group_size=config.schedule.step_group_size,
        layer_group_size=config.schedule.layer_group_size,
        name=(
            f"{config.run.name}_gen_{generation:03d}_cand_{index:03d}"
        ),
        top_level_config={
            "model_family": config.model.family,
            "model_path": config.model.path,
            "generation": config.to_dict()["generation"],
            "grouping": config.to_dict()["schedule"],
        },
        attributes={
            "generation": generation,
            "candidate_index": index,
        },
        schedule_domain=config.schedule_domain,
    )


def _evaluate_generation(
    *,
    generation: int,
    genomes: np.ndarray,
    evaluator: DLMResidualCacheScheduleEvaluator,
    num_layers: int,
    config: DLMResidualCacheExperimentConfig,
) -> tuple[np.ndarray, np.ndarray]:
    generation_dir = config.run_dir / "generations" / f"gen_{generation:03d}"
    candidates_dir = generation_dir / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    objectives = np.empty((len(genomes), 2), dtype=np.float64)
    syntax_invalid_counts = np.zeros(len(genomes), dtype=np.int64)
    results_by_genome: dict[str, dict[str, Any]] = {}

    for index, genome in enumerate(genomes):
        schedule = _schedule_for_candidate(
            genome, index, generation, num_layers, config
        )
        schedule_path = candidates_dir / f"cand_{index:03d}.json"
        generation_path = candidates_dir / f"cand_{index:03d}.generation.json"
        result_path = candidates_dir / f"cand_{index:03d}.metrics.json"
        genome_key = hashlib.sha256(
            np.packbits(genome.astype(np.bool_)).tobytes()
        ).hexdigest()

        if result_path.exists() and config.run.resume:
            prior = json.loads(result_path.read_text(encoding="utf-8"))
            if prior.get("genome_sha256") != genome_key:
                raise ValueError(f"Stale candidate result at {result_path}")
            prior_baseline_signature = prior.get("baseline_signature")
            if config.evaluation.quality_objective == "base_regression_rate":
                # Correctness results are meaningful only against the exact
                # all-recompute baseline that was executed in the sandbox.
                if prior_baseline_signature != evaluator.signature:
                    raise ValueError(
                        f"Candidate baseline signature mismatch at {result_path}"
                    )
                if (
                    prior.get("postprocessor_version")
                    != CODE_POSTPROCESSOR_VERSION
                ):
                    raise ValueError(
                        f"Candidate postprocessor version mismatch at {result_path}"
                    )
            elif prior_baseline_signature not in {None, evaluator.signature}:
                raise ValueError(
                    f"Candidate baseline signature mismatch at {result_path}"
                )
            metrics = prior["metrics"]
            prior_version = prior.get("objective_version") or metrics.get(
                "objective_version"
            )
            # Results written before objective versioning are valid only for
            # the legacy token-agreement objective.
            if prior_version is None and config.evaluation.quality_objective == "token_agreement":
                prior_version = "token-agreement-v1"
                metrics.setdefault(
                    "token_agreement_loss", 1.0 - metrics["token_agreement"]
                )
                metrics.setdefault("objective_version", prior_version)
            if prior_version != _objective_version(config):
                raise ValueError(
                    f"Candidate objective version mismatch at {result_path}: "
                    f"found {prior_version!r}, expected {_objective_version(config)!r}"
                )
            print(f"Reusing completed candidate {index:03d} from disk")
        elif genome_key in results_by_genome:
            metrics = results_by_genome[genome_key]
            print(f"Candidate {index:03d} duplicates an evaluated genome")
        else:
            generated_metrics = None
            if generation_path.is_file() and config.run.resume:
                staged = json.loads(generation_path.read_text(encoding="utf-8"))
                if staged.get("genome_sha256") != genome_key:
                    raise ValueError(
                        f"Stale candidate generation at {generation_path}"
                    )
                if staged.get("baseline_signature") != evaluator.signature:
                    raise ValueError(
                        f"Candidate generation baseline mismatch at {generation_path}"
                    )
                if staged.get("postprocessor_version") != CODE_POSTPROCESSOR_VERSION:
                    raise ValueError(
                        f"Candidate generation postprocessor mismatch at "
                        f"{generation_path}"
                    )
                if staged.get("format_version") != GENERATION_ARTIFACT_VERSION:
                    raise ValueError(
                        f"Unsupported candidate generation artifact: {generation_path}"
                    )
                generated_metrics = staged["metrics"]
                print(
                    f"Reusing generated responses for generation {generation:03d}, "
                    f"candidate {index:03d}"
                )
            if generated_metrics is None:
                print(
                    f"Generating generation {generation:03d}, "
                    f"candidate {index:03d}/{len(genomes) - 1:03d}"
                )
                generated_metrics = evaluator.generate_candidate(schedule)
                # Persist expensive GPU output before starting untrusted code.
                _json_write(
                    generation_path,
                    {
                        "format_version": GENERATION_ARTIFACT_VERSION,
                        "generation": generation,
                        "candidate_index": index,
                        "genome_sha256": genome_key,
                        "baseline_signature": evaluator.signature,
                        "postprocessor_version": CODE_POSTPROCESSOR_VERSION,
                        "metrics": generated_metrics,
                    },
                )
            if config.evaluation.quality_objective == "base_regression_rate":
                print(
                    f"Running sandboxed base tests for generation "
                    f"{generation:03d}, candidate {index:03d}"
                )
            metrics = evaluator.score_candidate(
                generated_metrics,
                execution_work_dir=(
                    config.run_dir
                    / "code_execution"
                    / f"gen_{generation:03d}_cand_{index:03d}_base"
                ),
            )
        results_by_genome[genome_key] = metrics

        schedule.metrics = metrics
        schedule.to_json(schedule_path)
        _json_write(
            result_path,
            {
                "objective_version": _objective_version(config),
                "generation": generation,
                "candidate_index": index,
                "genome_sha256": genome_key,
                "baseline_signature": evaluator.signature,
                "postprocessor_version": CODE_POSTPROCESSOR_VERSION,
                "objectives": [
                    float(metrics["quality_loss"]),
                    float(metrics["compute_fraction"]),
                ],
                "metrics": metrics,
            },
        )
        objectives[index] = (
            metrics["quality_loss"],
            metrics["compute_fraction"],
        )
        if "correctness" in metrics:
            syntax_invalid_counts[index] = int(
                metrics["correctness"]["syntax_invalid_count"]
            )
        print(
            f"  quality loss={metrics['quality_loss']:.4f}, "
            f"token agreement={metrics['token_agreement']:.4f}, "
            f"compute fraction={metrics['compute_fraction']:.4f}, "
            f"measured speedup={metrics['speedup_vs_baseline']:.3f}x"
        )
        if "correctness" in metrics:
            correctness = metrics["correctness"]
            print(
                f"  base pass@1={correctness['candidate_base_pass_at_1']:.4f}, "
                f"regressions={correctness['base_regression_count']}, "
                f"recoveries={correctness['base_recovery_count']}, "
                f"syntax invalid={correctness['syntax_invalid_count']}"
            )

    _json_write(
        generation_dir / "summary.json",
        {
            "generation": generation,
            "num_candidates": len(genomes),
            "objective_names": _objective_names(config),
            "objective_version": _objective_version(config),
            "objectives": objectives.tolist(),
            "syntax_invalid_counts": syntax_invalid_counts.tolist(),
        },
    )
    return objectives, syntax_invalid_counts


def _save_pareto(
    config: DLMResidualCacheExperimentConfig,
    algorithm: NSGA2,
    generation: int,
    num_layers: int,
) -> None:
    optimum = algorithm.opt
    if optimum is None or len(optimum) == 0:
        return
    genomes = optimum.get("X")
    objectives = optimum.get("F")
    constraint_values = optimum.get("G")
    records = []
    pareto_dir = config.run_dir / "pareto_schedules"
    pareto_dir.mkdir(parents=True, exist_ok=True)
    for row_index, (genome, objective) in enumerate(zip(genomes, objectives)):
        genome_hash = hashlib.sha256(
            np.packbits(genome.astype(np.bool_)).tobytes()
        ).hexdigest()[:12]
        schedule_path = pareto_dir / f"pareto_{genome_hash}.json"
        schedule = DLMCacheSchedule.from_genome(
            genome,
            num_layers=num_layers,
            num_inference_steps=config.num_schedule_steps,
            step_group_size=config.schedule.step_group_size,
            layer_group_size=config.schedule.layer_group_size,
            name=f"pareto_{genome_hash}",
            top_level_config={
                "model_family": config.model.family,
                "model_path": config.model.path,
                "generation": config.to_dict()["generation"],
                "grouping": config.to_dict()["schedule"],
            },
            attributes={"pareto_after_generation": generation},
            metrics={
                "quality_loss": float(objective[0]),
                "quality_objective": config.evaluation.quality_objective,
                "compute_fraction": float(objective[1]),
            },
            schedule_domain=config.schedule_domain,
        )
        schedule.to_json(schedule_path)
        records.append(
            {
                "genome": genome.astype(bool).tolist(),
                "quality_loss": float(objective[0]),
                "quality_objective": config.evaluation.quality_objective,
                "compute_fraction": float(objective[1]),
                "constraint_values": (
                    np.asarray(constraint_values[row_index], dtype=float).tolist()
                    if constraint_values is not None
                    else []
                ),
                "feasible": (
                    bool(np.all(np.asarray(constraint_values[row_index]) <= 0))
                    if constraint_values is not None
                    else True
                ),
                "schedule": str(schedule_path),
            }
        )
    _json_write(
        config.run_dir / "pareto_front.json",
        {
            "after_generation": generation,
            "objective_names": _objective_names(config),
            "objective_version": _objective_version(config),
            "constraint_names": (
                ["minimum_cached_groups", "syntax_invalid_excess"]
                if config.evaluation.quality_objective == "base_regression_rate"
                else ["minimum_cached_groups"]
            ),
            "solutions": records,
        },
    )


def run(config: DLMResidualCacheExperimentConfig, max_generations: int | None = None) -> None:
    config.run_dir.mkdir(parents=True, exist_ok=True)
    resolved_config_path = config.run_dir / "resolved_config.yaml"
    resolved_config_path.write_text(
        yaml.safe_dump(config.to_dict(), sort_keys=False), encoding="utf-8"
    )
    generation_tasks = load_generation_tasks(config.evaluation)
    _json_write(
        config.run_dir / "code_tasks.json",
        {"tasks": [task.to_dict() for task in generation_tasks]},
    )
    seed_everything(config.evolution.seed)

    print(
        f"Loading one persistent {config.model.family} replica from "
        f"{config.model.path}"
    )
    backend = load_dlm_backend(config)
    num_layers = backend.num_layers
    initial_schedule = DLMCacheSchedule.all_recompute(
        num_layers=num_layers,
        num_inference_steps=config.num_schedule_steps,
        step_group_size=config.schedule.step_group_size,
        layer_group_size=config.schedule.layer_group_size,
        schedule_domain=config.schedule_domain,
    )
    backend.attach_cache(initial_schedule)
    evaluator = DLMResidualCacheScheduleEvaluator(backend, generation_tasks, config)
    baseline = evaluator.load_or_create_baseline(config.run_dir / "baseline.json")
    print(
        f"Baseline ready: {len(baseline.generated_token_ids)} prompts in "
        f"{baseline.elapsed_seconds:.3f}s"
    )
    evaluator.prepare_quality_objective(config.run_dir / "baseline.base_eval.json")

    problem = DLMCachingScheduleProblem(
        num_layers=num_layers,
        num_inference_steps=config.num_schedule_steps,
        step_group_size=config.schedule.step_group_size,
        layer_group_size=config.schedule.layer_group_size,
        min_cached_groups=config.evolution.min_cached_groups,
        enforce_syntax_validity=(
            config.evaluation.quality_objective == "base_regression_rate"
        ),
        max_syntax_invalid=config.evaluation.max_syntax_invalid,
    )
    signature = _config_signature(config)
    loaded = _load_checkpoint(config.run_dir, signature) if config.run.resume else None
    if loaded is None:
        algorithm = _build_algorithm(problem, config)
        completed_generations = 0
        pending = None
    else:
        algorithm, completed_generations, pending = loaded
        print(f"Resumed after {completed_generations} completed generations")

    target_generations = config.evolution.generations
    if max_generations is not None:
        target_generations = min(
            target_generations, completed_generations + max_generations
        )

    while completed_generations < target_generations:
        generation = completed_generations
        if pending is not None:
            pending_generation, genomes = pending
            if pending_generation != generation:
                raise ValueError(
                    f"Pending generation {pending_generation} does not match "
                    f"completed count {generation}"
                )
            print(f"Resuming pending generation {generation:03d}")
        else:
            proposed = algorithm.ask()
            genomes = proposed.get("X").astype(np.bool_)
            # ECAD's operators may use process-global RNG state. Checkpoint the
            # post-ask algorithm and exact genomes before expensive evaluation.
            _save_checkpoint(
                config.run_dir,
                algorithm,
                completed_generations,
                signature,
                pending_generation=generation,
                pending_genomes=genomes,
            )
        objectives, syntax_invalid_counts = _evaluate_generation(
            generation=generation,
            genomes=genomes,
            evaluator=evaluator,
            num_layers=num_layers,
            config=config,
        )
        constraints = problem.constraint_violations(
            genomes, syntax_invalid_counts=syntax_invalid_counts
        )
        evaluated = Population.new("X", genomes, F=objectives, G=constraints)
        algorithm.tell(evaluated)
        completed_generations += 1
        pending = None
        _save_pareto(config, algorithm, generation, num_layers)
        _save_checkpoint(config.run_dir, algorithm, completed_generations, signature)
        print(f"Completed and checkpointed generation {generation:03d}")

    backend.uninstall()
    print(f"ECAD run complete at {config.run_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Override run.output_dir; created automatically")
    parser.add_argument(
        "--model-path",
        type=str,
        help="Override model.path from YAML without editing the config",
    )
    parser.add_argument(
        "--max-generations",
        type=int,
        help="Run at most this many additional generations in this invocation",
    )
    parser.add_argument(
        "--evalplus-image",
        type=str,
        help="Override evaluation.evalplus_image from YAML",
    )
    parser.add_argument(
        "--apptainer-bin",
        type=str,
        help="Override evaluation.evalplus_apptainer_bin from YAML",
    )
    parser.add_argument(
        "--no-resume", action="store_true", help="Ignore an existing checkpoint"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dlm_residual_cache_config(args.config)
    if args.output_dir is not None:
        config.run.output_dir = str(args.output_dir.expanduser())
    if args.model_path:
        config.model.path = args.model_path
    if args.evalplus_image:
        config.evaluation.evalplus_image = args.evalplus_image
    if args.apptainer_bin:
        config.evaluation.evalplus_apptainer_bin = args.apptainer_bin
    if args.no_resume:
        config.run.resume = False
    config.validate()
    run(config, args.max_generations)


if __name__ == "__main__":
    main()
