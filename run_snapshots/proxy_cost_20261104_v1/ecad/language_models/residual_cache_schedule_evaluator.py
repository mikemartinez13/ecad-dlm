"""Persistent-model evaluator for evolutionary DLM cache schedules."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from typing import Any, Iterable

import numpy as np
import torch

from ecad.language_models.residual_cache_experiment_config import DLMResidualCacheExperimentConfig
from ecad.language_models.code_tasks_and_response_processing import (
    CODE_POSTPROCESSOR_VERSION,
    CodeGenerationTask,
    make_code_task_output,
)
from ecad.language_models.code_execution import (
    compare_base_correctness,
    run_evalplus_apptainer,
)
from ecad.language_models.residual_cache_model_backend import DLMBackend
from ecad.language_models.experiment_artifact_paths import resolve_experiment_result_path
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


@dataclass
class BaselineRecord:
    signature: str
    generated_token_ids: list[list[int]]
    responses: list[str]
    decoded: list[str]
    task_outputs: list[dict[str, Any]]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature": self.signature,
            "generated_token_ids": self.generated_token_ids,
            "responses": self.responses,
            "decoded": self.decoded,
            "task_outputs": self.task_outputs,
            "elapsed_seconds": self.elapsed_seconds,
        }


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def infer_component_weights(
    model: torch.nn.Module, sequence_length: int
) -> tuple[float, float, float]:
    """Estimate dense MACs of one attention and one gated-MLP branch."""

    config = model.config
    d_model = int(
        getattr(config, "d_model", None)
        or getattr(config, "hidden_size", None)
    )
    num_heads = int(
        getattr(config, "n_heads", None)
        or getattr(config, "num_attention_heads", None)
    )
    num_kv_heads = int(
        getattr(config, "n_kv_heads", None)
        or getattr(config, "num_key_value_heads", None)
        or num_heads
    )
    kv_width = d_model * num_kv_heads / num_heads
    mlp_width = int(
        getattr(config, "mlp_hidden_size", None)
        or getattr(config, "intermediate_size", None)
        or getattr(config, "mlp_ratio", 4) * d_model
    )
    attention_macs = sequence_length * (
        d_model * d_model + 2 * d_model * kv_width + d_model * d_model
    ) + 2 * sequence_length * sequence_length * d_model
    mlp_macs = 3 * sequence_length * d_model * mlp_width
    vocabulary_size = int(
        getattr(config, "embedding_size", None) or config.vocab_size
    )
    # Token embedding lookup is negligible, but the full-vocabulary output
    # projection is paid at every denoising step and cannot be branch-cached.
    fixed_step_macs = sequence_length * d_model * vocabulary_size
    return float(attention_macs), float(mlp_macs), float(fixed_step_macs)


class DLMResidualCacheScheduleEvaluator:
    """Evaluate many schedules without ever reloading the model."""

    def __init__(
        self,
        backend: DLMBackend,
        tasks: list[CodeGenerationTask],
        config: DLMResidualCacheExperimentConfig,
    ) -> None:
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.controller = backend.controller
        if self.controller is None:
            raise RuntimeError("The DLM backend cache must be attached first")
        self.tasks = tasks
        self.prompts = [task.prompt for task in tasks]
        self.config = config
        self.device = config.model.device
        self.signature = self._baseline_signature()
        self._baseline: BaselineRecord | None = None
        self._baseline_execution: dict[str, Any] | None = None
        self._evalplus_image_identity: dict[str, Any] | None = None
        self._evalplus_dataset_identities: dict[str, dict[str, Any]] | None = None

        lengths = []
        for batch in iter_batches(self.prompts, config.evaluation.batch_size):
            input_ids, _ = backend.prepare_prompt_batch(batch)
            lengths.append(input_ids.shape[1] + config.generation.gen_length)
        representative_length = int(round(sum(lengths) / len(lengths)))
        inferred_attention, inferred_mlp, inferred_fixed = infer_component_weights(
            self.model, representative_length
        )
        self.attention_weight = (
            config.schedule.attention_weight
            if config.schedule.attention_weight is not None
            else inferred_attention
        )
        self.mlp_weight = (
            config.schedule.mlp_weight
            if config.schedule.mlp_weight is not None
            else inferred_mlp
        )
        self.fixed_step_weight = (
            config.schedule.fixed_step_weight
            if config.schedule.fixed_step_weight is not None
            else inferred_fixed
        )

    def _baseline_signature(self) -> str:
        device_name = self.config.model.device
        if self.config.model.device.startswith("cuda") and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(
                torch.device(self.config.model.device)
            )
        generation = self.config.to_dict()["generation"]
        relevant = {
            "model_path": self.config.model.path,
            "dtype": self.config.model.dtype,
            "prompts": self.prompts,
            "generation": generation,
            "batch_size": self.config.evaluation.batch_size,
            "seed": self.config.evolution.seed,
            "device_name": device_name,
        }
        if self.config.model.family == "llada":
            for key in ("top_p", "top_k", "algorithm", "algorithm_temperature"):
                generation.pop(key, None)
        else:
            relevant["model_family"] = self.config.model.family
        payload = json.dumps(relevant, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _write_json_atomic(path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
        temporary.replace(path)

    @staticmethod
    def _file_sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _current_evalplus_dataset_identities(self) -> dict[str, dict[str, Any]]:
        configured = {
            "humaneval": self.config.evaluation.evalplus_humaneval_path,
            "mbpp": self.config.evaluation.evalplus_mbpp_path,
        }
        identities: dict[str, dict[str, Any]] = {}
        for dataset, value in configured.items():
            if not value:
                raise ValueError(f"Missing configured EvalPlus data for {dataset}")
            original = Path(value)
            if original.is_symlink():
                raise RuntimeError(
                    f"Refusing symlinked EvalPlus {dataset} data file: {original}"
                )
            path = original.resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"EvalPlus {dataset} data file not found: {path}"
                )
            stat = path.stat()
            identities[dataset] = {
                "path": str(path),
                "device": stat.st_dev,
                "inode": stat.st_ino,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": self._file_sha256(path),
            }
        return identities

    def _task_outputs_sha256(self, outputs: list[dict[str, Any]]) -> str:
        payload = [
            {
                "task_id": output["task_id"],
                "dataset": output["dataset"],
                "executable_code": output["executable_code"],
            }
            for output in outputs
        ]
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode("utf-8")
        ).hexdigest()

    def _record_from_data(
        self, data: dict[str, Any], source: Path
    ) -> BaselineRecord | None:
        if data.get("signature") != self.signature:
            print(f"Ignoring baseline with a different signature: {source}")
            return None
        responses = data.get("responses", data.get("decoded", []))
        if self.config.evaluation.source == "evalplus" and not responses:
            print(f"Ignoring code baseline without raw responses: {source}")
            return None
        task_outputs = data.get("task_outputs", [])
        if self.config.evaluation.source == "evalplus":
            if len(responses) != len(self.tasks):
                print(f"Ignoring baseline with the wrong response count: {source}")
                return None
            # Derived code is always rebuilt with the current postprocessor.
            task_outputs = [
                make_code_task_output(task, response)
                for task, response in zip(self.tasks, responses)
            ]
        return BaselineRecord(
            signature=data["signature"],
            generated_token_ids=data["generated_token_ids"],
            responses=responses,
            decoded=data.get("decoded", []),
            task_outputs=task_outputs,
            elapsed_seconds=float(data["elapsed_seconds"]),
        )

    def _all_recompute_schedule(self) -> DLMCacheSchedule:
        return DLMCacheSchedule.all_recompute(
            num_layers=self.backend.num_layers,
            num_inference_steps=self.backend.num_schedule_steps,
            step_group_size=self.config.schedule.step_group_size,
            layer_group_size=self.config.schedule.layer_group_size,
            name="all_recompute_baseline",
            top_level_config=self.config.to_dict(),
            schedule_domain=self.backend.schedule_domain,
        )

    def _run_schedule(self, schedule: DLMCacheSchedule) -> dict[str, Any]:
        self.backend.set_schedule(schedule)
        generated_token_ids: list[list[int]] = []
        responses: list[str] = []
        decoded: list[str] = []
        total_elapsed = 0.0
        aggregate_stats: dict[str, Any] = {
            "recomputed": {"attention": 0, "mlp": 0},
            "reused": {"attention": 0, "mlp": 0},
            "forced_recomputes": {"attention": 0, "mlp": 0},
            "blocks_started": 0,
            "steps_started": 0,
        }
        for batch_index, task_batch in enumerate(
            iter_batches(self.tasks, self.config.evaluation.batch_size)
        ):
            prompt_batch = [task.prompt for task in task_batch]
            seed_everything(self.config.evolution.seed + batch_index)
            input_ids, attention_mask = self.backend.prepare_prompt_batch(prompt_batch)
            result = self.backend.generate(input_ids, attention_mask)
            generated = result.token_ids[:, input_ids.shape[1] :]
            generated_token_ids.extend(generated.cpu().tolist())
            response_batch = self.backend.decode_generated(generated)
            responses.extend(response_batch)
            if self.config.evaluation.save_decoded_text:
                decoded.extend(response_batch)
            total_elapsed += result.elapsed_seconds
            stats = result.cache_stats or {}
            for component in ("attention", "mlp"):
                for counter in ("recomputed", "reused", "forced_recomputes"):
                    aggregate_stats[counter][component] += stats.get(counter, {}).get(
                        component, 0
                    )
            aggregate_stats["blocks_started"] += stats.get("blocks_started", 0)
            aggregate_stats["steps_started"] += stats.get("steps_started", 0)

        task_outputs = [
            make_code_task_output(task, response)
            for task, response in zip(self.tasks, responses)
        ]
        return {
            "generated_token_ids": generated_token_ids,
            "responses": responses,
            "decoded": decoded,
            "task_outputs": task_outputs,
            "elapsed_seconds": total_elapsed,
            "cache_stats": aggregate_stats,
        }

    def load_or_create_baseline(self, baseline_path: Path) -> BaselineRecord:
        baseline_path = resolve_experiment_result_path(baseline_path)
        sources = [baseline_path]
        external = self.config.evaluation.baseline_artifact
        if external:
            external_path = resolve_experiment_result_path(external).resolve()
            if not external_path.is_file():
                raise FileNotFoundError(
                    f"Configured baseline artifact not found: {external_path}"
                )
            if external_path != baseline_path.resolve():
                sources.append(external_path)
        if self.config.run.resume:
            for source in sources:
                if not source.is_file():
                    continue
                data = json.loads(source.read_text(encoding="utf-8"))
                record = self._record_from_data(data, source)
                if record is None:
                    continue
                self._baseline = record
                if source.resolve() != baseline_path.resolve():
                    self._write_json_atomic(baseline_path, record.to_dict())
                    self._all_recompute_schedule().to_json(
                        baseline_path.with_name("baseline_schedule.json")
                    )
                    print(f"Reused validated baseline artifact: {source}")
                return self._baseline

        schedule = self._all_recompute_schedule()
        for _ in range(self.config.evaluation.baseline_warmup_runs):
            self._run_schedule(schedule)
        result = self._run_schedule(schedule)
        self._baseline = BaselineRecord(
            signature=self.signature,
            generated_token_ids=result["generated_token_ids"],
            responses=result["responses"],
            decoded=result["decoded"],
            task_outputs=result["task_outputs"],
            elapsed_seconds=result["elapsed_seconds"],
        )
        self._write_json_atomic(baseline_path, self._baseline.to_dict())
        schedule.to_json(baseline_path.with_name("baseline_schedule.json"))
        return self._baseline

    def prepare_quality_objective(self, result_path: Path) -> None:
        """Prepare any expensive baseline state needed by the quality objective."""

        if self.config.evaluation.quality_objective == "token_agreement":
            return
        if self._baseline is None:
            raise RuntimeError("load_or_create_baseline() must be called first")
        image = Path(str(self.config.evaluation.evalplus_image)).resolve()
        if not image.is_file():
            raise FileNotFoundError(f"EvalPlus image not found: {image}")
        image_stat = image.stat()
        self._evalplus_image_identity = {
            "path": str(image),
            "device": image_stat.st_dev,
            "inode": image_stat.st_ino,
            "size": image_stat.st_size,
            "mtime_ns": image_stat.st_mtime_ns,
            "sha256": self._file_sha256(image),
        }
        self._evalplus_dataset_identities = self._current_evalplus_dataset_identities()
        execution_signature_payload = {
            "objective_version": "evalplus-base-regression-v1",
            "baseline_signature": self.signature,
            "task_outputs_sha256": self._task_outputs_sha256(
                self._baseline.task_outputs
            ),
            "postprocessor_version": CODE_POSTPROCESSOR_VERSION,
            "image": self._evalplus_image_identity,
            "datasets": self._evalplus_dataset_identities,
        }
        execution_signature = hashlib.sha256(
            json.dumps(execution_signature_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
        if result_path.is_file() and self.config.run.resume:
            saved = json.loads(result_path.read_text(encoding="utf-8"))
            if saved.get("execution_signature") == execution_signature:
                result = saved.get("result", {})
                # Validate the task set and ensure that a useful baseline exists.
                compare_base_correctness(result, result)
                self._baseline_execution = result
                print(f"Reused sandboxed baseline base-test results: {result_path}")
                return
            print(f"Ignoring stale baseline base-test results: {result_path}")

        print("Running sandboxed EvalPlus base tests for the all-recompute baseline")
        result = run_evalplus_apptainer(
            self._baseline.task_outputs,
            image=str(image),
            work_dir=self.config.run_dir / "code_execution" / "baseline_base",
            parallel=self.config.evaluation.evalplus_parallel,
            apptainer_bin=self.config.evaluation.evalplus_apptainer_bin,
            scope="base",
            timeout_seconds=self.config.evaluation.evalplus_timeout_seconds,
            dataset_paths={
                dataset: identity["path"]
                for dataset, identity in self._evalplus_dataset_identities.items()
            },
        )
        compare_base_correctness(result, result)
        self._baseline_execution = result
        self._write_json_atomic(
            result_path,
            {
                "execution_signature": execution_signature,
                "signature_inputs": execution_signature_payload,
                "result": result,
            },
        )

    def generate_candidate(self, schedule: DLMCacheSchedule) -> dict[str, Any]:
        """Generate and measure one candidate without executing its programs."""

        if self._baseline is None:
            raise RuntimeError("load_or_create_baseline() must be called first")
        result = self._run_schedule(schedule)
        candidate_tokens = result.pop("generated_token_ids")
        baseline_tokens = self._baseline.generated_token_ids
        if len(candidate_tokens) != len(baseline_tokens):
            raise RuntimeError("Candidate and baseline prompt counts differ")

        matches = 0
        total = 0
        exact_matches = 0
        for candidate, baseline in zip(candidate_tokens, baseline_tokens):
            if len(candidate) != len(baseline):
                raise RuntimeError("Candidate and baseline generation lengths differ")
            equal = [left == right for left, right in zip(candidate, baseline)]
            matches += sum(equal)
            total += len(equal)
            exact_matches += int(all(equal))

        token_agreement = matches / total if total else 1.0
        exact_match_fraction = exact_matches / len(baseline_tokens)
        compute_fraction = schedule.compute_fraction(
            self.attention_weight, self.mlp_weight, self.fixed_step_weight
        )
        result.update(
            {
                "token_agreement_loss": 1.0 - token_agreement,
                "token_agreement": token_agreement,
                "exact_match_fraction": exact_match_fraction,
                "compute_fraction": compute_fraction,
                "speedup_vs_baseline": (
                    self._baseline.elapsed_seconds / result["elapsed_seconds"]
                    if result["elapsed_seconds"] > 0
                    else None
                ),
                "component_weights": {
                    "attention": self.attention_weight,
                    "mlp": self.mlp_weight,
                    "fixed_per_step": self.fixed_step_weight,
                },
            }
        )
        return result

    def score_candidate(
        self, result: dict[str, Any], *, execution_work_dir: Path | None = None
    ) -> dict[str, Any]:
        """Attach the configured quality objective to generated candidate data."""

        objective = self.config.evaluation.quality_objective
        if objective == "token_agreement":
            result["objective_version"] = "token-agreement-v1"
            result["quality_loss"] = float(result["token_agreement_loss"])
            return result
        if objective != "base_regression_rate":
            raise RuntimeError(f"Unsupported quality objective: {objective}")
        if self._baseline_execution is None:
            raise RuntimeError("prepare_quality_objective() must be called first")
        if self._evalplus_image_identity is None:
            raise RuntimeError("EvalPlus image identity was not recorded")
        if self._evalplus_dataset_identities is None:
            raise RuntimeError("EvalPlus dataset identities were not recorded")
        if execution_work_dir is None:
            raise ValueError("execution_work_dir is required for code correctness")

        image = Path(self._evalplus_image_identity["path"])
        image_stat = image.stat()
        current_identity = {
            "device": image_stat.st_dev,
            "inode": image_stat.st_ino,
            "size": image_stat.st_size,
            "mtime_ns": image_stat.st_mtime_ns,
        }
        expected_identity = {
            key: self._evalplus_image_identity[key]
            for key in current_identity
        }
        if current_identity != expected_identity:
            raise RuntimeError("EvalPlus image changed after baseline evaluation")
        current_datasets = self._current_evalplus_dataset_identities()
        if current_datasets != self._evalplus_dataset_identities:
            raise RuntimeError("EvalPlus data changed after baseline evaluation")

        candidate_execution = run_evalplus_apptainer(
            result["task_outputs"],
            image=str(image),
            work_dir=execution_work_dir,
            parallel=self.config.evaluation.evalplus_parallel,
            apptainer_bin=self.config.evaluation.evalplus_apptainer_bin,
            scope="base",
            timeout_seconds=self.config.evaluation.evalplus_timeout_seconds,
            dataset_paths={
                dataset: identity["path"]
                for dataset, identity in self._evalplus_dataset_identities.items()
            },
        )
        correctness = compare_base_correctness(
            self._baseline_execution, candidate_execution
        )
        syntax_invalid = sorted(
            str(output["task_id"])
            for output in result["task_outputs"]
            if output.get("syntax_valid") is False
        )
        correctness.update(
            {
                "candidate_base_pass_at_1": candidate_execution["aggregate"][
                    "base_pass_at_1"
                ],
                "syntax_invalid_count": len(syntax_invalid),
                "syntax_invalid_task_ids": syntax_invalid,
                "evalplus_image_sha256": self._evalplus_image_identity["sha256"],
                "evalplus_dataset_sha256": {
                    dataset: identity["sha256"]
                    for dataset, identity in self._evalplus_dataset_identities.items()
                },
            }
        )
        result.update(
            {
                "objective_version": correctness["objective_version"],
                "quality_loss": correctness["base_regression_rate"],
                "correctness": correctness,
                "code_execution": candidate_execution,
            }
        )
        return result

    def evaluate(
        self,
        schedule: DLMCacheSchedule,
        *,
        execution_work_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Generate and fully score one candidate."""

        result = self.generate_candidate(schedule)
        return self.score_candidate(
            result, execution_work_dir=execution_work_dir
        )
