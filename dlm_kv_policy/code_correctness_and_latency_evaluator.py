"""Paired coding correctness and measured generation latency for KV policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import random
from statistics import median
from typing import Any

import torch

from ecad.language_models.code_tasks_and_response_processing import (
    CodeGenerationTask,
    load_code_task_dataset,
    make_code_task_output,
)
from ecad.language_models.code_execution import compare_base_correctness, run_evalplus_apptainer
from ecad.language_models.experiment_artifact_paths import resolve_experiment_result_path
from ecad.language_models.diffucoder_loader import (
    load_diffucoder_model_and_tokenizer,
    prepare_diffucoder_prompt_batch,
)

from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .kv_policy_experiment_config import KVPolicyExperimentConfig
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .cache_and_trajectory_diagnostics import trajectory_metrics
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from .fixed_token_refresh_policy import FixedTokenRefreshPolicy
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler


# Persisted identifiers remain stable so existing policy artifacts can be reused.
FULL_EVALUATION_MODE = "full_validation_v1"
EVOLUTION_EVALUATION_MODE = "evolution_selection_fast_v1"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _merge_diagnostics(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge online summaries across prompts without retaining model tensors."""

    active: dict[str, list[int]] = {}
    reasons: dict[str, int] = {}
    lm_rows = [0, 0]
    drift: dict[str, list[float]] = {}
    for item in items:
        for key, value in item.get("active_rows_by_step_layer", {}).items():
            current = active.setdefault(key, [0, 0])
            current[0] += int(value["active_rows"])
            current[1] += int(value["total_rows"])
        for key, value in item.get("policy_reason_counts", {}).items():
            reasons[key] = reasons.get(key, 0) + int(value)
        lm = item.get("lm_head", {})
        lm_rows[0] += int(lm.get("active_rows", 0))
        lm_rows[1] += int(lm.get("total_rows", 0))
        for key, value in item.get("kv_drift", {}).items():
            observations = int(value["observations"])
            current = drift.setdefault(key, [0.0, 0.0, 0.0, 0.0, 0.0])
            current[0] += observations * float(value["relative_l2_k"])
            current[1] += observations * float(value["relative_l2_v"])
            current[2] += observations * float(value["cosine_k"])
            current[3] += observations * float(value["cosine_v"])
            current[4] += observations
    return {
        "active_rows_by_step_layer": {
            key: {
                "active_rows": values[0],
                "total_rows": values[1],
                "active_fraction": values[0] / values[1] if values[1] else 0.0,
            }
            for key, values in sorted(active.items())
        },
        "lm_head": {
            "active_rows": lm_rows[0],
            "total_rows": lm_rows[1],
            "active_fraction": lm_rows[0] / lm_rows[1] if lm_rows[1] else 0.0,
        },
        "policy_reason_counts": reasons,
        "kv_drift": {
            key: {
                "relative_l2_k": values[0] / values[4],
                "relative_l2_v": values[1] / values[4],
                "cosine_k": values[2] / values[4],
                "cosine_v": values[3] / values[4],
                "observations": int(values[4]),
            }
            for key, values in sorted(drift.items())
            if values[4]
        },
    }


@dataclass
class PolicyGenerationArtifact:
    record: dict[str, Any]
    histories: list[list[torch.Tensor]]


class DiffuCoderKVPolicyEvaluator:
    """Runs fixed policies against a paired stock-DiffuCoder dense baseline."""

    def __init__(self, config: KVPolicyExperimentConfig) -> None:
        self.config = config
        self.tasks: list[CodeGenerationTask] = load_code_task_dataset(
            config.evaluation.code_tasks_file
        )[: config.evaluation.max_prompts]
        if len(self.tasks) != config.evaluation.max_prompts:
            raise ValueError("Code task dataset has fewer tasks than evaluation.max_prompts")
        self.model, self.tokenizer = load_diffucoder_model_and_tokenizer(
            config.model.path,
            device=config.model.device,
            dtype=config.model.dtype,
            local_files_only=config.model.local_files_only,
        )
        self.signature = self._signature()

    def _signature(self) -> str:
        device_name = torch.cuda.get_device_name(next(self.model.parameters()).device)
        payload = {
            "project": "dlm-kv-policy-v1",
            "model_path": self.config.model.path,
            "device": device_name,
            "dtype": self.config.model.dtype,
            "generation": self.config.generation.__dict__,
            "tasks": [task.to_dict() for task in self.tasks],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def _run_once(
        self,
        policy: Any,
        *,
        collect_history: bool,
        collect_diagnostics: bool = True,
        tasks: list[CodeGenerationTask] | None = None,
    ) -> PolicyGenerationArtifact:
        """Generate one task set with either stock dense or active-row execution."""

        tasks = self.tasks if tasks is None else tasks
        backend = (
            DiffuCoderActiveTokenKVBackend(
                self.model,
                policy,
                collect_diagnostics=collect_diagnostics,
            )
            if not policy.force_dense
            else DenseDiffuCoderBaseline(self.model)
        )
        sampler = DiffuCoderDenoisingSampler(backend)
        token_ids: list[list[int]] = []
        responses: list[str] = []
        histories: list[list[torch.Tensor]] = []
        diagnostics: list[dict[str, Any]] = []
        elapsed = 0.0
        for index, task in enumerate(tasks):
            seed_everything(self.config.generation.seed + index)
            prompt, attention_mask = prepare_diffucoder_prompt_batch(
                self.tokenizer, [task.prompt], self.config.model.device
            )
            result = sampler.generate(
                prompt,
                attention_mask=attention_mask,
                steps=self.config.generation.steps,
                gen_length=self.config.generation.gen_length,
                temperature=self.config.generation.temperature,
                top_p=self.config.generation.top_p,
                top_k=self.config.generation.top_k,
                algorithm=self.config.generation.algorithm,
                algorithm_temperature=self.config.generation.algorithm_temperature,
                collect_history=collect_history,
                collect_diagnostics=collect_diagnostics,
            )
            generated = result.sequences[:, prompt.shape[1] :]
            token_ids.append(generated[0].cpu().tolist())
            responses.extend(self.tokenizer.batch_decode(generated, skip_special_tokens=True))
            histories.append(result.history)
            diagnostics.append(result.diagnostics)
            elapsed += result.elapsed_seconds
        task_outputs = [
            make_code_task_output(task, response) for task, response in zip(tasks, responses)
        ]
        return PolicyGenerationArtifact(
            record={
                "signature": self.signature,
                "policy": {"name": policy.name, **policy.to_dict()},
                "generated_token_ids": token_ids,
                "responses": responses,
                "task_outputs": task_outputs,
                "elapsed_seconds": elapsed,
                "diagnostics": _merge_diagnostics(diagnostics),
            },
            histories=histories,
        )

    def _measure(self, policy: Any) -> list[float]:
        """Timing-only reruns: no histories, cache diagnostics, or code evaluation."""

        self._run_once(policy, collect_history=False, collect_diagnostics=False)
        return [
            self._run_once(policy, collect_history=False, collect_diagnostics=False).record[
                "elapsed_seconds"
            ]
            for _ in range(self.config.evaluation.timing_repeats)
        ]

    def _dkv_reference(self) -> DKVDecodeRefreshPolicy:
        return DKVDecodeRefreshPolicy(
            cache_reloading_step=self.config.dkv_reference.cache_reloading_step
        )

    def _baseline_path(self) -> Path:
        return self.config.run.run_dir / "baseline.json"

    def create_or_load_baseline(self) -> PolicyGenerationArtifact:
        path = self._baseline_path()
        if self.config.run.resume and path.is_file():
            saved = _read_json(path)
            if saved.get("signature") == self.signature:
                return PolicyGenerationArtifact(record=saved, histories=[])
            print(f"Ignoring baseline with different signature: {path}", flush=True)
        dense = self.config.policies.get("dense")
        if dense is None or not dense.force_dense:
            raise ValueError("policies.dense must exist and set force_dense: true")
        for _ in range(self.config.evaluation.baseline_warmup_runs):
            self._run_once(dense, collect_history=False)
        generated = self._run_once(dense, collect_history=True)
        samples = self._measure(dense)
        generated.record["timing_samples_seconds"] = samples
        generated.record["median_elapsed_seconds"] = median(samples)
        _write_json(path, generated.record)
        return generated

    def _evalplus(
        self, outputs: list[dict[str, Any]], work_dir: Path, *, scope: str = "base"
    ) -> dict[str, Any]:
        settings = self.config.evaluation
        return run_evalplus_apptainer(
            outputs,
            image=settings.evalplus_image,
            work_dir=work_dir,
            parallel=settings.evalplus_parallel,
            apptainer_bin=settings.evalplus_apptainer_bin,
            scope=scope,
            timeout_seconds=settings.evalplus_timeout_seconds,
            dataset_paths={
                "humaneval": settings.evalplus_humaneval_path,
                "mbpp": settings.evalplus_mbpp_path,
            },
        )

    def _baseline_execution(self, baseline: PolicyGenerationArtifact) -> dict[str, Any]:
        path = self.config.run.run_dir / "baseline.base_eval.json"
        if self.config.run.resume and path.is_file():
            saved = _read_json(path)
            if saved.get("baseline_signature") == self.signature:
                return saved["result"]
        result = self._evalplus(
            baseline.record["task_outputs"], self.config.run.run_dir / "code_execution" / "baseline"
        )
        _write_json(path, {"baseline_signature": self.signature, "result": result})
        return result

    def evaluate_policy(
        self,
        policy: Any,
        *,
        baseline: PolicyGenerationArtifact,
        baseline_execution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        path = self.config.run.run_dir / "policies" / f"{policy.name}.json"
        if self.config.run.resume and path.is_file():
            saved = _read_json(path)
            expected_policy = {"name": policy.name, **policy.to_dict()}
            if (
                saved.get("signature") == self.signature
                and saved.get("policy") == expected_policy
                and saved.get("evaluation_mode", FULL_EVALUATION_MODE) == FULL_EVALUATION_MODE
            ):
                return saved
            print(f"Ignoring stale policy artifact: {path}", flush=True)
        generated = self._run_once(policy, collect_history=True)
        samples = self._measure(policy)
        result = generated.record
        result["evaluation_mode"] = FULL_EVALUATION_MODE
        result["timing_samples_seconds"] = samples
        result["median_elapsed_seconds"] = median(samples)
        result["speedup_vs_dense"] = baseline.record["median_elapsed_seconds"] / result["median_elapsed_seconds"]
        if baseline.histories:
            per_task = [
                trajectory_metrics(left, right)
                for left, right in zip(baseline.histories, generated.histories)
            ]
            result["trajectory_diagnostics"] = {
                "per_task": per_task,
                "first_divergence_steps": [item["first_trajectory_divergence_step"] for item in per_task],
                "mean_token_agreement": sum(item["token_agreement"] for item in per_task) / len(per_task),
            }
        else:
            result["trajectory_diagnostics"] = {"available": False, "reason": "baseline was resumed"}
        if baseline_execution is not None:
            execution = self._evalplus(
                result["task_outputs"], self.config.run.run_dir / "code_execution" / policy.name
            )
            correctness = compare_base_correctness(baseline_execution, execution)
            invalid = [
                item["task_id"] for item in result["task_outputs"] if not item.get("syntax_valid", True)
            ]
            result.update(
                {
                    "quality_loss": correctness["base_regression_rate"],
                    "correctness": correctness,
                    "candidate_base_pass_at_1": execution["aggregate"]["base_pass_at_1"],
                    "syntax_invalid_count": len(invalid),
                    "syntax_invalid_task_ids": invalid,
                    "code_execution": execution,
                }
            )
        _write_json(path, result)
        return result

    def prepare_evolution_baseline(
        self, reuse_baseline: Path | None
    ) -> tuple[PolicyGenerationArtifact, dict[str, Any], dict[str, Any]]:
        """Warm once and time dense once for an entire evolutionary Slurm job."""

        if reuse_baseline is None:
            baseline = self.create_or_load_baseline()
            base_execution = self._baseline_execution(baseline)
            source = "generated_in_this_run"
        else:
            baseline, base_execution = self._load_reused_baseline(reuse_baseline)
            source = str(resolve_experiment_result_path(reuse_baseline).resolve())
        dense = self.config.policies["dense"]
        self._run_once(dense, collect_history=False, collect_diagnostics=False)  # GPU warm-up
        timed = self._run_once(dense, collect_history=False, collect_diagnostics=False)
        baseline.record = dict(baseline.record)
        baseline.record["timing_samples_seconds"] = [timed.record["elapsed_seconds"]]
        baseline.record["median_elapsed_seconds"] = timed.record["elapsed_seconds"]
        return baseline, base_execution, {
            "source": source,
            "timing_reference": "one dense pass after one warm-up, shared by this Slurm job",
            "timing_samples_seconds": baseline.record["timing_samples_seconds"],
            "median_elapsed_seconds": baseline.record["median_elapsed_seconds"],
            "base_pass_at_1": base_execution["aggregate"]["base_pass_at_1"],
        }

    def _evaluate_evolution_policy(
        self,
        policy: DKVDecodeRefreshPolicy,
        *,
        baseline: PolicyGenerationArtifact,
        baseline_execution: dict[str, Any],
    ) -> dict[str, Any]:
        """One no-diagnostics generation supplies both candidate code and timing."""

        path = self.config.run.run_dir / "policies" / f"{policy.name}.json"
        expected_policy = {"name": policy.name, **policy.to_dict()}
        if self.config.run.resume and path.is_file():
            saved = _read_json(path)
            if (
                saved.get("signature") == self.signature
                and saved.get("policy") == expected_policy
                and saved.get("evaluation_mode") == EVOLUTION_EVALUATION_MODE
            ):
                return saved

        generated = self._run_once(policy, collect_history=False, collect_diagnostics=False)
        result = generated.record
        elapsed = float(result["elapsed_seconds"])
        result.update(
            {
                "evaluation_mode": EVOLUTION_EVALUATION_MODE,
                "timing_protocol": "one timed no-diagnostics generation after shared dense warm-up",
                "timing_samples_seconds": [elapsed],
                "median_elapsed_seconds": elapsed,
                "speedup_vs_dense": baseline.record["median_elapsed_seconds"] / elapsed,
                "trajectory_diagnostics": {
                    "available": False,
                    "reason": "recorded only during repeated-timing policy evaluation",
                },
            }
        )
        execution = self._evalplus(
            result["task_outputs"], self.config.run.run_dir / "code_execution" / policy.name
        )
        correctness = compare_base_correctness(baseline_execution, execution)
        invalid = [
            item["task_id"] for item in result["task_outputs"] if not item.get("syntax_valid", True)
        ]
        result.update(
            {
                "quality_loss": correctness["base_regression_rate"],
                "correctness": correctness,
                "candidate_base_pass_at_1": execution["aggregate"]["base_pass_at_1"],
                "syntax_invalid_count": len(invalid),
                "syntax_invalid_task_ids": invalid,
                "code_execution": execution,
            }
        )
        _write_json(path, result)
        return result

    def evaluate_evolution_policies(
        self,
        policies: list[DKVDecodeRefreshPolicy],
        *,
        baseline: PolicyGenerationArtifact,
        baseline_execution: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Evaluate a population with the shared dense reference from this job."""

        return [
            self._evaluate_evolution_policy(
                policy, baseline=baseline, baseline_execution=baseline_execution
            )
            for policy in policies
        ]

    def _load_reused_baseline(self, source_dir: Path) -> tuple[PolicyGenerationArtifact, dict[str, Any]]:
        """Load a hardware- and signature-matched dense baseline without copying it."""

        source = resolve_experiment_result_path(source_dir).resolve()
        baseline_path = source / "baseline.json"
        execution_path = source / "baseline.base_eval.json"
        if not baseline_path.is_file() or not execution_path.is_file():
            raise FileNotFoundError(
                f"Expected baseline.json and baseline.base_eval.json under {source}"
            )
        baseline = _read_json(baseline_path)
        execution = _read_json(execution_path)
        if baseline.get("signature") != self.signature:
            raise RuntimeError("Refusing a dense baseline with a different signature")
        if execution.get("baseline_signature") != self.signature:
            raise RuntimeError("Refusing EvalPlus base results with a different signature")
        result = execution.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("Baseline EvalPlus artifact has no result mapping")
        return PolicyGenerationArtifact(record=baseline, histories=[]), result

    def _full_dkv_baseline(
        self, reuse_baseline: Path | None
    ) -> tuple[PolicyGenerationArtifact, dict[str, Any], str]:
        """Get paired outcomes and fresh dense timing for one full dKV job."""
        if reuse_baseline is None:
            baseline = self.create_or_load_baseline()
            baseline_execution = self._baseline_execution(baseline)
            source = "generated_in_this_run"
        else:
            baseline, baseline_execution = self._load_reused_baseline(reuse_baseline)
            source = str(resolve_experiment_result_path(reuse_baseline).resolve())
            # EvalPlus outcomes are signature-matched and reusable, but timing
            # must be measured by this job on this GPU allocation.
            timing_samples = self._measure(self.config.policies["dense"])
            baseline.record = dict(baseline.record)
            baseline.record["timing_samples_seconds"] = timing_samples
            baseline.record["median_elapsed_seconds"] = median(timing_samples)
        return baseline, baseline_execution, source

    def _summarize_dkv_full(
        self, policies: list[DKVDecodeRefreshPolicy], *, reuse_baseline: Path | None
    ) -> dict[str, Any]:
        baseline, records = self.evaluate_dkv_policies(policies, reuse_baseline=reuse_baseline)
        summaries = [
            {
                "name": policy.name,
                "parameters": policy.to_dict(),
                "candidate_base_pass_at_1": record.get("candidate_base_pass_at_1"),
                "base_regression_rate": record.get("quality_loss"),
                "speedup_vs_dense": record["speedup_vs_dense"],
                "syntax_invalid_count": record.get("syntax_invalid_count"),
            }
            for policy, record in zip(policies, records)
        ]
        return {
            "signature": self.signature,
            "objectives": ["base_regression_rate", "median_generation_wall_time"],
            "baseline": baseline,
            "policies": summaries,
        }

    def evaluate_dkv_policies(
        self, policies: list[DKVDecodeRefreshPolicy], *, reuse_baseline: Path | None
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """Evaluate policies against one shared dense timing reference."""

        if not policies:
            raise ValueError("At least one dKV policy is required")
        dense, base_execution, source = self._full_dkv_baseline(reuse_baseline)
        records = [
            self.evaluate_policy(policy, baseline=dense, baseline_execution=base_execution)
            for policy in policies
        ]
        return (
            {
                "source": source,
                "timing_reference": "measured_in_this_job",
                "timing_samples_seconds": dense.record["timing_samples_seconds"],
                "median_elapsed_seconds": dense.record["median_elapsed_seconds"],
                "base_pass_at_1": base_execution["aggregate"]["base_pass_at_1"],
            },
            records,
        )

    def evaluate_dkv_decode_reference(self, *, reuse_baseline: Path | None = None) -> dict[str, Any]:
        """Evaluate the fixed native-SDPA dKV rule on all 64 paired base tasks."""

        summary = self._summarize_dkv_full([self._dkv_reference()], reuse_baseline=reuse_baseline)
        summary["policy"] = summary["policies"][0]
        _write_json(self.config.run.run_dir / "dkv_decode_reference_evaluation_summary.json", summary)
        return summary

    def evaluate_dkv_decode_ablations(
        self, policy_names: list[str] | None, *, reuse_baseline: Path | None = None
    ) -> dict[str, Any]:
        """Evaluate selected dKV ablations; normally select one policy per job."""

        selected = policy_names or list(self.config.dkv_policies)
        unknown = sorted(set(selected) - set(self.config.dkv_policies))
        if unknown:
            raise ValueError(f"Unknown dKV ablation policies: {unknown}")
        if not selected:
            raise ValueError("No dKV ablation policies were configured")
        summary = self._summarize_dkv_full(
            [self.config.dkv_policies[name] for name in selected], reuse_baseline=reuse_baseline
        )
        _write_json(self.config.run.run_dir / "dkv_decode_ablation_evaluation_summary.json", summary)
        return summary

    def run(self, policy_names: list[str] | None = None) -> dict[str, Any]:
        selected = policy_names or self.config.fixed_policy_evaluation.policies
        unknown = sorted(set(selected) - set(self.config.policies))
        if unknown:
            raise ValueError(f"Unknown policies requested: {unknown}")
        baseline = self.create_or_load_baseline()
        baseline_execution = self._baseline_execution(baseline) if self.config.fixed_policy_evaluation.run_evalplus else None
        for name in selected:
            if name == "dense":
                continue
            self.evaluate_policy(
                self.config.policies[name], baseline=baseline, baseline_execution=baseline_execution
            )
        results = []
        for name in self.config.fixed_policy_evaluation.policies:
            if name == "dense":
                continue
            path = self.config.run.run_dir / "policies" / f"{name}.json"
            if path.is_file():
                record = _read_json(path)
                if record.get("signature") == self.signature:
                    results.append(record)
        summary = {
            "signature": self.signature,
            "objectives": ["base_regression_rate", "median_generation_wall_time"],
            "baseline": {
                "median_elapsed_seconds": baseline.record["median_elapsed_seconds"],
                "base_pass_at_1": (
                    baseline_execution["aggregate"]["base_pass_at_1"] if baseline_execution else None
                ),
            },
            "policies": [
                {
                    "name": item["policy"].get("name", "unknown"),
                    "quality_loss": item.get("quality_loss"),
                    "speedup_vs_dense": item["speedup_vs_dense"],
                    "syntax_invalid_count": item.get("syntax_invalid_count"),
                }
                for item in results
            ],
        }
        _write_json(self.config.run.run_dir / "summary.json", summary)
        return summary

    def evaluate_heldout_plus(
        self, policy_name: str, *, reuse_baseline: Path | None = None
    ) -> dict[str, Any]:
        """Run EvalPlus Plus on an existing candidate or the dense reference."""

        if policy_name == "dense":
            if reuse_baseline is None:
                record = self.create_or_load_baseline().record
                source = "generated_in_this_run"
            else:
                baseline, _ = self._load_reused_baseline(reuse_baseline)
                record = baseline.record
                source = str(resolve_experiment_result_path(reuse_baseline).resolve())
        else:
            artifact = self.config.run.run_dir / "policies" / f"{policy_name}.json"
            if not artifact.is_file():
                raise FileNotFoundError(f"No evaluated policy artifact found: {artifact}")
            record = _read_json(artifact)
            source = str(artifact)
        if record.get("signature") != self.signature:
            raise RuntimeError("Refusing held-out evaluation for a different baseline signature")
        result = self._evalplus(
            record["task_outputs"],
            self.config.run.run_dir / "code_execution" / f"{policy_name}_plus",
            scope="full",
        )
        path = self.config.run.run_dir / "policies" / f"{policy_name}.plus_eval.json"
        payload = {
            "signature": self.signature,
            "policy_name": policy_name,
            "source": source,
            "result": result,
        }
        _write_json(path, payload)
        return payload
