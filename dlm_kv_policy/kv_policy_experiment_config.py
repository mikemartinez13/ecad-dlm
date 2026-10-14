"""Configuration for DiffuCoder token-level KV evaluation and NSGA-II search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ecad.language_models.experiment_artifact_paths import (
    DEFAULT_RESULTS_ROOT,
    resolve_experiment_result_path,
)

from .fixed_token_refresh_policy import FixedTokenRefreshPolicy
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy


@dataclass
class RunConfig:
    name: str
    output_dir: str = str(DEFAULT_RESULTS_ROOT / "dlm_kv_policy")
    resume: bool = True

    @property
    def run_dir(self) -> Path:
        return resolve_experiment_result_path(self.output_dir).resolve() / self.name


@dataclass
class ModelConfig:
    path: str
    device: str = "cuda"
    dtype: str = "bfloat16"
    local_files_only: bool = True


@dataclass
class GenerationConfig:
    steps: int = 256
    gen_length: int = 256
    temperature: float = 0.4
    top_p: float = 0.95
    top_k: int | None = None
    algorithm: str = "entropy"
    algorithm_temperature: float = 0.0
    seed: int = 0


@dataclass
class EvaluationConfig:
    code_tasks_file: str
    max_prompts: int = 64
    prompt_batch_size: int = 1
    evalplus_image: str = ""
    evalplus_humaneval_path: str = ""
    evalplus_mbpp_path: str = ""
    evalplus_parallel: int = 8
    evalplus_apptainer_bin: str = "apptainer"
    evalplus_timeout_seconds: int = 1800
    timing_repeats: int = 3
    baseline_warmup_runs: int = 1


@dataclass
class FixedPolicyEvaluationConfig:
    policies: list[str] = field(default_factory=list)
    run_evalplus: bool = True


@dataclass
class DkvReferenceConfig:
    cache_reloading_step: int = 8


@dataclass
class KVPolicyNSGA2SearchConfig:
    """Small categorical search space for the code-domain KV policy."""

    population_size: int = 8
    generations: int = 8
    seed: int = 0
    delays: list[int] = field(default_factory=lambda: [1, 2])
    refresh_intervals: list[int] = field(default_factory=lambda: [4, 8, 12, 16])
    refresh_depths: list[int] = field(default_factory=lambda: [1, 18, 20, 22, 24])
    terminal_refresh_steps: list[int] = field(default_factory=lambda: [0, 8, 16])
    crossover_probability: float = 0.9
    two_gene_mutation_probability: float = 0.2

    def validate(self) -> None:
        choices = (
            self.delays,
            self.refresh_intervals,
            self.refresh_depths,
            self.terminal_refresh_steps,
        )
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("evolution population_size must be >= 2 and generations positive")
        if any(not values for values in choices):
            raise ValueError("every evolution genome coordinate needs at least one choice")
        if any(delay < 1 for delay in self.delays):
            raise ValueError("evolution delays must be positive; immediate reuse is excluded")
        if any(value < 1 for value in self.refresh_intervals + self.refresh_depths):
            raise ValueError("evolution refresh intervals and depths must be positive")
        if any(value < 0 for value in self.terminal_refresh_steps):
            raise ValueError("evolution terminal refresh steps must be non-negative")
        search_space = 1
        for values in choices:
            search_space *= len(set(values))
        if search_space < self.population_size * self.generations:
            raise ValueError("evolution search space is smaller than all requested unique evaluations")
        if not 0 <= self.crossover_probability <= 1:
            raise ValueError("evolution crossover_probability must be in [0, 1]")
        if not 0 <= self.two_gene_mutation_probability <= 1:
            raise ValueError("evolution two_gene_mutation_probability must be in [0, 1]")


@dataclass
class KVPolicyExperimentConfig:
    run: RunConfig
    model: ModelConfig
    generation: GenerationConfig
    evaluation: EvaluationConfig
    fixed_policy_evaluation: FixedPolicyEvaluationConfig
    policies: dict[str, FixedTokenRefreshPolicy]
    dkv_reference: DkvReferenceConfig = field(default_factory=DkvReferenceConfig)
    dkv_policies: dict[str, DKVDecodeRefreshPolicy] = field(default_factory=dict)
    evolution: KVPolicyNSGA2SearchConfig | None = None

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["policies"] = {
            name: policy.to_dict() for name, policy in self.policies.items()
        }
        return data

    def validate(self) -> None:
        if self.model.device != "cuda" and not self.model.device.startswith("cuda:"):
            raise ValueError("The active-token KV backend is GPU-only; model.device must be cuda")
        if self.generation.steps < 2 or self.generation.gen_length < 1:
            raise ValueError("generation.steps must be >= 2 and gen_length positive")
        if not 0 < self.generation.top_p <= 1:
            raise ValueError("generation.top_p must be in (0, 1]")
        if self.generation.algorithm not in {
            "origin", "maskgit_plus", "topk_margin", "entropy"
        }:
            raise ValueError("Unsupported DiffuCoder generation algorithm")
        if self.evaluation.prompt_batch_size != 1:
            raise ValueError("The first active-row backend intentionally supports batch_size=1")
        if self.evaluation.max_prompts < 1 or self.evaluation.timing_repeats < 1:
            raise ValueError("max_prompts and timing_repeats must be positive")
        if not Path(self.evaluation.code_tasks_file).is_absolute():
            raise ValueError("evaluation.code_tasks_file must be absolute")
        if not self.fixed_policy_evaluation.policies:
            raise ValueError("fixed_policy_evaluation.policies must name at least one fixed policy")
        if self.dkv_reference.cache_reloading_step < 1:
            raise ValueError("dkv_reference.cache_reloading_step must be positive")
        unknown = sorted(set(self.fixed_policy_evaluation.policies) - set(self.policies))
        if unknown:
            raise ValueError(f"fixed_policy_evaluation.policies names unknown policies: {unknown}")
        for policy in self.policies.values():
            policy.validate()
        for policy in self.dkv_policies.values():
            policy.validate()
        if self.evolution is not None:
            self.evolution.validate()


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Missing mapping section {name!r}")
    return value



def _fixed_policy_evaluation_section(raw: dict[str, Any]) -> dict[str, Any]:
    if "fixed_policy_evaluation" in raw:
        return _section(raw, "fixed_policy_evaluation")
    # Saved experiment configs used this name before the runner supported full
    # policy evaluation and search. Retired diagnostic settings are not restored.
    legacy = _section(raw, "pilot")
    return {key: legacy[key] for key in ("policies", "run_evalplus") if key in legacy}


def load_kv_policy_experiment_config(path: str | Path) -> KVPolicyExperimentConfig:
    """Load a KV policy experiment config without importing the model or evaluator."""

    source = Path(path)
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Top-level YAML value must be a mapping")
    raw_policies = _section(raw, "policies")
    policies = {
        name: FixedTokenRefreshPolicy.from_dict(name, value)
        for name, value in raw_policies.items()
    }
    raw_dkv_policies = raw.get("dkv_policies", {})
    if not isinstance(raw_dkv_policies, dict):
        raise ValueError("dkv_policies must be a mapping when provided")
    dkv_policies = {
        str(name): DKVDecodeRefreshPolicy.from_dict(str(name), value)
        for name, value in raw_dkv_policies.items()
    }
    raw_evolution = raw.get("evolution")
    if raw_evolution is not None and not isinstance(raw_evolution, dict):
        raise ValueError("evolution must be a mapping when provided")
    config = KVPolicyExperimentConfig(
        run=RunConfig(**_section(raw, "run")),
        model=ModelConfig(**_section(raw, "model")),
        generation=GenerationConfig(**_section(raw, "generation")),
        evaluation=EvaluationConfig(**_section(raw, "evaluation")),
        fixed_policy_evaluation=FixedPolicyEvaluationConfig(**_fixed_policy_evaluation_section(raw)),
        policies=policies,
        dkv_reference=DkvReferenceConfig(**raw.get("dkv_reference", {})),
        dkv_policies=dkv_policies,
        evolution=None if raw_evolution is None else KVPolicyNSGA2SearchConfig(**raw_evolution),
    )
    config.validate()
    return config
