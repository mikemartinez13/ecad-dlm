"""Validated YAML configuration for DLM ECAD runs."""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any

import yaml

from ecad.language_models.experiment_artifact_paths import (
    DEFAULT_RESULTS_ROOT,
    resolve_experiment_result_path,
)


@dataclass
class ModelSettings:
    path: str
    family: str = "llada"
    device: str = "cuda"
    dtype: str = "bfloat16"
    local_files_only: bool = True

    def validate(self) -> None:
        self.family = self.family.lower()
        if self.family not in {"llada", "diffucoder"}:
            raise ValueError("model.family must be llada or diffucoder")
        if not self.path:
            raise ValueError("model.path must not be empty")


@dataclass
class GenerationSettings:
    steps: int = 64
    gen_length: int = 64
    block_length: int = 32
    temperature: float = 0.0
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"
    mask_id: int = 126336
    eos_token_id: int = 126081
    eot_token_id: int = 126348
    logits_eos_inf: bool = False
    confidence_eos_eot_inf: bool = False
    # DiffuCoder's native Dream sampler settings. They are retained in the
    # resolved config for LLaDA runs but ignored by that backend.
    top_p: float = 0.95
    top_k: int | None = None
    algorithm: str = "entropy"
    algorithm_temperature: float = 0.0

    @property
    def num_generation_blocks(self) -> int:
        return self.gen_length // self.block_length

    @property
    def local_steps(self) -> int:
        return self.steps // self.num_generation_blocks

    def validate(self, model_family: str = "llada") -> None:
        if self.steps < 2 or self.gen_length < 1:
            raise ValueError("ECAD requires at least two steps and positive gen_length")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if model_family == "diffucoder":
            if not 0 < self.top_p <= 1:
                raise ValueError("top_p must be in (0, 1]")
            if self.top_k is not None and self.top_k < 1:
                raise ValueError("top_k must be positive when provided")
            if self.algorithm not in {
                "origin",
                "maskgit_plus",
                "topk_margin",
                "entropy",
            }:
                raise ValueError("Unsupported DiffuCoder generation algorithm")
            if self.algorithm_temperature < 0:
                raise ValueError("algorithm_temperature must be non-negative")
            return

        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        if self.gen_length % self.block_length:
            raise ValueError("gen_length must be divisible by block_length")
        if self.steps % self.num_generation_blocks:
            raise ValueError("steps must be divisible by generation blocks")
        if self.local_steps < 2:
            raise ValueError("ECAD requires at least two local denoising steps")
        if self.local_steps > self.block_length:
            raise ValueError("local_steps cannot exceed block_length")
        if self.remasking not in {"low_confidence", "random"}:
            raise ValueError("remasking must be low_confidence or random")
        if self.eos_token_id < 0 or self.eot_token_id < 0:
            raise ValueError("EOS/EOT token IDs must be non-negative")


@dataclass
class ScheduleSettings:
    step_group_size: int = 4
    layer_group_size: int = 4
    attention_weight: float | None = None
    mlp_weight: float | None = None
    fixed_step_weight: float | None = None

    def validate(self) -> None:
        if self.step_group_size < 1 or self.layer_group_size < 1:
            raise ValueError("Schedule group sizes must be positive")
        if self.attention_weight is not None and self.attention_weight <= 0:
            raise ValueError("attention_weight must be positive")
        if self.mlp_weight is not None and self.mlp_weight <= 0:
            raise ValueError("mlp_weight must be positive")
        if self.fixed_step_weight is not None and self.fixed_step_weight < 0:
            raise ValueError("fixed_step_weight must be non-negative")


@dataclass
class EvaluationSettings:
    prompts_file: str | None = None
    source: str = "text"
    code_tasks_file: str | None = None
    code_datasets: list[str] = field(default_factory=lambda: ["humaneval", "mbpp"])
    code_max_tasks_per_dataset: int | None = None
    code_seed: int = 0
    code_prompt_style: str = "strict_python"
    max_prompts: int = 8
    batch_size: int = 1
    baseline_warmup_runs: int = 1
    save_decoded_text: bool = True
    # Token agreement remains available for non-code/mechanics runs. Coding
    # searches can instead execute only the public base tests inside Apptainer
    # and minimize regressions relative to the paired all-recompute baseline.
    quality_objective: str = "token_agreement"
    evalplus_image: str | None = None
    # EvalPlus normally downloads these lazily. Correctness runs deliberately
    # disable networking, so pin the two downloaded benchmark files explicitly.
    evalplus_humaneval_path: str | None = None
    evalplus_mbpp_path: str | None = None
    evalplus_parallel: int = 8
    evalplus_apptainer_bin: str = "apptainer"
    evalplus_timeout_seconds: int = 1800
    baseline_artifact: str | None = None
    max_syntax_invalid: int = 0

    def validate(self) -> None:
        if self.source not in {"text", "evalplus"}:
            raise ValueError("evaluation.source must be text or evalplus")
        if self.source == "text" and not self.prompts_file:
            raise ValueError("evaluation.prompts_file is required for text source")
        if self.source == "evalplus" and self.code_tasks_file is not None:
            if not Path(self.code_tasks_file).is_absolute():
                raise ValueError("evaluation.code_tasks_file must be absolute")
        if self.source == "evalplus" and self.code_prompt_style not in {
            "strict_python",
            "opencompass",
        }:
            raise ValueError(
                "evaluation.code_prompt_style must be strict_python or opencompass"
            )
        if self.source == "evalplus" and not self.code_datasets and not self.code_tasks_file:
            raise ValueError("At least one code dataset is required")
        if self.code_max_tasks_per_dataset is not None and self.code_max_tasks_per_dataset < 1:
            raise ValueError("code_max_tasks_per_dataset must be positive")
        if (
            self.max_prompts < 1
            or self.batch_size < 1
            or self.baseline_warmup_runs < 0
        ):
            raise ValueError(
                "max_prompts/batch_size must be positive and warmup runs non-negative"
            )
        if self.quality_objective not in {
            "token_agreement",
            "base_regression_rate",
        }:
            raise ValueError(
                "evaluation.quality_objective must be token_agreement or "
                "base_regression_rate"
            )
        if self.quality_objective == "base_regression_rate":
            if self.source != "evalplus":
                raise ValueError(
                    "base_regression_rate requires evaluation.source=evalplus"
                )
            if not self.evalplus_image:
                raise ValueError(
                    "evaluation.evalplus_image is required for "
                    "base_regression_rate"
                )
            if not Path(self.evalplus_image).is_absolute():
                raise ValueError("evaluation.evalplus_image must be absolute")
            for name in ("evalplus_humaneval_path", "evalplus_mbpp_path"):
                value = getattr(self, name)
                if not value:
                    raise ValueError(
                        f"evaluation.{name} is required for "
                        "base_regression_rate"
                    )
                if not Path(value).is_absolute():
                    raise ValueError(f"evaluation.{name} must be absolute")
        if self.evalplus_parallel < 1 or self.evalplus_timeout_seconds < 1:
            raise ValueError(
                "evalplus_parallel and evalplus_timeout_seconds must be positive"
            )
        if self.max_syntax_invalid < 0:
            raise ValueError("max_syntax_invalid must be non-negative")
        if not self.evalplus_apptainer_bin:
            raise ValueError("evalplus_apptainer_bin must not be empty")
        if self.baseline_artifact is not None:
            if self.source != "evalplus":
                raise ValueError("baseline_artifact is only supported for code runs")
            if not Path(self.baseline_artifact).is_absolute():
                raise ValueError("evaluation.baseline_artifact must be absolute")


@dataclass
class EvolutionSettings:
    population_size: int = 16
    generations: int = 10
    seed: int = 0
    crossover_probability: float = 0.9
    crossover_points: int = 4
    mutation_probability: float = 0.05
    # When set, overrides mutation_probability with a probability derived from
    # the genome length so each child changes this many grouped decisions on
    # average. This keeps mutation local when schedule granularity changes.
    mutation_expected_flips: float | None = None
    initial_recompute_probability: float = 0.8
    # When set, initialize every individual with exactly this many cached
    # grouped decisions instead of sampling each decision independently.
    initial_cached_groups: int | None = None
    min_cached_groups: int = 1

    def validate(self) -> None:
        if self.population_size < 2 or self.generations < 1:
            raise ValueError("population_size >= 2 and generations >= 1 are required")
        for name in (
            "crossover_probability",
            "mutation_probability",
            "initial_recompute_probability",
        ):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.crossover_points < 1 or self.min_cached_groups < 0:
            raise ValueError("Invalid crossover_points or min_cached_groups")
        if (
            self.mutation_expected_flips is not None
            and self.mutation_expected_flips < 0
        ):
            raise ValueError("mutation_expected_flips must be non-negative")
        if self.initial_cached_groups is not None and self.initial_cached_groups < 0:
            raise ValueError("initial_cached_groups must be non-negative")


@dataclass
class RunSettings:
    name: str
    output_dir: str = str(DEFAULT_RESULTS_ROOT / "dlm_ecad")
    resume: bool = True


@dataclass
class DLMResidualCacheExperimentConfig:
    run: RunSettings
    model: ModelSettings
    generation: GenerationSettings
    schedule: ScheduleSettings
    evaluation: EvaluationSettings
    evolution: EvolutionSettings = field(default_factory=EvolutionSettings)

    def validate(self) -> None:
        if not self.run.name:
            raise ValueError("run.name must not be empty")
        self.model.validate()
        self.generation.validate(self.model.family)
        self.schedule.validate()
        self.evaluation.validate()
        self.evolution.validate()

    @property
    def num_schedule_steps(self) -> int:
        """Number of denoising decisions represented by one cache schedule."""

        if self.model.family == "llada":
            return self.generation.local_steps
        return self.generation.steps

    @property
    def schedule_domain(self) -> str:
        if self.model.family == "llada":
            return "local_step_repeated_per_block"
        return "global_denoising_step"

    @property
    def run_dir(self) -> Path:
        return resolve_experiment_result_path(self.run.output_dir) / self.run.name

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


def _expand_strings(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, dict):
        return {key: _expand_strings(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_strings(item) for item in value]
    return value


def load_dlm_residual_cache_config(path: str | Path) -> DLMResidualCacheExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    if not isinstance(raw, dict):
        raise TypeError("DLM config must be a YAML mapping")
    raw = _expand_strings(raw)
    config = DLMResidualCacheExperimentConfig(
        run=RunSettings(**raw["run"]),
        model=ModelSettings(**raw["model"]),
        generation=GenerationSettings(**raw["generation"]),
        schedule=ScheduleSettings(**raw["schedule"]),
        evaluation=EvaluationSettings(**raw["evaluation"]),
        evolution=EvolutionSettings(**raw.get("evolution", {})),
    )
    config.validate()
    return config
