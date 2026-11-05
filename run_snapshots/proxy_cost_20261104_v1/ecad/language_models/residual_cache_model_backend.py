"""Backend boundary between ECAD and supported diffusion language models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterable

import torch

from ecad.language_models.diffucoder_residual_output_cache import attach_diffucoder_residual_cache
from ecad.language_models.diffucoder_generation import generate_diffucoder
from ecad.language_models.diffucoder_loader import (
    load_diffucoder_model_and_tokenizer,
    prepare_diffucoder_prompt_batch,
)
from ecad.language_models.residual_cache_experiment_config import DLMResidualCacheExperimentConfig
from ecad.language_models.llada_residual_output_cache import attach_llada_residual_cache
from ecad.language_models.llada_generation import generate_llada
from ecad.language_models.llada_loader import (
    load_llada_model_and_tokenizer,
    prepare_prompt_batch,
)
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


class DLMBackend(ABC):
    """Uniform model, sampler, decoding, and cache-controller interface."""

    family: str
    schedule_domain: str

    def __init__(self, model: torch.nn.Module, tokenizer, config: DLMResidualCacheExperimentConfig) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.controller: Any | None = None

    @property
    @abstractmethod
    def num_layers(self) -> int:
        pass

    @property
    def num_schedule_steps(self) -> int:
        return self.config.num_schedule_steps

    @abstractmethod
    def prepare_prompt_batch(self, prompts: Iterable[str]):
        pass

    @abstractmethod
    def attach_cache(self, schedule: DLMCacheSchedule) -> None:
        pass

    @abstractmethod
    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        pass

    def _validate_schedule(self, schedule: DLMCacheSchedule) -> None:
        if schedule.num_layers != self.num_layers:
            raise ValueError(
                f"{self.family} has {self.num_layers} layers but the schedule has "
                f"{schedule.num_layers}"
            )
        if schedule.num_inference_steps != self.num_schedule_steps:
            raise ValueError(
                f"{self.family} requires {self.num_schedule_steps} schedule steps but "
                f"the schedule has {schedule.num_inference_steps}"
            )
        if schedule.schedule_domain != self.schedule_domain:
            raise ValueError(
                f"{self.family} requires schedule_domain={self.schedule_domain!r}, "
                f"got {schedule.schedule_domain!r}"
            )

    def set_schedule(self, schedule: DLMCacheSchedule) -> None:
        self._validate_schedule(schedule)
        if self.controller is None:
            raise RuntimeError("attach_cache() must be called before set_schedule()")
        self.controller.set_schedule(schedule)

    @property
    def cache_runtime(self):
        if self.controller is None:
            raise RuntimeError("attach_cache() must be called before generation")
        return self.controller.runtime

    def decode_generated(self, generated: torch.Tensor) -> list[str]:
        responses = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        # Dream tokenizers normally classify this as special. The explicit split
        # keeps decoding safe with older remote-code tokenizer revisions too.
        return [response.split("<|dlm_pad|>", 1)[0] for response in responses]

    def uninstall(self) -> None:
        if self.controller is not None:
            self.controller.uninstall()
            self.controller = None


class LLaDABackend(DLMBackend):
    family = "llada"
    schedule_domain = "local_step_repeated_per_block"

    @property
    def num_layers(self) -> int:
        return int(self.model.config.n_layers)

    def prepare_prompt_batch(self, prompts: Iterable[str]):
        return prepare_prompt_batch(self.tokenizer, prompts, self.config.model.device)

    def attach_cache(self, schedule: DLMCacheSchedule) -> None:
        self._validate_schedule(schedule)
        self.controller = attach_llada_residual_cache(self.model, schedule)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        generation = self.config.generation
        return generate_llada(
            self.model,
            input_ids,
            attention_mask=attention_mask,
            steps=generation.steps,
            gen_length=generation.gen_length,
            block_length=generation.block_length,
            temperature=generation.temperature,
            cfg_scale=generation.cfg_scale,
            remasking=generation.remasking,
            mask_id=generation.mask_id,
            eos_token_id=generation.eos_token_id,
            eot_token_id=generation.eot_token_id,
            logits_eos_inf=generation.logits_eos_inf,
            confidence_eos_eot_inf=generation.confidence_eos_eot_inf,
            cache_runtime=self.cache_runtime,
        )


class DiffuCoderBackend(DLMBackend):
    family = "diffucoder"
    schedule_domain = "global_denoising_step"

    @property
    def num_layers(self) -> int:
        return int(self.model.config.num_hidden_layers)

    def prepare_prompt_batch(self, prompts: Iterable[str]):
        return prepare_diffucoder_prompt_batch(
            self.tokenizer, prompts, self.config.model.device
        )

    def attach_cache(self, schedule: DLMCacheSchedule) -> None:
        self._validate_schedule(schedule)
        self.controller = attach_diffucoder_residual_cache(self.model, schedule)

    def generate(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None):
        generation = self.config.generation
        return generate_diffucoder(
            self.model,
            input_ids,
            attention_mask=attention_mask,
            steps=generation.steps,
            gen_length=generation.gen_length,
            temperature=generation.temperature,
            top_p=generation.top_p,
            top_k=generation.top_k,
            algorithm=generation.algorithm,
            algorithm_temperature=generation.algorithm_temperature,
            cache_runtime=self.cache_runtime,
        )


def load_dlm_backend(config: DLMResidualCacheExperimentConfig) -> DLMBackend:
    """Load the model selected by ``model.family`` and return its ECAD adapter."""

    settings = config.model
    loader_kwargs = {
        "device": settings.device,
        "dtype": settings.dtype,
        "local_files_only": settings.local_files_only,
    }
    if settings.family == "llada":
        model, tokenizer = load_llada_model_and_tokenizer(
            settings.path, **loader_kwargs
        )
        return LLaDABackend(model, tokenizer, config)
    if settings.family == "diffucoder":
        model, tokenizer = load_diffucoder_model_and_tokenizer(
            settings.path, **loader_kwargs
        )
        return DiffuCoderBackend(model, tokenizer, config)
    raise ValueError(f"Unsupported DLM model family {settings.family!r}")
