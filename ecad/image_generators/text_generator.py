from __future__ import annotations

from abc import ABC, abstractmethod
import gc
from pathlib import Path
from typing import Any, Callable, Sequence, Generic, TypeVar

import torch
from torch.utils.data import DataLoader

# These are "text analogs" of your image-side abstractions.
# You'll implement them in your text stack similarly to:
#   - DiTSchedule / CacheSchedule
#   - DiTScheduler
#   - PromptEmbeddingDataset
#
# If you already have equivalents, just swap the imports/types.
from ecad.schedulers.cache_scheduler.cache_schedule import CacheSchedule
from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.schedulers.dit_scheduler.dit_scheduler import DiTScheduler

from ecad.dataset_utils.prompt_embedding_dataset import PromptEmbeddingDataset, GSM8KTokenizedPtDataset, pad_tokenized_batch

from ecad.types import (
    TextGeneratorConfig,   # analogous to ImageGeneratorConfig
    PromptEmbedding,       # can be reused if it's just tensors + metadata
    PromptEmbeddingType,   # e.g. dict[str, torch.Tensor] batch format
)

from transformers import PreTrainedTokenizerBase

PE = TypeVar("PE", bound=PromptEmbedding)

class TextGenerator(ABC, Generic[PE]):
    """
    Base class for text generators. Mirrors the ImageGenerator surface area
    that the GA / offline-eval loop actually relies on.

    Intended for diffusion-style language models, but also works for any
    iterative generator that can expose per-step callbacks.
    """

    def __init__(
        self,
        default_transformer_weights: str = None,
        default_pipeline_weights: str = None,
        default_pipeline_name: str = None,
        schedule_path: Path | None = None,
        start_seed: int = 0,
        seed_step: int = 1,
        device: str = "cuda",
        additional_callbacks: list[Callable] | None = None,
        dit_schedule_type: type[DiTSchedule] = None,
        cache_schedule_type: type[CacheSchedule] = None,
    ):
        self.device: torch.device = torch.device(device)
        self.default_transformer_weights: str = default_transformer_weights
        self.default_pipeline_weights: str = default_pipeline_weights
        self.default_pipeline_name: str = default_pipeline_name
        self.schedule_path: Path | None = schedule_path
        self.start_seed: int = start_seed
        self.seed_step: int = seed_step
        self.additional_callbacks: list[Callable] = additional_callbacks or []

        self.dit_schedule_type: type[DiTSchedule] = dit_schedule_type
        self.cache_schedule_type: type[CacheSchedule] = cache_schedule_type

        # Subclasses can use these however they want (HF pipeline, custom wrapper, etc.)
        self.encoder_pipeline: Any | None = None
        self.generation_pipeline: Any | None = None

        # set by _load_schedule_file / _load_config
        self.random_generator: torch.Generator
        self.dit_scheduler: DiTScheduler
        self.cache_schedule: CacheSchedule
        self.num_inference_steps: int
        self.transformer_weights: str
        self.pipeline_weights: str

        self._initialize_random_generator()
        self._load_schedule_file()

    @property
    @abstractmethod
    def _tokenizer(self) -> PreTrainedTokenizerBase:
        """
        The tokenizer used by this text generator.
        """
        raise NotImplementedError("Must be provided by subclass.")

    def _initialize_random_generator(self) -> None:
        # always use CPU generator to ensure reproducibility
        self.random_generator = torch.Generator(device="cpu")
        self.random_generator.manual_seed(self.start_seed)
        print(
            f"Using random start seed: {self.start_seed}. "
            f"Stepping seed by {self.seed_step} for each successive generation per prompt."
        )

    def _load_schedule_file(self) -> None:
        """
        Load schedules for iterative generation.

        We reuse the same DiT + cache schedule mechanism because your diffusion-LM
        presumably exposes (step, timestep, **kwargs) similarly.
        """
        dit_was_none = True
        cache_was_none = True

        if self.schedule_path is None:
            dit_schedule = None
            cache_schedule = None
        else:
            try:
                dit_schedule = self.dit_schedule_type.from_json(self.schedule_path)
                dit_was_none = False
            except KeyError:
                dit_schedule = None

            try:
                cache_schedule = self.cache_schedule_type.from_json(self.schedule_path)
                cache_was_none = False
            except KeyError:
                cache_schedule = None

        if dit_schedule is None:
            dit_schedule = self._default_dit_schedule()
        if cache_schedule is None:
            cache_schedule = self._default_cache_schedule()

        self.dit_scheduler = DiTScheduler(dit_schedule)
        self.cache_schedule = cache_schedule
        print(f"Using DiT schedule: {dit_schedule.name}.")
        print(f"Using cache schedule: {self.cache_schedule.name}.")

        if dit_was_none and cache_was_none:
            dit_n_steps = self.dit_scheduler.schedule.num_inference_steps
            cache_n_steps = self.cache_schedule.num_inference_steps
            if dit_n_steps != cache_n_steps:
                raise ValueError(
                    "DiT and cache schedules have different numbers of inference steps; should be impossible."
                )
            self.num_inference_steps = dit_n_steps
        if dit_was_none and not cache_was_none:
            self.num_inference_steps = self.cache_schedule.num_inference_steps
        if not dit_was_none and cache_was_none:
            self.num_inference_steps = self.dit_scheduler.schedule.num_inference_steps
        if not dit_was_none and not cache_was_none:
            # dit_n_steps = self.dit_scheduler.schedule.num_inference_steps
            cache_n_steps = self.cache_schedule.num_inference_steps
            # if dit_n_steps != cache_n_steps:
            #     raise ValueError("DiT and cache schedules have different numbers of inference steps.")
            self.num_inference_steps = cache_n_steps

        print(f"Using {self.num_inference_steps} inference steps.")

        self.callbacks = [
            self.dit_scheduler.per_step_callback,
            self.cache_schedule.per_step_callback,
        ]
        self.callbacks.extend(self.additional_callbacks)
        # IMPORTANT: reset MUST be LAST in the list
        self.callbacks.append(self._reset_schedules_callback)

        dit_config = dit_schedule.top_level_config
        cache_config = cache_schedule.top_level_config
        if dit_config and cache_config and dit_config != cache_config:
            raise ValueError(
                "DiT and cache schedules have different top level configs; should be impossible."
            )
        config: TextGeneratorConfig = (dit_config if dit_config else cache_config)  # type: ignore
        self._load_config(config)

    def _load_config(self, config: TextGeneratorConfig) -> None:
        self.config: TextGeneratorConfig = config
        self.transformer_weights = config.get("transformer_weights", self.default_transformer_weights)
        self.pipeline_weights = config.get("pipeline_weights", self.default_pipeline_weights)
        self._load_subclass_config_defaults(config)

    def _load_subclass_config_defaults(self, config: TextGeneratorConfig) -> None:
        # Optional: subclasses can read extra fields from config here.
        pass

    def _reset_schedules_callback(self, step: int, timestep: int, **kwargs: Any) -> None:
        # callback is called at the end of each inference step
        if step >= self.num_inference_steps - 1:
            self.dit_scheduler.reset_step()
            self.cache_schedule.reset_step()

            # If your generation pipeline has an internal transformer with a cache:
            if self.generation_pipeline is not None:
                transformer = getattr(self.generation_pipeline, "transformer", None)
                if transformer is not None and hasattr(transformer, "reset_cache"):
                    transformer.reset_cache()

    def _call_callbacks(self, step: int, timestep: int, **kwargs: Any) -> None:
        for callback in self.callbacks:
            callback(step, timestep, **kwargs)

    # ---- required schedule defaults (mirrors ImageGenerator) ----

    @abstractmethod
    def _default_dit_schedule(self) -> DiTSchedule:
        raise NotImplementedError("Subclasses must implement _default_dit_schedule")

    @abstractmethod
    def _default_cache_schedule(self) -> CacheSchedule:
        raise NotImplementedError("Subclasses must implement _default_cache_schedule")

    # ---- lifecycle: pipelines (mirror only what offline eval likely needs) ----

    @abstractmethod
    def create_encoder_pipeline(self) -> None:
        """
        Create the text encoder pipeline (tokenizer / embedder / conditioner).
        """
        raise NotImplementedError("Subclasses must implement create_encoder_pipeline")

    @abstractmethod
    def create_generation_pipeline(self) -> None:
        """
        Create the diffusion/iterative text generation pipeline.
        """
        raise NotImplementedError("Subclasses must implement create_generation_pipeline")

    def free_encoder_pipeline(self) -> None:
        if not hasattr(self, "encoder_pipeline"):
            raise AttributeError("encoder_pipeline not setup by subclass.")

        if self.encoder_pipeline is not None:
            del self.encoder_pipeline
            self.encoder_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("WARNING: No encoder pipeline to free.")

    def free_generation_pipeline(self) -> None:
        if not hasattr(self, "generation_pipeline"):
            raise AttributeError("generation_pipeline not setup by subclass.")

        if self.generation_pipeline is not None:
            del self.generation_pipeline
            self.generation_pipeline = None
            torch.cuda.empty_cache()
            gc.collect()
        else:
            print("WARNING: No generation pipeline to free.")

    # ---- prompt embedding IO (kept because your offline eval is embedding-driven) ----

    def load_and_batch_embeddings(
        self, embedding_dir: Path, batch_size: int, shuffle: bool = False
    ) -> DataLoader:
        dataset = PromptEmbeddingDataset(embedding_dir)
        return DataLoader(
            dataset,
            batch_size,
            shuffle,
            pin_memory=True,
            pin_memory_device=str(self.device),
        )
    

    def load_and_batch_tokenized_prompts(
            self, input_dir: Path, batch_size: int, shuffle: bool = False
        ) -> DataLoader:
        ds = GSM8KTokenizedPtDataset(root_dir=input_dir, split="train")
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=2,
            collate_fn=lambda b: pad_tokenized_batch(b, pad_token_id=self._tokenizer.pad_token_id),
        )
        return dl 
        

    @abstractmethod
    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompts: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Sequence[PromptEmbedding]:
        """
        Encode prompts (and optional negative prompts) into prompt embeddings.
        """
        raise NotImplementedError("Subclasses must implement encode_prompts")

    @abstractmethod
    def encode_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """
        Encode prompts and save embeddings to disk for offline evaluation runs.
        """
        raise NotImplementedError("Subclasses must implement encode_and_save_prompts")

    @abstractmethod
    def tokenize_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ) -> None:
        """
        Tokenize prompts and save tokens to disk for offline evaluation runs.
        """
        raise NotImplementedError("Subclasses must implement tokenize_and_save_prompts")


    # ---- generation (the main thing offline eval calls) ----

    @abstractmethod
    def generate_text(
        self,
        prompt_embeds: PE,
        generations_per_prompt: int,
        **kwargs: Any,
    ) -> list[list[str]]:
        """
        Generate text from prompt embeddings.

        Returns:
            A nested list shaped like:
                [batch_size][generations_per_prompt]
        """
        raise NotImplementedError("Subclasses must implement generate_text")

    @torch.inference_mode()
    def generate_from_saved_prompts(
        self,
        input_dir: Path,
        output_dir: Path,
        batch_size: int = 1,
        generations_per_prompt: int = 1,
        free_after: bool = False,
        include_seed_in_name: bool = True,
        **kwargs: Any,
    ) -> None:
        """
        Offline-eval entrypoint: load saved embeddings and write generations to disk.
        Mirrors `ImageGenerator.generate_from_saved_prompts`.
        """
        if self.generation_pipeline is None:
            print("Creating generation pipeline.")
            self.create_generation_pipeline()

        print("Loading and batching tokens.")
        token_loader = self.load_and_batch_tokenized_prompts(input_dir, batch_size, False)

        for batch_idx, tokens in enumerate(token_loader):
            print(f"Running batch {batch_idx} of {len(token_loader) - 1}.")
            generations = self.generate_text(
                tokens,
                generations_per_prompt,
                **kwargs,
            )

            for name, rel_path, gens_for_one_prompt in zip(
                tokens["name"], tokens["relative_path"], generations
            ):
                print(f"Saving generations for prompt '{name}'.")

                for i, text in enumerate(gens_for_one_prompt):
                    gen_seed = self.start_seed + i * self.seed_step
                    name_with_seed = (
                        f"{name}__gen_seed:{gen_seed:03}"
                        if include_seed_in_name
                        else name
                    )
                    output_path = output_dir / rel_path / name_with_seed
                    self.save_text(text, output_path)

            # free up some memory
            del tokens
            del generations
            gc.collect()
            torch.cuda.empty_cache()

        if free_after:
            print("Freeing generation pipeline.")
            self.free_generation_pipeline()

    def save_text(self, text: str, output_path: Path) -> None:
        """
        Save a single generated string to disk.

        Mirrors `save_image` behavior: if no extension, add `.txt`.
        """
        output_path = Path(output_path)
        if output_path.suffix == "":
            output_path = Path(f"{output_path}.txt")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving text to '{output_path}'.")
        output_path.write_text(text, encoding="utf-8")

    @torch.inference_mode()
    def time_text_generation(
        self,
        input_dir: Path,
        batch_size: int = 1,
        num_batches: int = 1,
        free_after: bool = False,
        **kwargs: Any,
    ) -> list[float]:
        """
        Mirrors `time_image_generation`: measures time per batch.
        Subclasses provide the timed kernel in `generate_text_timed`.
        """
        if self.generation_pipeline is None:
            print("Creating generation pipeline.")
            self.create_generation_pipeline()

        all_times: list[float] = []
        batches_run = 0

        while batches_run < num_batches:
            print("Loading and batching embeddings.")
            embedding_loader = self.load_and_batch_embeddings(input_dir, batch_size, False)

            for batch_idx, embeds in enumerate(embedding_loader):
                print(f"Running batch {batch_idx} of {len(embedding_loader) - 1}.")

                t_ms = self.generate_text_timed(embeds, **kwargs)
                all_times.append(t_ms)
                print(f"Time for batch {batch_idx}: {t_ms:.2f} ms.")

                del embeds
                gc.collect()
                torch.cuda.empty_cache()

                batches_run += 1
                if batches_run >= num_batches:
                    break

        if free_after:
            print("Freeing generation pipeline.")
            self.free_generation_pipeline()

        return all_times

    @abstractmethod
    def generate_text_timed(
        self,
        prompt_embeds: PE,
        **kwargs: Any,
    ) -> float:
        """
        Run one batch generation and return elapsed time in milliseconds.
        """
        raise NotImplementedError
