import gc
import time
from abc import ABC
from pathlib import Path
from typing import Any, Dict, Sequence, Type

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedModel, AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ecad.image_generators.text_generator import TextGenerator  # your ABC from earlier
from ecad.schedulers.cache_scheduler.dream_cache_schedule import Dream7bCacheSchedule
from ecad.schedulers.dit_scheduler.dit_schedule import DiTSchedule
from ecad.schedulers.dit_scheduler.generators.pixart_schedule_generators import (
    gen_default as dit_gen_default,  # swap for DLM default generator when you have it
)
from ecad.schedulers.cache_scheduler.generators.pixart_schedule_generators import (
    gen_default as cache_gen_default,  # swap for DLM default generator when you have it
)

# Reuse your existing embedding dataset pipeline (it expects per-prompt .pt dicts)
from ecad.types import TextGeneratorConfig, DreamPromptEmbedding, PromptEmbeddingType

from ecad.lm_models.dream.modeling_dream import DreamForCausalLM
from ecad.lm_models.dream_flash.modeling_dream import DreamForCausalLM as DreamFlashForCausalLM

from ecad.lm_models.dream_flash.generation_utils import DreamGenerationConfig

from ecad.lm_models.dream.configuration_dream import ODreamConfig
from ecad.lm_models.dream_flash.configuration_dream import ODreamConfig as DreamFlashConfig


# register the Dream model
AutoConfig.register("odream", ODreamConfig)
AutoModelForCausalLM.register(ODreamConfig, DreamForCausalLM)

# register the Dream Flash model
AutoConfig.register("odream_flash", DreamFlashConfig)
AutoModelForCausalLM.register(DreamFlashConfig, DreamFlashForCausalLM)



def load_tokenizer(name: str):
    canon = (name.replace("-Custom","")
                 .replace("-Opt","")
                 .replace("-Block-Cached","")
                 .replace("-Flash","-v0")
                 .replace("-v2","-v0")
                 .replace("-v3","-v0"))
    tok = AutoTokenizer.from_pretrained(canon, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id or 0
    return tok


def make_default_recompute_all(num_layers: int, num_inference_steps: int) -> Dream7bCacheSchedule:
    schedule = {}
    for step in range(num_inference_steps):
        schedule[step] = {
            str(layer): {"layer": True, "kv": True, "mlp": True}
            for layer in range(num_layers)
        }
    return Dream7bCacheSchedule(
        num_blocks=num_layers,
        num_inference_steps=num_inference_steps,
        name="default_recompute_all",
        schedule=schedule,
        attributes={},
    )


class DreamTextGenerator(TextGenerator, ABC):
    """
    Generic diffusion-language-model text generator that mirrors the PixArtImageGenerator structure:

      - Loads schedules + callbacks via TextGenerator base
      - Provides default schedules (until you add DLM-specific schedule generators)
      - Creates an "encoder pipeline" (tokenizer / conditioning prep)
      - Creates a "generation pipeline" by loading an edited DLM transformer via .from_pretrained
        and injecting dit_scheduler + cache_schedule
      - Encodes prompts to CPU tensors and saves as .pt for offline evaluation
      - Generates text stepwise with per-step callbacks and decodes outputs
      - Provides a timed generation path

    This is intentionally model-agnostic: concrete DLMs (dream/llada/others) should share:
      - from_pretrained(...) that accepts (dit_scheduler, cache_schedule, ...)
      - generate(...) or diffusion_generate(...) that supports per-step callback
      - reset_cache() on the underlying transformer (optional but recommended)
    """

    # Override these in concrete subclasses if desired
    DEFAULT_WEIGHTS = "Dream-org/Dream-Flash-Instruct-7B"
    DEFAULT_TOKENIZER_NAME = None

    # Your edited DLM transformer class, injected by subclasses/constructor
    # Must expose: .from_pretrained(...)
    DLM_MODEL_CLS: Type[PreTrainedModel]

    def __init__(
        self,
        *,
        weights_name: str | None = None,
        start_seed: int = 0,
        seed_step: int = 1,
        schedule_path: Path | None = None,
        device: str = "cuda",
    ):
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise ValueError("CUDA requested but not available.")

        self.tokenizer = load_tokenizer(self.DEFAULT_WEIGHTS)

        self.weights_name = weights_name or self.DEFAULT_WEIGHTS


        super().__init__(
            default_transformer_weights=self.weights_name,
            default_pipeline_weights=None,
            default_pipeline_name=None,
            schedule_path=schedule_path,
            start_seed=start_seed,
            seed_step=seed_step,
            device=device,
            additional_callbacks=None,
            dit_schedule_type=DiTSchedule,      # swap to DLM-specific schedule types if you have them
            cache_schedule_type=Dream7bCacheSchedule,  # swap to DLM-specific schedule types if you have them
        )

        # Optional: diffusion LMs sometimes need a "null" conditioning for CFG-like guidance.
        # Keep as placeholders; concrete models can set/use these.
        self.null_condition: Any | None = None

        self.generation_pipeline: PreTrainedModel | None = None
        self.encoder_pipeline: Any | None = None  # typically just the tokenizer
    # ---------------------------------------------------------------------
    # Config + default schedules
    # ---------------------------------------------------------------------

    def _load_subclass_config_defaults(self, config: TextGeneratorConfig) -> None:
        # Keep defaults aligned with your YAML/JSON style config under "dream" or similar.
        # You can add more keys later without changing the GA harness.
        self.max_new_tokens_default: int = int(config.get("max_new_tokens", 128))
        self.temperature_default: float = float(config.get("temperature", 0.2))
        self.top_p_default: float = float(config.get("top_p", 0.95))
        self.sampling_strategy_default: str = str(config.get("sampling_strategy", "deterministic"))
        self.early_stop_default: bool = bool(config.get("early_stop", True))
        self.early_stop_consecutive_default: int = int(config.get("early_stop_consecutive", 1))
        self.stop_on_dream_eos_default: bool = bool(config.get("stop_on_dream_eos", True))

    def _default_dit_schedule(self) -> DiTSchedule:
        # Placeholder: replace with your DLM schedule generator when available.
        # The PixArt default uses (num_blocks=28, num_steps=20).
        return next(dit_gen_default(28, 20))

    def _default_cache_schedule(self) -> Dream7bCacheSchedule:
        # Placeholder: replace with your DLM cache schedule generator when available.
        return make_default_recompute_all(num_layers=32, num_inference_steps=20)

    # ---------------------------------------------------------------------
    # Pipelines
    # ---------------------------------------------------------------------

    def create_encoder_pipeline(self) -> None:
        """
        Encoder pipeline for DLM:
          - ensure tokenizer exists
          - optionally precompute a null condition if your model uses CFG-like guidance
        """
        # if self._tokenizer is None:
        #     if self._tokenizer_name is None:
        #         raise ValueError("Provide tokenizer or tokenizer_name.")
        #     from transformers import AutoTokenizer

        #     self._tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_name, use_fast=True)

        # Ensure required special tokens exist for diffusion-LM style configs
        # (mask token is common; if absent you must decide on a strategy per model)
        if getattr(self.tokenizer, "mask_token_id", None) is None:
            # Not all tokenizers have a mask token; allow it but warn.
            print("WARNING: tokenizer has no mask_token_id; diffusion masking may require model-specific handling.")

        self.encoder_pipeline = True  # sentinel: we only need tokenizer here

    def create_generation_pipeline(self) -> None:
        """
        Generation pipeline mirrors PixArt's create_diffusion_pipeline:

          model = DLM_MODEL_CLS.from_pretrained(
              self.transformer_weights,
              torch_dtype=...,
              dit_scheduler=self.dit_scheduler,
              cache_schedule=self.cache_schedule,
              ...
          ).to(self.device)
        """
        print(f"Creating DLM transformer with weights {self.weights_name}.")

        if "-Flash" in self.weights_name:
            self.weights_name = self.weights_name.replace("-Flash", "-v0")
            config = DreamFlashConfig().from_pretrained(self.weights_name)
        else:
            config = ODreamConfig().from_pretrained(self.weights_name)
        # Keep kwargs minimal and aligned with the PixArt pattern.
        # model = self.DLM_MODEL_CLS.from_pretrained(
        #     self.weights_name,
        #     torch_dtype=torch.float16 if self.device.type == "cuda" else None,
        #     dit_scheduler=self.dit_scheduler,
        #     cache_schedule=self.cache_schedule,
        # )

        model = AutoModelForCausalLM.from_pretrained(
            self.weights_name,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            dit_scheduler=self.dit_scheduler,
            cache_schedule=self.cache_schedule,
        )
        # raise NotImplementedError("Testing model instantiation; remove when ready")

        if hasattr(model, "to"):
            model = model.to(self.device)

        # Some backends need eval mode for inference
        if hasattr(model, "eval"):
            model.eval()

        self.generation_pipeline = model

    # ---------------------------------------------------------------------
    # Encoding + saving prompts (offline eval)
    # ---------------------------------------------------------------------

    # @property
    # def tokenizer(self) -> PreTrainedTokenizerBase:
    #     if self._tokenizer is None:
    #         self.create_encoder_pipeline()
    #     assert self._tokenizer is not None
    #     return self._tokenizer

    @torch.inference_mode()
    def encode_prompts(
        self,
        prompts: list[str],
        negative_prompts: list[str] | None = None,
        batch_size: int | None = None,
        **kwargs: Any,
    ) -> Sequence[DreamPromptEmbedding]:
        """
        Encode prompts into per-prompt CPU tensors.
        IMPORTANT: avoid torch.cat across batches because padding length varies per batch.
        """
        if negative_prompts is not None:
            print("WARNING: negative_prompts not used in generic DLMTextGenerator; ignoring.")

        tok = self.tokenizer
        if batch_size is None:
            batch_size = len(prompts)

        embedded: list[DreamPromptEmbedding] = []

        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i : i + batch_size]
            enc = tok(
                batch_prompts,
                return_tensors="pt",
                padding=True,        # pads within this batch only (fine now)
                truncation=False,
            )

            input_ids = enc["input_ids"].to("cpu")
            attention_mask = enc["attention_mask"].to("cpu")

            # Add one embedding dict per prompt in this batch.
            for j in range(input_ids.shape[0]):
                embedded.append(
                    {
                        "input_ids": input_ids[j].clone(),
                        "attention_mask": attention_mask[j].clone(),
                    }
                )

            del enc, input_ids, attention_mask
            gc.collect()

        return embedded


    @torch.inference_mode()
    def encode_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ) -> None:
        if not name_to_prompt:
            print("No prompts found.")
            return

        # Normalize mapping: name -> prompt string
        if isinstance(next(iter(name_to_prompt.values())), dict):
            name_to_prompt = {name: v["prompt"] for name, v in name_to_prompt.items()}  # type: ignore

        if self.encoder_pipeline is None:
            print("Creating encoder pipeline.")
            self.create_encoder_pipeline()

        prompts = list(name_to_prompt.values())
        print("Encoding prompts.")
        embedded_prompts = self.encode_prompts(prompts, batch_size=batch_size)

        if free_after:
            print("Freeing encoder pipeline.")
            self.free_encoder_pipeline()

        output_dir.mkdir(parents=True, exist_ok=True)
        for name, emb in zip(name_to_prompt.keys(), embedded_prompts):
            output_fname = Path(name)
            if output_fname.suffix != ".pt":
                output_fname = Path(f"{output_fname}.pt")
            output_path = output_dir / output_fname
            print(f"Saving prompt to '{output_path}'.")
            torch.save(emb, output_path)


    @torch.inference_mode()
    def tokenize_and_save_prompts(
        self,
        name_to_prompt: dict[str, dict[str, str]] | dict[str, str],
        output_dir: Path,
        free_after: bool = False,
        batch_size: int | None = None,
    ) -> None:
        if not name_to_prompt:
            print("No prompts found.")
            return

        # Normalize mapping: name -> prompt string
        if isinstance(next(iter(name_to_prompt.values())), dict):
            name_to_prompt = {name: v["prompt"] for name, v in name_to_prompt.items()}  # type: ignore

        if getattr(self, "encoder_pipeline", None) is None and hasattr(self, "create_encoder_pipeline"):
            print("Creating encoder pipeline.")
            self.create_encoder_pipeline()

        if getattr(self, "tokenizer", None) is None:
            raise ValueError("self.tokenizer is not set. Cannot tokenize prompts.")

        prompts = list(name_to_prompt.values())
        names = list(name_to_prompt.keys())

        print("Tokenizing prompts (via encode_prompts).")
        tokenized_prompts = self.encode_prompts(prompts, batch_size=batch_size)

        if free_after and hasattr(self, "free_encoder_pipeline"):
            print("Freeing encoder pipeline.")
            self.free_encoder_pipeline()

        output_dir.mkdir(parents=True, exist_ok=True)

        for name, prompt, toks in zip(names, prompts, tokenized_prompts):
            output_fname = Path(name)
            if output_fname.suffix != ".pt":
                output_fname = Path(f"{output_fname}.pt")
            output_path = output_dir / output_fname

            payload: Dict[str, Any] = {
                "input_ids": toks["input_ids"],           # CPU tensor
                "attention_mask": toks["attention_mask"], # CPU tensor
                "name": name,
                "prompt": prompt,
            }

            # Optional: trim right-padding to reduce file size.
            attn = payload["attention_mask"]
            if attn.numel() > 0 and attn.any():
                last_one = int(attn.nonzero(as_tuple=False)[-1].item())
                payload["input_ids"] = payload["input_ids"][: last_one + 1].contiguous()
                payload["attention_mask"] = payload["attention_mask"][: last_one + 1].contiguous()

            print(f"Saving tokenized prompt to '{output_path}'.")
            torch.save(payload, output_path)

    # ---------------------------------------------------------------------
    # Generation (mirrors your loop: encode -> build cfg -> call model -> decode)
    # ---------------------------------------------------------------------

    def _build_generation_config(
        self,
        *,
        input_len: int,
        max_new_tokens: int,
        **overrides: Any,
    ) -> Any:
        """
        Mirrors the snippet's per-sample config assembly.

        We intentionally keep this generic: your dream/llada configs are interchangeable.
        Expect the actual model's generate function to accept this config object (or dict).
        """
        tok = self.tokenizer

        # Pull defaults from loaded top-level config, but allow overrides.
        temperature = overrides.get("temperature", self.temperature_default)
        top_p = overrides.get("top_p", self.top_p_default)
        sampling_strategy = overrides.get("sampling_strategy", self.sampling_strategy_default)

        cfg = {
            "max_length": input_len + max_new_tokens,
            "max_new_tokens": max_new_tokens,
            "mask_token_id": getattr(tok, "mask_token_id", None),
            "eos_token_id": tok.eos_token_id,
            "early_stop": overrides.get("early_stop", self.early_stop_default),
            "early_stop_consecutive": overrides.get(
                "early_stop_consecutive", self.early_stop_consecutive_default
            ),
            "temperature": temperature,
            "top_p": top_p,
            "sampling_strategy": sampling_strategy,
            "stop_on_dream_eos": overrides.get(
                "stop_on_dream_eos", self.stop_on_dream_eos_default
            ),
            "num_inference_steps": overrides.get("num_inference_steps", self.num_inference_steps),
            "return_dict_in_generate": True,
            "save_cache": True,
            "use_full_query_attn": overrides.get('use_full_query_attn', False),

            'use_block_diffusion': overrides.get('use_block_diffusion', True),
            'block_size': overrides.get('block_size', 16),
            'alg':'entropy'
        }

        print('\nmax new tokens:', cfg["max_new_tokens"], 'block size:', cfg['block_size'], 'alg:', cfg['alg'],'\n')

        gen_cfg = DreamGenerationConfig(**cfg)

        return gen_cfg

    @torch.inference_mode()
    def generate_text(
        self,
        prompt_tokens: Dict[str, Any],
        generations_per_prompt: int,
        max_new_tokens: int | None = None,
        **kwargs: Any,
    ) -> list[list[str]]:
        """
        Generate text from saved prompt "embeddings" (token tensors).

        Returns:
            list of length batch_size, each element is a list[str] of length generations_per_prompt.
        """
        if self.generation_pipeline is None:
            print("Creating generation pipeline.")
            self.create_generation_pipeline()

        model = self.generation_pipeline
        tok = self.tokenizer

        # prompt_embeds comes from PromptEmbeddingDataset batching.
        # It will typically be shape [B, L] if collated; handle both [L] and [B, L].
        input_ids = prompt_tokens["input_ids"]
        attention_mask = prompt_tokens.get("attention_mask", None)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            mask = (attention_mask == 0)  # pad positions
            # attn_mask = mask.to(dtype=torch.float16) * -1e4  # or -65504 for fp16

        B, L = input_ids.shape
        
        max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens_default

        # B, S = attention_mask.shape
        # L = S  # for self-attn; if decoding with different query length, use that L
        # pad = (attention_mask == 0)                      # (B, S) True=pad
        # attn_mask = pad[:, None, None, :].expand(B, 1, L, S)  # (B,1,L,S)
        
        # Build cfg once per batch (like your snippet, but batch-aware).
        cfg = self._build_generation_config(input_len=L, max_new_tokens=max_new_tokens, **kwargs)

        # Generate sequentially across generations_per_prompt to mimic PixArt's stable seeding pattern.
        all_out: list[list[str]] = [[] for _ in range(B)]

        for k in range(generations_per_prompt):
            self.random_generator.manual_seed(self.start_seed + k * self.seed_step)

            # IMPORTANT:
            # Your DLM model should expose a generation entrypoint that accepts:
            #   - input_ids, attention_mask
            #   - config (dict or config object)
            #   - generator/seed
            #   - callback called each diffusion step
            #
            # We keep this generic by checking for common method names.
            
            # out = model(
            #         input_ids=input_ids,
            #         attention_mask=attn_mask,
            #         position_ids = None,
            #         #use_cache = cfg.use_cache,
            #         return_dict = cfg.return_dict_in_generate,
            #         save_cache = cfg.save_cache, 
            #         max_length=cfg.max_length,
            #         use_block_diffusion = cfg.use_block_diffusion,
            #         block_size = cfg.block_size,
            #         use_full_query_attn = cfg.use_full_query_attn,
            #         clean_idx = 0
            #     )
            
            out = model.diffusion_generate(
                inputs=input_ids,
                attention_mask=attention_mask,
                generation_config=cfg,
                generator=self.random_generator,
                callback=self._call_callbacks,
                callback_steps=1,
                )

            # Model config and generation config NOT the same!

            # Normalize outputs to token ids
            seqs = None
            if hasattr(out, "sequences"):
                seqs = out.sequences
            elif isinstance(out, dict) and "logits" in out:
                seqs = out["logits"]
            elif torch.is_tensor(out):
                seqs = out
            else:
                raise ValueError("Unrecognized output format from model generation.")

            # might need to slice to only include answer (seqs[:, L:])
            decoded = tok.batch_decode(seqs, skip_special_tokens=True)
            for i, s in enumerate(decoded):
                all_out[i].append(s)

            # safety: reset caches/schedules between generations if needed
            transformer = getattr(model, "transformer", None)
            if transformer is not None and hasattr(transformer, "reset_cache"):
                transformer.reset_cache()

        return all_out

    @torch.inference_mode()
    def generate_text_timed(self, prompt_embeds: DreamPromptEmbedding, **kwargs: Any) -> float:
        """
        Time one generation pass and return ms per sample (batch-normalized),
        mirroring PixArt's generate_images_timed.
        """
        if self.generation_pipeline is None:
            print("Creating generation pipeline.")
            self.create_generation_pipeline()

        model = self.generation_pipeline
        tok = self.tokenizer

        input_ids = prompt_embeds["input_ids"]
        attention_mask = prompt_embeds.get("attention_mask", None)

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

        input_ids = input_ids.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        B, L = input_ids.shape
        max_new_tokens = int(kwargs.get("max_new_tokens", self.max_new_tokens_default))
        cfg = self._build_generation_config(input_len=L, max_new_tokens=max_new_tokens, **kwargs)

        self.random_generator.manual_seed(self.start_seed)

        torch.cuda.synchronize() if self.device.type == "cuda" else None
        t0 = time.perf_counter()

        _ = model.diffusion_generate(
            inputs=input_ids,
            attention_mask=attention_mask,
            generation_config=cfg,
            generator=self.random_generator,
            callback=self._call_callbacks,
            callback_steps=1,
        )
        
        torch.cuda.synchronize() if self.device.type == "cuda" else None
        ms = (time.perf_counter() - t0) * 1000.0
        return ms / B
