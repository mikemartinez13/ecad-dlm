"""Loading and prompt preparation helpers for DiffuCoder checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

from ecad.language_models.llada_loader import DTYPES


def load_diffucoder_model_and_tokenizer(
    model_path: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    local_files_only: bool = True,
):
    from transformers import AutoModel, AutoTokenizer

    if dtype not in DTYPES:
        raise ValueError(f"Unsupported dtype {dtype!r}; choose from {sorted(DTYPES)}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")

    model_path = str(model_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
    )
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=local_files_only,
        torch_dtype=DTYPES[dtype],
    )
    model = model.to(device).eval()
    model.config.use_cache = False

    layers = getattr(getattr(model, "model", None), "layers", None)
    if layers is None:
        raise TypeError("Loaded model does not expose DiffuCoder model.model.layers")
    configured_layers = int(getattr(model.config, "num_hidden_layers", -1))
    if len(layers) != configured_layers:
        raise ValueError(
            f"DiffuCoder exposes {len(layers)} layers but config says "
            f"{configured_layers}"
        )
    return model, tokenizer


def prepare_diffucoder_prompt_batch(
    tokenizer,
    prompts: Iterable[str],
    device: str,
):
    messages = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        for prompt in prompts
    ]
    rendered = [
        tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=False,
        )
        for conversation in messages
    ]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
