"""Loading and prompt preparation helpers for LLaDA checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch

DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


def load_llada_model_and_tokenizer(
    model_path: str | Path,
    *,
    device: str = "cuda",
    dtype: str = "bfloat16",
    local_files_only: bool = True,
):
    """Load one full model replica on the requested device.

    Imports Transformers lazily so schedule/config tooling does not require the
    model runtime dependencies.
    """

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
    if bool(getattr(model.config, "use_cache", False)):
        model.config.use_cache = False
    return model, tokenizer


def prepare_prompt_batch(tokenizer, prompts: Iterable[str], device: str):
    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        padding=True,
        return_tensors="pt",
    )
    mask_token_id = int(getattr(tokenizer, "mask_token_id", 126336) or 126336)
    if tokenizer.pad_token_id == mask_token_id:
        raise ValueError("LLaDA padding token must differ from its mask token")
    return encoded["input_ids"].to(device), encoded["attention_mask"].to(device)

