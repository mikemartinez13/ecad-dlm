"""Run one official dKV-Cache LLaDA implementation from a local checkpoint.

This wrapper deliberately imports the author implementations unchanged. Its
only roles are local/offline loading, provenance, and synchronized timing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, __version__ as transformers_version


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ecad.language_models.experiment_artifact_paths import DEFAULT_RESULTS_ROOT

OFFICIAL_ROOT = ROOT / "dkv_cache_official"
OFFICIAL_SOURCE_METADATA = ROOT / "dkv_repro" / "metadata" / "official_source.json"
DEFAULT_MODEL = Path(
    "/scratch/gqs7zr/model_checkpoints/models--GSAI-ML--LLaDA-8B-Instruct/"
    "snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07"
)
LILY_PROMPT = (
    "Lily can run 12 kilometers per hour for 4 hours. After that, she runs 6 "
    "kilometers per hour. How many kilometers can she run in 8 hours?"
)


def _command(*args: str) -> str | None:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("config.json", "model.safetensors.index.json"):
        digest.update((path / name).read_bytes())
    return digest.hexdigest()


def official_source_commit() -> str | None:
    """Read the upstream commit without accidentally using ECAD's Git history."""

    if (OFFICIAL_ROOT / ".git").exists():
        return _command("git", "-C", str(OFFICIAL_ROOT), "rev-parse", "HEAD")
    return json.loads(OFFICIAL_SOURCE_METADATA.read_text(encoding="utf-8")).get(
        "official_repository_commit"
    )


def verify_checkpoint(path: Path) -> dict[str, Any]:
    required = [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "modeling_llada.py",
        "model.safetensors.index.json",
    ]
    missing = [name for name in required if not (path / name).is_file()]
    shards = sorted(path.glob("model-*-of-*.safetensors"))
    if missing or not shards:
        raise FileNotFoundError(f"Incomplete local LLaDA checkpoint: missing={missing}, shards={len(shards)}")
    return {"path": str(path), "shard_count": len(shards), "manifest_sha256": _checkpoint_digest(path)}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def author_imports(mode: str):
    if not OFFICIAL_ROOT.is_dir():
        raise FileNotFoundError(f"Missing official source: {OFFICIAL_ROOT}")
    source = str(OFFICIAL_ROOT)
    if source not in sys.path:
        sys.path.insert(0, source)
    if mode == "origin":
        from generation_utils.llada_generate import generate

        return AutoModel, generate
    if mode == "decode":
        from generation_utils.llada_dkv_cache_decode import generate
        from models.modeling_llada_dkv_cache_decode import LLaDAModelLM

        return LLaDAModelLM, generate
    if mode == "greedy":
        from generation_utils.llada_dkv_cache_greedy import generate
        from models.modeling_llada_dkv_cache_greedy import LLaDAModelLM

        return LLaDAModelLM, generate
    raise ValueError(f"Unsupported mode {mode!r}")


def load(mode: str, model_path: Path):
    model_cls, generate = author_imports(mode)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = model_cls.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        # Match the authors' launcher. On a single A6000 this places the
        # complete model on CUDA while retaining their Accelerate dispatch.
        device_map="auto",
    ).eval()
    return model, tokenizer, generate


def tokenized_prompt(tokenizer, question: str) -> torch.Tensor:
    messages = [[{"role": "user", "content": "Please answer the question step by step and put the answer in \\boxed{}." + question}]]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    token_ids = tokenizer(prompt, padding_side="left", padding="longest")["input_ids"]
    return torch.tensor(token_ids, device="cuda")


def generate_once(generate, model, tokenizer, input_ids: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    kwargs = {
        "steps": args.steps,
        "gen_length": args.gen_length,
        "block_length": args.block_length,
        "temperature": args.temperature,
        "cfg_scale": args.cfg_scale,
        "remasking": args.remasking,
    }
    if args.mode != "origin":
        kwargs.update(enable_cache=True, cache_reloading_step=args.cache_reloading_step)
    if args.mode == "greedy":
        kwargs["window_size"] = args.window_size
    return generate(model, tokenizer, input_ids, **kwargs)


def runtime_metadata(model_path: Path) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers_version,
        "gpu": torch.cuda.get_device_name(0),
        "driver": _command("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
        "official_commit": official_source_commit(),
        "checkpoint": verify_checkpoint(model_path),
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("This reproduction is GPU-only; submit its Slurm launcher")
    if transformers_version != "4.46.3":
        raise RuntimeError(f"Expected transformers 4.46.3, found {transformers_version}")
    model_path = Path(args.model_path).resolve()
    verify_checkpoint(model_path)
    set_seed(args.seed)
    model, tokenizer, generate = load(args.mode, model_path)
    input_ids = tokenized_prompt(tokenizer, args.prompt)
    if args.mode != "origin" and input_ids.shape[0] != 1:
        raise ValueError("The official dKV LLaDA generators support batch size one")

    for _ in range(args.warmup):
        with torch.inference_mode():
            generate_once(generate, model, tokenizer, input_ids, args)
    torch.cuda.synchronize()
    samples = []
    output = None
    for _ in range(args.repeats):
        torch.cuda.synchronize()
        started = perf_counter()
        with torch.inference_mode():
            output = generate_once(generate, model, tokenizer, input_ids, args)
        torch.cuda.synchronize()
        samples.append(perf_counter() - started)
    assert output is not None
    generated = tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)
    median = sorted(samples)[len(samples) // 2]
    record = {
        "metadata": runtime_metadata(model_path),
        "method": args.mode,
        "generation": {
            "steps": args.steps,
            "gen_length": args.gen_length,
            "block_length": args.block_length,
            "temperature": args.temperature,
            "cfg_scale": args.cfg_scale,
            "remasking": args.remasking,
            "cache_reloading_step": args.cache_reloading_step if args.mode != "origin" else None,
            "window_size": args.window_size if args.mode == "greedy" else None,
            "seed": args.seed,
        },
        "prompt": args.prompt,
        "timing_seconds": samples,
        "median_seconds": median,
        "generated_tokens_per_second": args.gen_length / median,
        "generated_text": generated,
    }
    output_path = Path(args.output_dir) / args.label / args.mode / "reproduction.json"
    record["artifact_path"] = str(output_path)
    write_json(output_path, record)
    # The benchmark invokes dense and dKV sequentially in one process. Do not
    # leave the first 8B model resident while loading the second one.
    del model, tokenizer, input_ids, output
    torch.cuda.empty_cache()
    print(json.dumps({"output": str(output_path), "median_seconds": median, "tokens_per_second": record["generated_tokens_per_second"]}, indent=2))
    return record


def default_run_args() -> argparse.Namespace:
    """Defaults shared by the standalone runner and comparison driver."""
    return argparse.Namespace(
        mode="origin",
        model_path=os.environ.get("LLADA_PATH", str(DEFAULT_MODEL)),
        prompt=LILY_PROMPT,
        steps=128,
        gen_length=128,
        block_length=64,
        temperature=0.0,
        cfg_scale=0.0,
        remasking="low_confidence",
        cache_reloading_step=8,
        window_size=0,
        seed=42,
        warmup=1,
        repeats=3,
        output_dir=str(DEFAULT_RESULTS_ROOT / "dkv_reproduction"),
        label="run",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("origin", "decode", "greedy"),
        required=True,
        help="origin: author dense baseline; decode: dKV-Decode; greedy: dKV-Greedy",
    )
    parser.add_argument("--model-path", default=os.environ.get("LLADA_PATH", str(DEFAULT_MODEL)))
    parser.add_argument("--prompt", default=LILY_PROMPT)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--gen-length", type=int, default=128)
    parser.add_argument("--block-length", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cfg-scale", type=float, default=0.0)
    parser.add_argument("--remasking", choices=("low_confidence", "random"), default="low_confidence")
    parser.add_argument("--cache-reloading-step", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_ROOT / "dkv_reproduction"))
    parser.add_argument("--label", default="run")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
