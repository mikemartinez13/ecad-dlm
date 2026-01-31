# ecad/benchmark/generate_gsm8k_embeddings.py
"""
Precompute and save Dream DLM prompt "embeddings" (token tensors) for GSM8K.

This is the text-analog of generate_coco_embeddings.py / generate_mjhq_embeddings.py,
but hardcoded for GSM8K's expected layout:

  <gsm8k_dir>/
    train.jsonl
    gsm8k_test.jsonl

Each line must be JSON with at least:
  - "question": str
(Optional: "answer": str, which we ignore here.)

Outputs:
  <output_dir>/train/<id>.pt
  <output_dir>/test/<id>.pt

Each .pt file is produced by DreamTextGenerator.encode_and_save_prompts(...) and
contains a dict like:
  {"input_ids": ..., "attention_mask": ...}

NOTE:
- This script intentionally does NOT modify DreamTextGenerator.
- It explicitly uses DreamTextGenerator.encode_and_save_prompts().
- For any dataset other than GSM8K, it raises NotImplementedError.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

from ecad.image_generators.load_text_generator import get_text_generator_type
from ecad.image_generators.text_generator import TextGenerator


def _read_gsm8k_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path} line {line_no}") from e
            rows.append(obj)
    return rows


def _gsm8k_rows_to_name_to_prompt(rows: List[dict], split: str) -> Dict[str, str]:
    """
    Build a name->prompt mapping compatible with DreamTextGenerator.encode_and_save_prompts.

    We hardcode prompt = row["question"].

    Name format is deterministic:
      gsm8k_<split>_<index:06d>
    """
    out: Dict[str, str] = {}
    for i, r in enumerate(rows):
        if "question" not in r:
            raise KeyError(f"GSM8K row missing 'question' field at index {i}")
        name = f"gsm8k_{split}_{i:06d}"
        out[name] = str(r["question"])
    return out


def encode_split(
    generator: TextGenerator,
    input_dir: Path,
    output_dir: Path,
    split: str,
    batch_size: int,
    free_after: bool,
) -> None:
    if split == "train":
        in_file = input_dir / "train.jsonl"
        out_split_dir = output_dir / "train"
    elif split == "test":
        in_file = input_dir / "gsm8k_test.jsonl"
        out_split_dir = output_dir / "test"
    else:
        raise NotImplementedError(f"Unsupported split: {split}")

    rows = _read_gsm8k_jsonl(in_file)
    name_to_prompt = _gsm8k_rows_to_name_to_prompt(rows, split=split)

    # print(name_to_prompt.keys(), len(name_to_prompt))

    print(f"Encoding {len(name_to_prompt)} GSM8K '{split}' prompts from {in_file}")

    generator.tokenize_and_save_prompts(
        name_to_prompt=name_to_prompt,
        output_dir=out_split_dir,
        free_after=free_after,
        batch_size=batch_size,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Precompute Dream DLM token-tensor '.pt' prompt files for GSM8K."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="gsm8k",
        help="Dataset type. Only 'gsm8k' is supported for now.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing GSM8K JSONL files: train.jsonl and gsm8k_test.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write .pt files into (train/ and test/ subfolders).",
    )
    parser.add_argument(
        "--text-generator",
        type=str,
        default="DreamTextGenerator",
        help="Registered text generator name (e.g., DreamTextGenerator).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1024,
        help="How many prompts to tokenize at once (adjust for memory).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to construct the generator on. Tokenization is CPU-bound but generator init may check CUDA.",
    )
    parser.add_argument(
        "--weights-name",
        type=str,
        default=None,
        help="Optional HF model repo id / weights name passed into DreamTextGenerator(weights_name=...).",
    )
    parser.add_argument(
        "--free-after",
        action="store_true",
        help="Call generator.free_encoder_pipeline() after encoding (via encode_and_save_prompts free_after=True).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "test"],
        help="Which splits to encode: train, test",
    )

    args = parser.parse_args()

    if args.dataset.lower() != "gsm8k":
        raise NotImplementedError(
            f"Only dataset='gsm8k' is supported right now; got {args.dataset}"
        )

    if not args.input_dir.exists():
        raise FileNotFoundError(f"input-dir not found: {args.input_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Instantiate your generator using the registry
    gen_type = get_text_generator_type(args.text_generator)

    print(f"Using generator class: {gen_type}")
    try:
        generator: TextGenerator = gen_type(  # type: ignore[call-arg]
            weights_name=args.weights_name,
            schedule_path=None,  # encoding doesn't need schedules
            device=args.device,
        )
    except TypeError:
        # If your DreamTextGenerator __init__ signature differs, you may need to adjust here.
        # Per your instructions: if we can't use DreamTextGenerator as-is, tell you.
        raise RuntimeError(
            "Could not instantiate the text generator with (weights_name, schedule_path, device). "
            "Your DreamTextGenerator __init__ signature likely differs; adjust instantiation args here."
        )

    for split in args.splits:
        encode_split(
            generator=generator,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            split=split,
            batch_size=args.batch_size,
            free_after=args.free_after,
        )

    print("Done.")


if __name__ == "__main__":
    main()
