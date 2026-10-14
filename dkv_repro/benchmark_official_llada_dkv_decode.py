"""Compare unchanged official LLaDA dense and dKV-Decode generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_official_llada_dkv_generation import DEFAULT_RESULTS_ROOT, LILY_PROMPT, default_run_args, run, write_json


CASES = {
    "lily_128tokens_refresh4": {"steps": 128, "gen_length": 128, "block_length": 64, "cache_reloading_step": 4},
    "short_128tokens_refresh8": {"steps": 128, "gen_length": 128, "block_length": 64, "cache_reloading_step": 8},
    "paper_512tokens_refresh8": {"steps": 512, "gen_length": 512, "block_length": 32, "cache_reloading_step": 8},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, default="lily_128tokens_refresh4")
    parser.add_argument("--model-path")
    parser.add_argument("--prompt", default=LILY_PROMPT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", default=str(DEFAULT_RESULTS_ROOT / "dkv_reproduction"))
    parser.add_argument("--label", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = CASES[args.case]
    records = {}
    for mode in ("origin", "decode"):
        single = default_run_args()
        single.mode = mode
        single.model_path = args.model_path or single.model_path
        single.prompt = args.prompt
        single.repeats = args.repeats
        single.warmup = args.warmup
        single.output_dir = args.output_dir
        single.label = args.label or args.case
        for key, value in settings.items():
            setattr(single, key, value)
        records[mode] = run(single)
    dense, cached = records["origin"], records["decode"]
    comparison = {
        "case": args.case,
        "dense_median_seconds": dense["median_seconds"],
        "decode_median_seconds": cached["median_seconds"],
        "speedup": dense["median_seconds"] / cached["median_seconds"],
        "dense_tokens_per_second": dense["generated_tokens_per_second"],
        "decode_tokens_per_second": cached["generated_tokens_per_second"],
        "outputs": {mode: record["generated_text"] for mode, record in records.items()},
    }
    path = Path(args.output_dir) / (args.label or args.case) / "comparison.json"
    write_json(path, comparison)
    print(json.dumps({"comparison": str(path), "speedup": comparison["speedup"]}, indent=2))


if __name__ == "__main__":
    main()
