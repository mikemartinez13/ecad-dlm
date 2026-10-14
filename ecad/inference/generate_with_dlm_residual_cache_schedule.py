"""Run any configured DLM with an all-recompute or evolved ECAD schedule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ecad.language_models.residual_cache_model_backend import load_dlm_backend
from ecad.language_models.residual_cache_experiment_config import load_dlm_residual_cache_config
from ecad.language_models.residual_cache_schedule_evaluator import seed_everything
from ecad.language_models.generation_task_loader import load_generation_tasks, read_prompts
from ecad.language_models.experiment_artifact_paths import (
    DEFAULT_RESULTS_ROOT,
    resolve_experiment_result_path,
)
from ecad.schedulers.cache_scheduler.dlm_cache_schedule import DLMCacheSchedule


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-family", choices=("llada", "diffucoder"))
    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt", type=str)
    prompts.add_argument("--prompt-file", type=Path)
    parser.add_argument(
        "--output", type=Path,
        default=DEFAULT_RESULTS_ROOT / "dlm_inference/sample.json",
        help="Response JSON destination; parent directories are created automatically",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dlm_residual_cache_config(args.config)
    if args.model_path:
        config.model.path = args.model_path
    if args.model_family:
        config.model.family = args.model_family
    config.validate()

    if args.prompt:
        prompts = [args.prompt]
    elif args.prompt_file:
        prompts = read_prompts(args.prompt_file, config.evaluation.max_prompts)
    else:
        prompts = [task.prompt for task in load_generation_tasks(config.evaluation)]

    print(
        f"Loading {config.model.family} from {config.model.path} for inference",
        flush=True,
    )
    backend = load_dlm_backend(config)
    if args.schedule:
        schedule = DLMCacheSchedule.from_json(resolve_experiment_result_path(args.schedule))
    else:
        schedule = DLMCacheSchedule.all_recompute(
            num_layers=backend.num_layers,
            num_inference_steps=backend.num_schedule_steps,
            step_group_size=config.schedule.step_group_size,
            layer_group_size=config.schedule.layer_group_size,
            schedule_domain=backend.schedule_domain,
        )
    backend.attach_cache(schedule)

    records = []
    try:
        for index, prompt in enumerate(prompts):
            seed_everything(config.evolution.seed + index)
            input_ids, attention_mask = backend.prepare_prompt_batch([prompt])
            result = backend.generate(input_ids, attention_mask)
            generated = result.token_ids[:, input_ids.shape[1] :]
            response = backend.decode_generated(generated)[0]
            record = {
                "model_family": config.model.family,
                "prompt": prompt,
                "response": response,
                "elapsed_seconds": result.elapsed_seconds,
                "cache_stats": result.cache_stats,
            }
            records.append(record)
            print(f"PROMPT: {prompt}\nRESPONSE: {response}\n", flush=True)
    finally:
        backend.uninstall()

    if args.output:
        output = resolve_experiment_result_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(records, indent=2), encoding="utf-8")
        print(f"Saved responses to {output}", flush=True)


if __name__ == "__main__":
    main()
