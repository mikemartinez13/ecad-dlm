"""Autoregressive reference: Qwen2.5-Coder-7B-Instruct on the same coding tasks.

DiffuCoder-7B was adapted from Qwen2.5-Coder-7B, so this model has the same
shape. It decodes greedily with a KV cache, one token per forward, until end
of turn or the config's gen_length new tokens. It uses the same chat prompt as
DiffuCoder and no repetition penalty (Qwen's default is 1.1). Records the wall
time and token count of every task and EvalPlus base pass@1.

Compare with decoding_oracles.py on the same GPU type: it records DiffuCoder's
measured dense time per task and the modeled ceilings.

Usage (inside a GPU allocation, from the repository root):

    python -m dlm_kv_policy.autoregressive_reference \\
        --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from ecad.language_models.code_execution import run_evalplus_apptainer
from ecad.language_models.code_tasks_and_response_processing import (
    load_code_task_dataset,
    make_code_task_output,
)
from ecad.language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

from .code_correctness_and_latency_evaluator import gpu_folder_name
from .kv_policy_experiment_config import load_kv_policy_experiment_config

QWEN_CODER_7B_INSTRUCT = (
    "/scratch/gqs7zr/model_checkpoints/models--Qwen--Qwen2.5-Coder-7B-Instruct/"
    "snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-path", default=QWEN_CODER_7B_INSTRUCT)
    args = parser.parse_args()

    config = load_kv_policy_experiment_config(args.config)
    device = config.model.device
    tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    # transformers 4.52 on torch 2.4 rejects Qwen2's tensor-parallel plan; we use one GPU.
    model_config = AutoConfig.from_pretrained(args.model_path, local_files_only=True)
    model_config.base_model_tp_plan = None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        config=model_config,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        local_files_only=True,
    ).to(device).eval()
    output_dir = config.run.run_dir / gpu_folder_name() / "autoregressive_reference"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir}", flush=True)
    greedy = dict(do_sample=False, temperature=None, top_p=None, top_k=None, repetition_penalty=1.0)

    # One untimed call so CUDA start-up is not charged to the first task.
    prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [tasks[0].prompt], device)
    model.generate(prompt, attention_mask=attention_mask, max_new_tokens=8, **greedy)

    records, responses = [], []
    for index, task in enumerate(tasks):
        prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], device)
        torch.cuda.synchronize()
        started = perf_counter()
        with torch.no_grad():
            output = model.generate(
                prompt, attention_mask=attention_mask, max_new_tokens=config.generation.gen_length, **greedy
            )
        torch.cuda.synchronize()
        seconds = perf_counter() - started
        generated = output[0, prompt.shape[1] :]
        responses.append(tokenizer.decode(generated, skip_special_tokens=True))
        records.append({
            "task_id": task.task_id,
            "prompt_length": prompt.shape[1],
            "new_tokens": int(generated.numel()),
            "seconds": seconds,
        })
        print(f"task {index + 1}/{len(tasks)} {task.task_id}: {generated.numel()} tokens, {seconds:.2f} s", flush=True)

    task_outputs = [make_code_task_output(task, response) for task, response in zip(tasks, responses)]
    settings = config.evaluation
    code_execution = run_evalplus_apptainer(
        task_outputs,
        image=settings.evalplus_image,
        work_dir=output_dir / "code_execution",
        parallel=settings.evalplus_parallel,
        apptainer_bin=settings.evalplus_apptainer_bin,
        scope="base",
        timeout_seconds=settings.evalplus_timeout_seconds,
        dataset_paths={"humaneval": settings.evalplus_humaneval_path, "mbpp": settings.evalplus_mbpp_path},
    )
    total_seconds = sum(record["seconds"] for record in records)
    total_tokens = sum(record["new_tokens"] for record in records)
    summary = {
        "model_path": args.model_path,
        "gpu": gpu_folder_name(),
        "task_count": len(tasks),
        "seconds_per_task": total_seconds / len(tasks),
        "new_tokens_per_task": total_tokens / len(tasks),
        "seconds_per_token": total_seconds / total_tokens,
        "base_pass_at_1": code_execution["aggregate"]["base_pass_at_1"],
        "tasks": records,
        "task_outputs": task_outputs,
        "code_execution": code_execution,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"\nQwen2.5-Coder-7B-Instruct on {summary['gpu']}: {summary['seconds_per_task']:.2f} s/task, "
        f"{summary['new_tokens_per_task']:.0f} tokens/task, "
        f"{summary['seconds_per_token'] * 1e3:.1f} ms/token, base pass@1 {summary['base_pass_at_1']:.3f}"
    )


if __name__ == "__main__":
    main()
