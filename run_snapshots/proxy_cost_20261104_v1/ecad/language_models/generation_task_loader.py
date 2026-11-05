"""Load fixed code task datasets or free-form text prompts for DLM generation."""

from __future__ import annotations

from pathlib import Path

from ecad.language_models.task_dataset_path_compatibility import resolve_task_source_path

from ecad.language_models.code_tasks_and_response_processing import (
    CodeGenerationTask,
    load_code_task_dataset,
    select_evalplus_code_tasks,
)


def read_prompts(path: str | Path, max_prompts: int | None = None) -> list[str]:
    prompts = [
        line.strip()
        for line in resolve_task_source_path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if max_prompts is not None:
        prompts = prompts[:max_prompts]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def load_generation_tasks(evaluation) -> list[CodeGenerationTask]:
    """Load free-form prompts or a reproducible EvalPlus task selection."""

    if evaluation.source == "text":
        if evaluation.prompts_file is None:
            raise ValueError("prompts_file is required for text generation tasks")
        prompts = read_prompts(evaluation.prompts_file, evaluation.max_prompts)
        return [
            CodeGenerationTask(
                task_id=f"prompt_{index:05d}",
                dataset="text",
                prompt=prompt,
                original_prompt=prompt,
                completion_mode="free_form",
            )
            for index, prompt in enumerate(prompts)
        ]

    if evaluation.code_tasks_file:
        tasks = load_code_task_dataset(evaluation.code_tasks_file)
        return tasks[: evaluation.max_prompts]

    return select_evalplus_code_tasks(
        evaluation.code_datasets,
        max_tasks=evaluation.max_prompts,
        max_tasks_per_dataset=evaluation.code_max_tasks_per_dataset,
        seed=evaluation.code_seed,
        prompt_style=evaluation.code_prompt_style,
    )
