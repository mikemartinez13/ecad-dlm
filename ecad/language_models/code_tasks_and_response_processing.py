"""Code task datasets, benchmark prompts, and generated-response reconstruction."""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
import re
import textwrap
from typing import Any, Iterable

from ecad.language_models.task_dataset_path_compatibility import resolve_task_source_path


SUPPORTED_EVALPLUS_DATASETS = ("humaneval", "mbpp")
SUPPORTED_CODE_PROMPT_STYLES = ("strict_python", "opencompass")

# Increment this whenever executable reconstruction changes. Saved artifacts keep
# raw model responses, so post-hoc evaluation can deterministically rebuild code
# with a newer version without rerunning the model.
CODE_POSTPROCESSOR_VERSION = "opencompass-compatible-v1"

OPENCOMPASS_HUMANEVAL_POSTPROCESSOR_SOURCE = (
    "https://github.com/open-compass/opencompass/blob/main/"
    "opencompass/datasets/humaneval.py"
)
OPENCOMPASS_MBPP_POSTPROCESSOR_SOURCE = (
    "https://github.com/open-compass/opencompass/blob/main/"
    "opencompass/datasets/mbpp.py"
)

_TOP_LEVEL_FUNCTION_RE = re.compile(
    r"(?m)^(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("
)

# Kept in the same order as OpenCompass MBPPEvaluator._process_answer. The
# standard-library re module is sufficient here because generations have a
# fixed, small token budget; OpenCompass uses the third-party regex timeout as
# an additional guard for arbitrary-length inputs.
_OPENCOMPASS_MBPP_PATTERNS = (
    r"\[BEGIN\]\s*'(.*)'\s*\[DONE\]",
    r"BEGIN\s*'(.*)'\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)'\s*DONE",
    r"BEGIN\s*'(.*)'\s*DONE",
    r"\[BEGIN\]\s*'(.*)\s*\[DONE\]",
    r"BEGIN\s*'(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*'(.*)\s*DONE",
    r"BEGIN\s*'(.*)\s*DONE",
    r"\[BEGIN\]\s*(.*)\s*\[DONE\]",
    r"BEGIN\s*(.*)\s*\[DONE\]",
    r"\[BEGIN\]\s*(.*)\s*DONE",
    r"BEGIN\s*(.*)\s*DONE",
    r"```python\s*(.*)\s*```",
    r"```\s*(.*)\s*```",
    r"```python\s*(.*)\s*$",
    r"```\s*(.*)\s*$",
    r"(.*)\s*```.*",
    r"\[BEGIN\]\s*'(.*)",
    r"\[BEGIN\](.*)",
    r"'(.*)'\s*\[DONE\]",
)


@dataclass(frozen=True)
class CodePostprocessResult:
    normalized_response: str
    executable_code: str
    extractor: str
    extraction_strategy: str
    reconstruction_strategy: str
    source: str | None

    def provenance(self) -> dict[str, str | None]:
        return {
            "version": CODE_POSTPROCESSOR_VERSION,
            "extractor": self.extractor,
            "extraction_strategy": self.extraction_strategy,
            "reconstruction_strategy": self.reconstruction_strategy,
            "source": self.source,
        }


_MBPP_OPENCOMPASS_FEWSHOT = """You are an expert Python programmer, and here is your task: Write a function to find the similar elements from the given two tuple lists.
L1: Your code should pass these tests:

 assert similar_elements((3, 4, 5, 6),(5, 7, 4, 10)) == (4, 5)
assert similar_elements((1, 2, 3, 4),(5, 4, 3, 7)) == (3, 4)
assert similar_elements((11, 12, 14, 13),(17, 15, 14, 13)) == (13, 14)

[BEGIN]
 'def similar_elements(test_tup1, test_tup2):
 res = tuple(set(test_tup1) & set(test_tup2))
 return (res)'
[DONE]

You are an expert Python programmer, and here is your task: Write a python function to identify non-prime numbers.
L2: Your code should pass these tests:

 assert is_not_prime(2) == False
assert is_not_prime(10) == True
assert is_not_prime(35) == True

[BEGIN]
 'import math
def is_not_prime(n):
 result = False
 for i in range(2,int(math.sqrt(n)) + 1):
 if n % i == 0:
 result = True
 return result'
[DONE]

You are an expert Python programmer, and here is your task: Write a function to find the largest integers from a given list of numbers using heap queue algorithm.
L3: Your code should pass these tests:

 assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 3) == [85, 75, 65]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 2) == [85, 75]
assert heap_queue_largest([25, 35, 22, 85, 14, 65, 75, 22, 58], 5) == [85, 75, 65, 58, 35]

[BEGIN]
 'import heapq as hq
def heap_queue_largest(nums,n):
 largest_nums = hq.nlargest(n, nums)
 return largest_nums'
[DONE]

"""


@dataclass(frozen=True)
class CodeGenerationTask:
    """A reproducible coding task used for code generation and correctness evaluation."""

    task_id: str
    dataset: str
    prompt: str
    original_prompt: str
    completion_mode: str
    language: str = "python"
    entry_point: str | None = None
    record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def render_strict_python_prompt(
    dataset: str, original_prompt: str, task_id: str
) -> tuple[str, str]:
    """Render a prompt that asks LLaDA for executable Python only.

    HumanEval and MBPP both ask for executable Python. HumanEval receives a
    replacement function definition so that models do not need to reliably
    follow a suffix-only completion format; the evaluator preserves imports
    from the original HumanEval prompt.
    """

    dataset = dataset.lower()
    if dataset == "humaneval":
        instruction = (
            "Complete the Python function in the task below. Return only the "
            "complete replacement function definition, starting with def or "
            "async def. Do not repeat imports from the task prompt. Do not use "
            "Markdown code fences. Do not include explanations, tests, or "
            "commentary."
        )
        completion_mode = "function_definition"
    elif dataset == "mbpp":
        instruction = (
            "Solve the Python programming task below. Return only the complete "
            "Python solution code, including the requested function definition. "
            "Do not use Markdown code fences. Do not include explanations, tests, "
            "or commentary."
        )
        completion_mode = "complete_program"
    else:
        raise ValueError(
            f"Unsupported coding dataset {dataset!r}; "
            f"choose from {SUPPORTED_EVALPLUS_DATASETS}"
        )

    rendered = (
        f"{instruction}\n\n"
        f"Task ID: {task_id}\n\n"
        f"Task:\n{original_prompt.strip()}"
    )
    return rendered, completion_mode


def _split_mbpp_prompt(original_prompt: str) -> tuple[str, str]:
    """Recover MBPP's task text and base assertions from EvalPlus metadata."""

    text = original_prompt.strip()
    if text.startswith("\"\"\"") and text.endswith("\"\"\""):
        text = text[3:-3].strip()
    assertion = re.search(r"(?m)^\s*assert\b", text)
    if assertion is None:
        return text, ""
    return text[: assertion.start()].strip(), text[assertion.start() :].strip()


def render_opencompass_prompt(
    dataset: str,
    original_prompt: str,
    task_id: str,
) -> tuple[str, str]:
    """Render the coding prompts used by LLaDA's official OpenCompass path."""

    del task_id
    dataset = dataset.lower()
    if dataset == "humaneval":
        return f"Complete the following python code:\n{original_prompt.strip()}", "function_suffix"
    if dataset == "mbpp":
        task_text, tests = _split_mbpp_prompt(original_prompt)
        target = (
            "You are an expert Python programmer, and here is your task: "
            f"{task_text} Your code should pass these tests:\n\n {tests} \n"
        )
        return _MBPP_OPENCOMPASS_FEWSHOT + target + "\n[BEGIN]\n", "complete_program"
    raise ValueError(
        f"Unsupported coding dataset {dataset!r}; "
        f"choose from {SUPPORTED_EVALPLUS_DATASETS}"
    )


def _json_safe_record(record: dict[str, Any]) -> dict[str, Any]:
    """Keep EvalPlus metadata JSON-serializable for a portable task dataset."""

    try:
        json.dumps(record)
    except TypeError:
        return {key: str(value) for key, value in record.items()}
    return record


def select_evalplus_code_tasks(
    datasets: Iterable[str],
    *,
    max_tasks: int | None = None,
    max_tasks_per_dataset: int | None = None,
    seed: int = 0,
    prompt_style: str = "strict_python",
) -> list[CodeGenerationTask]:
    """Load and deterministically select HumanEval+/MBPP+ tasks.

    EvalPlus is imported lazily so the rest of the DLM schedule tooling can be
    used without installing the execution benchmark. Its public data helpers
    download/cache the current pinned benchmark data on first use.
    """

    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise RuntimeError(
            "EvalPlus is required for code task preparation. Install it in the "
            "runtime environment with: pip install evalplus"
        ) from exc

    requested = [name.lower() for name in datasets]
    invalid = [name for name in requested if name not in SUPPORTED_EVALPLUS_DATASETS]
    if invalid:
        raise ValueError(
            f"Unsupported EvalPlus dataset(s): {invalid}; "
            f"choose from {SUPPORTED_EVALPLUS_DATASETS}"
        )
    if not requested:
        raise ValueError("At least one EvalPlus dataset is required")
    if prompt_style not in SUPPORTED_CODE_PROMPT_STYLES:
        raise ValueError(
            f"Unsupported code prompt style {prompt_style!r}; "
            f"choose from {SUPPORTED_CODE_PROMPT_STYLES}"
        )
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    if max_tasks_per_dataset is not None and max_tasks_per_dataset < 1:
        raise ValueError("max_tasks_per_dataset must be positive")

    loaders = {
        "humaneval": get_human_eval_plus,
        "mbpp": get_mbpp_plus,
    }
    per_dataset: dict[str, list[CodeGenerationTask]] = {}
    for dataset in requested:
        raw_tasks = loaders[dataset]()
        if not isinstance(raw_tasks, dict):
            raw_tasks = {
                str(item["task_id"]): item
                for item in raw_tasks
            }
        tasks = []
        for fallback_id, raw in raw_tasks.items():
            task_id = str(raw.get("task_id", fallback_id))
            original_prompt = str(raw["prompt"]).strip()
            if prompt_style == "opencompass":
                rendered, completion_mode = render_opencompass_prompt(
                    dataset, original_prompt, task_id
                )
            else:
                rendered, completion_mode = render_strict_python_prompt(
                    dataset, original_prompt, task_id
                )
            tasks.append(
                CodeGenerationTask(
                    task_id=task_id,
                    dataset=dataset,
                    prompt=rendered,
                    original_prompt=original_prompt,
                    completion_mode=completion_mode,
                    language="python",
                    entry_point=raw.get("entry_point"),
                    record=_json_safe_record(dict(raw)),
                )
            )
        random.Random(seed + requested.index(dataset)).shuffle(tasks)
        if max_tasks_per_dataset is not None:
            tasks = tasks[:max_tasks_per_dataset]
        per_dataset[dataset] = tasks

    # Round-robin selection prevents the first dataset from consuming the
    # entire global budget and makes mixed task datasets reproducible.
    selected: list[CodeGenerationTask] = []
    position = 0
    while True:
        added = False
        for dataset in requested:
            tasks = per_dataset[dataset]
            if position < len(tasks):
                selected.append(tasks[position])
                added = True
                if max_tasks is not None and len(selected) >= max_tasks:
                    return selected
        if not added:
            break
        position += 1
    return selected


def load_code_task_dataset(path: str | Path) -> list[CodeGenerationTask]:
    """Load task records from JSONL, a JSON list, or a saved run task object."""

    dataset_path = resolve_task_source_path(path)
    if not dataset_path.is_absolute():
        raise ValueError(f"Code task dataset path must be absolute: {dataset_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Code task dataset not found: {dataset_path}")

    text = dataset_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Code task dataset is empty: {dataset_path}")
    try:
        raw_tasks = json.loads(text)
    except json.JSONDecodeError:
        raw_tasks = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(raw_tasks, dict):
        raw_tasks = raw_tasks["tasks"] if "tasks" in raw_tasks else [raw_tasks]
    tasks = [CodeGenerationTask(**item) for item in raw_tasks]
    if not tasks:
        raise ValueError(f"Code task dataset contains no tasks: {dataset_path}")
    return tasks


def write_code_task_dataset(
    tasks: Iterable[CodeGenerationTask], path: str | Path
) -> Path:
    """Write a deterministic JSONL task dataset at an absolute path."""

    dataset_path = Path(path)
    if not dataset_path.is_absolute():
        raise ValueError(f"Code task dataset path must be absolute: {dataset_path}")
    task_list = list(tasks)
    if not task_list:
        raise ValueError("Cannot write an empty code task dataset")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8") as handle:
        for task in task_list:
            handle.write(json.dumps(task.to_dict(), sort_keys=True) + "\n")
    return dataset_path


def normalize_python_response(response: str) -> str:
    """Remove a Markdown wrapper when it encloses the complete response.

    Dataset evaluators should use :func:`postprocess_code_response`, which
    reproduces their benchmark-specific extraction behavior. This helper stays
    intentionally small for free-form and backward-compatible callers.
    """

    text = response.strip()
    if text.startswith("```python"):
        text = text[len("```python") :].lstrip("\n")
    elif text.startswith("```"):
        text = text[3:].lstrip("\n")
    if text.endswith("```"):
        text = text[:-3].rstrip()
    return text


def _extract_humaneval_response(response: str) -> tuple[str, str]:
    """Apply OpenCompass humaneval_postprocess_v2 with safe fallbacks."""

    # This is the exact first-choice expression used by OpenCompass v2.
    blocks = re.findall(r"```\w*\n(.*?)```", response, re.DOTALL)
    if blocks:
        return blocks[0].lstrip(), "opencompass_first_fenced_block"

    # Handle an unlabeled or whitespace-separated fence without changing the
    # official priority above. This also makes truncated historical responses
    # diagnosable instead of silently retaining their prose prefix.
    fallback = re.search(r"```[^\n`]*\n?(.*?)(?:```|$)", response, re.DOTALL)
    if fallback:
        return fallback.group(1).lstrip(), "compatible_fenced_block_fallback"
    return response.lstrip(), "opencompass_raw_fallback"


def _extract_mbpp_response(response: str) -> tuple[str, str]:
    """Apply OpenCompass MBPPEvaluator._process_answer pattern ordering."""

    text = response
    strategy = "opencompass_raw_fallback"
    for index, pattern in enumerate(_OPENCOMPASS_MBPP_PATTERNS):
        match = re.search(pattern, text, re.DOTALL)
        if match:
            text = match.group(1)
            strategy = f"opencompass_pattern_{index:02d}"
            break
    text = text.split("```", 1)[0]
    text = re.split(r"'?\s*\[?DONE\]?", text, maxsplit=1)[0]
    text = text.replace(r"\_", "_").strip()
    return text, strategy


def _original_function_prelude(
    original_prompt: str, entry_point: str | None
) -> str:
    match = _target_definition_match(original_prompt, entry_point)
    return original_prompt[: match.start()].rstrip() if match else ""


def _target_definition_match(
    text: str, entry_point: str | None
) -> re.Match[str] | None:
    matches = list(_TOP_LEVEL_FUNCTION_RE.finditer(text))
    if entry_point:
        for match in matches:
            if match.group(1) == entry_point:
                return match
    return matches[0] if matches else None


def _reconstruct_humaneval(
    task: CodeGenerationTask, extracted: str, extraction_strategy: str
) -> tuple[str, str]:
    """Support both canonical suffixes and LLaDA's complete definitions."""

    definition = _target_definition_match(extracted, task.entry_point)
    if definition is not None:
        generated = extracted.rstrip()
        # Fenced blocks are already code-only. If an unfenced model response
        # contains prose before a top-level definition, discard that prose.
        if extraction_strategy == "opencompass_raw_fallback":
            generated = generated[definition.start() :]
        prelude = _original_function_prelude(
            task.original_prompt, task.entry_point
        )
        executable = f"{prelude}\n\n{generated}" if prelude else generated
        return executable, "complete_definition_replace"

    # HumanEval normally asks for a suffix. EvalPlus accepts a complete solution,
    # so restore the function-body indentation that can be removed by response
    # postprocessing and append it to the original stub.
    suffix = textwrap.dedent(extracted).strip("\n")
    suffix = textwrap.indent(suffix, "    ") if suffix else ""
    executable = task.original_prompt.rstrip()
    if suffix:
        executable = f"{executable}\n{suffix}"
    return executable, "function_suffix_append"


def postprocess_code_response(
    task: CodeGenerationTask, response: str
) -> CodePostprocessResult:
    """Extract and reconstruct one benchmark response without executing it."""

    dataset = task.dataset.lower()
    if dataset == "humaneval":
        extracted, extraction_strategy = _extract_humaneval_response(response)
        executable, reconstruction_strategy = _reconstruct_humaneval(
            task, extracted, extraction_strategy
        )
        return CodePostprocessResult(
            normalized_response=extracted,
            executable_code=executable,
            extractor="opencompass_humaneval_v2",
            extraction_strategy=extraction_strategy,
            reconstruction_strategy=reconstruction_strategy,
            source=OPENCOMPASS_HUMANEVAL_POSTPROCESSOR_SOURCE,
        )
    if dataset == "mbpp":
        extracted, extraction_strategy = _extract_mbpp_response(response)
        return CodePostprocessResult(
            normalized_response=extracted,
            executable_code=extracted,
            extractor="opencompass_mbpp_evaluator",
            extraction_strategy=extraction_strategy,
            reconstruction_strategy="complete_program",
            source=OPENCOMPASS_MBPP_POSTPROCESSOR_SOURCE,
        )

    normalized = normalize_python_response(response)
    if task.completion_mode not in {"complete_program", "free_form"}:
        raise ValueError(
            f"Unsupported completion mode {task.completion_mode!r} "
            f"for non-benchmark task {task.task_id}"
        )
    return CodePostprocessResult(
        normalized_response=normalized,
        executable_code=normalized,
        extractor="generic_python_wrapper",
        extraction_strategy="outer_markdown_wrapper",
        reconstruction_strategy=task.completion_mode,
        source=None,
    )


def reconstruct_executable_code(
    task: CodeGenerationTask, response: str
) -> str:
    """Turn a generated response into the code submitted to an evaluator."""

    return postprocess_code_response(task, response).executable_code


def _syntax_report(task: CodeGenerationTask, executable_code: str) -> dict[str, Any]:
    if task.dataset.lower() not in SUPPORTED_EVALPLUS_DATASETS:
        return {"checked": False, "valid": None, "error": None}
    try:
        ast.parse(executable_code)
    except SyntaxError as error:
        return {
            "checked": True,
            "valid": False,
            "error": {
                "message": error.msg,
                "line": error.lineno,
                "offset": error.offset,
            },
        }
    return {"checked": True, "valid": True, "error": None}


def make_code_task_output(
    task: CodeGenerationTask, response: str
) -> dict[str, Any]:
    """Serialize task provenance, raw response, and executable code."""

    processed = postprocess_code_response(task, response)
    syntax = _syntax_report(task, processed.executable_code)

    return {
        "task_id": task.task_id,
        "dataset": task.dataset,
        "language": task.language,
        "entry_point": task.entry_point,
        "completion_mode": task.completion_mode,
        "original_prompt": task.original_prompt,
        "response": response,
        "normalized_response": processed.normalized_response,
        "executable_code": processed.executable_code,
        "postprocessing": processed.provenance(),
        "syntax_valid": syntax["valid"],
        "syntax_error": syntax["error"],
    }
