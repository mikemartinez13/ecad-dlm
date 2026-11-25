"""Resolve renamed task sources and preserve historical checkpoint signatures."""

from pathlib import Path


LEGACY_TASK_SOURCE_FILENAMES = {
    "llada_evalplus_calibration_64_len1024.jsonl":
        "humaneval_mbpp_64_strict_python_seed0.jsonl",
    "llada_evalplus_calibration_64.jsonl":
        "humaneval_mbpp_64_strict_python_function_suffix_seed0.jsonl",
    "llada_humaneval_calibration_32.jsonl": "humaneval_32_opencompass_seed0.jsonl",
    "llada_mbpp_calibration_32.jsonl": "mbpp_32_opencompass_seed0.jsonl",
    "dlm_ecad_calibration.txt": "dlm_text_generation_prompts.txt",
}


def resolve_task_source_path(path: str | Path) -> Path:
    """Read a renamed dataset when a saved config still names its old file."""

    source = Path(path)
    replacement = LEGACY_TASK_SOURCE_FILENAMES.get(source.name)
    if replacement is not None and not source.exists():
        return source.with_name(replacement)
    return source


def legacy_task_source_path(path: str | Path) -> str:
    """Keep filename-only migrations out of existing ECAD config hashes."""

    source = Path(path)
    for legacy, current in LEGACY_TASK_SOURCE_FILENAMES.items():
        if source.name == current:
            return str(source.with_name(legacy))
    return str(path)
