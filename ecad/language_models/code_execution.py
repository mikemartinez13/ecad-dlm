"""Small Apptainer wrapper for isolated EvalPlus execution."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import tempfile
from typing import Any, Iterable, Mapping


SANDBOX_PROFILE = "apptainer-contained-no-network-v1"
EVAL_SCOPES = {"base", "full"}


def _secure_directory(path: Path) -> None:
    if path.is_symlink():
        raise RuntimeError(f"Refusing to use a symlinked sandbox directory: {path}")
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    if not path.is_dir():
        raise RuntimeError(f"Sandbox path is not a directory: {path}")
    path.chmod(0o700)


def _bind_spec(source: Path, destination: str, mode: str = "rw") -> str:
    source_text = str(source)
    if any(character in source_text for character in (",", ":", "\n")):
        raise ValueError(f"Unsafe character in Apptainer bind path: {source}")
    return f"{source_text}:{destination}:{mode}"


def _validate_dataset_paths(
    paths: Mapping[str, str | Path] | None,
    datasets: Iterable[str],
) -> dict[str, Path]:
    """Validate immutable, host-provided EvalPlus data for an offline run."""

    supplied = dict(paths or {})
    requested = set(datasets)
    unknown = sorted(set(supplied) - {"humaneval", "mbpp"})
    if unknown:
        raise ValueError(f"Unsupported EvalPlus dataset path keys: {unknown}")
    missing = sorted(requested - set(supplied))
    if missing:
        raise ValueError(
            "Offline EvalPlus data paths are required for: " + ", ".join(missing)
        )

    resolved: dict[str, Path] = {}
    for dataset in requested:
        original = Path(supplied[dataset])
        if not original.is_absolute():
            raise ValueError(f"EvalPlus {dataset} data path must be absolute")
        if original.is_symlink():
            raise RuntimeError(
                f"Refusing symlinked EvalPlus {dataset} data file: {original}"
            )
        path = original.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"EvalPlus {dataset} data file not found: {path}")
        resolved[dataset] = path
    return resolved


def _write_samples(task_outputs: Iterable[dict[str, Any]], path: Path) -> None:
    path.unlink(missing_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        for output in task_outputs:
            handle.write(
                json.dumps(
                    {
                        "task_id": output["task_id"],
                        "solution": output["executable_code"],
                    }
                )
                + "\n"
            )


def _aggregate(
    results: list[dict[str, Any]], *, scope: str = "full"
) -> dict[str, float]:
    if scope not in EVAL_SCOPES:
        raise ValueError(f"scope must be one of {sorted(EVAL_SCOPES)}")
    if not results:
        return {}

    def fraction(passed_key: str, total_key: str) -> float:
        total = sum(int(result[total_key]) for result in results)
        passed = sum(int(result[passed_key]) for result in results)
        return passed / total if total else 0.0

    base_pass_at_1 = sum(
        result["base_status"] == "pass" for result in results
    ) / len(results)
    aggregate = {
        # Official one-sample-per-task EvalPlus headline metrics.
        "base_pass_at_1": base_pass_at_1,
        # Backward-compatible names retained for existing result consumers.
        "base_task_pass_rate": base_pass_at_1,
        # Assertion-level diagnostics; these are not benchmark pass@1 scores.
        "base_pass_fraction": fraction("base_passed", "base_total"),
    }
    if scope == "full":
        plus_pass_at_1 = sum(
            result["base_status"] == result["plus_status"] == "pass"
            for result in results
        ) / len(results)
        aggregate.update(
            {
                "plus_pass_at_1": plus_pass_at_1,
                "plus_task_pass_rate": plus_pass_at_1,
                "plus_pass_fraction": fraction("plus_passed", "plus_total"),
            }
        )
    return aggregate


def compare_base_correctness(
    baseline_result: dict[str, Any], candidate_result: dict[str, Any]
) -> dict[str, Any]:
    """Compare paired base-test outcomes for one baseline and candidate."""

    if baseline_result.get("scope") != "base" or candidate_result.get("scope") != "base":
        raise ValueError("Correctness objectives require base-only execution results")
    baseline_datasets = baseline_result.get("datasets", {})
    candidate_datasets = candidate_result.get("datasets", {})
    if set(baseline_datasets) != set(candidate_datasets):
        raise ValueError("Baseline/candidate dataset sets differ")
    for dataset in baseline_datasets:
        baseline_metadata = baseline_datasets[dataset]
        candidate_metadata = candidate_datasets[dataset]
        for key in ("dataset_hash", "subset_key", "groundtruth_cache_key"):
            baseline_value = baseline_metadata.get(key)
            if not baseline_value or baseline_value != candidate_metadata.get(key):
                raise ValueError(
                    f"Baseline/candidate {dataset} {key} values differ"
                )

    def statuses(result: dict[str, Any]) -> dict[str, str]:
        return {
            str(task["task_id"]): str(task["base_status"])
            for dataset in result.get("datasets", {}).values()
            for task in dataset.get("tasks", [])
        }

    baseline = statuses(baseline_result)
    candidate = statuses(candidate_result)
    if not baseline:
        raise ValueError("Baseline execution result has no task outcomes")
    if set(candidate) != set(baseline):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(
            "Baseline/candidate task sets differ: "
            f"missing={missing}, extra={extra}"
        )

    baseline_passed = {task for task, status in baseline.items() if status == "pass"}
    candidate_passed = {
        task for task, status in candidate.items() if status == "pass"
    }
    if not baseline_passed:
        raise ValueError(
            "Baseline passed no base-test tasks; regression rate is undefined"
        )
    regressions = sorted(baseline_passed - candidate_passed)
    recoveries = sorted(candidate_passed - baseline_passed)
    regression_rate = len(regressions) / len(baseline_passed)
    return {
        "objective_version": "evalplus-base-regression-v1",
        "base_regression_rate": regression_rate,
        "base_correctness_retention": 1.0 - regression_rate,
        "base_regression_count": len(regressions),
        "base_recovery_count": len(recoveries),
        "baseline_base_pass_count": len(baseline_passed),
        "candidate_base_pass_count": len(candidate_passed),
        "task_count": len(baseline),
        "base_regression_task_ids": regressions,
        "base_recovery_task_ids": recoveries,
    }


def _launcher_environment(work_path: Path) -> dict[str, str]:
    """Build a small host environment without forwarding user credentials."""

    launcher_home = work_path / "launcher_home"
    _secure_directory(launcher_home)
    environment = {
        "HOME": str(launcher_home),
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
    }
    # A module-provided Apptainer binary may need its shared-library path.
    for name in (
        "LD_LIBRARY_PATH",
        "APPTAINER_CACHEDIR",
        "APPTAINER_TMPDIR",
        "TMPDIR",
    ):
        value = os.environ.get(name)
        if value:
            environment[name] = value
    return environment


def _run_contained(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    timeout_seconds: int,
) -> subprocess.CompletedProcess[str]:
    """Run one container in its own process group and fail closed on timeout."""

    def tail(handle, limit: int = 8000) -> str:
        handle.flush()
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")

    # File-backed capture prevents a noisy generated program or runtime error
    # from filling the parent process's memory through an unbounded PIPE.
    with tempfile.TemporaryFile(
        mode="w+b", dir=cwd
    ) as stdout_handle, tempfile.TemporaryFile(
        mode="w+b", dir=cwd
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=stdout_handle,
            stderr=stderr_handle,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            stderr = tail(stderr_handle)
            raise RuntimeError(
                f"EvalPlus sandbox exceeded {timeout_seconds}s and was terminated; "
                f"stderr tail:\n{stderr}"
            ) from error
        return subprocess.CompletedProcess(
            command,
            process.returncode,
            tail(stdout_handle),
            tail(stderr_handle),
        )


def run_evalplus_apptainer(
    task_outputs: Iterable[dict[str, Any]],
    *,
    image: str,
    work_dir: str | Path,
    parallel: int = 8,
    apptainer_bin: str = "apptainer",
    scope: str = "full",
    timeout_seconds: int = 1800,
    dataset_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Evaluate programs in a contained, no-network EvalPlus process."""

    work_input = Path(work_dir)
    if not work_input.is_absolute():
        raise ValueError("work_dir must be absolute")
    if work_input.is_symlink():
        raise RuntimeError(
            f"Refusing to use a symlinked sandbox directory: {work_input}"
        )
    work_path = work_input.resolve()
    if scope not in EVAL_SCOPES:
        raise ValueError(f"scope must be one of {sorted(EVAL_SCOPES)}")
    if parallel < 1 or timeout_seconds < 1:
        raise ValueError("parallel and timeout_seconds must be positive")
    image_path = Path(image).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"EvalPlus image not found: {image_path}")
    _secure_directory(work_path)
    resolved_apptainer = shutil.which(apptainer_bin)
    if resolved_apptainer is None:
        raise RuntimeError(f"Apptainer executable not found: {apptainer_bin}")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for output in task_outputs:
        grouped.setdefault(str(output["dataset"]), []).append(output)
    unsupported = sorted(set(grouped) - {"humaneval", "mbpp"})
    if unsupported:
        raise ValueError(f"Unsupported EvalPlus datasets: {unsupported}")
    if not grouped:
        raise ValueError("No task outputs were provided for EvalPlus execution")
    resolved_dataset_paths = _validate_dataset_paths(dataset_paths, grouped)

    worker_dir = Path(__file__).resolve().parent
    all_results: dict[str, Any] = {}
    for dataset, outputs in grouped.items():
        sample_name = f"samples_{dataset}.jsonl"
        result_name = f"results_{dataset}.json"
        sample_path = work_path / sample_name
        result_path = work_path / result_name
        _write_samples(outputs, sample_path)
        result_path.unlink(missing_ok=True)
        command = [
            resolved_apptainer,
            "exec",
            "--cleanenv",
            "--containall",
            "--no-eval",
            "--no-home",
            "--no-privs",
            "--no-mount",
            (
                "hostfs,cwd,bind-paths,/etc/hosts,/etc/resolv.conf,"
                "/etc/localtime"
            ),
            "--net",
            "--network",
            "none",
            "--pwd",
            "/work",
            "--env",
            (
                # Apptainer deliberately refuses HOME overrides under
                # --no-home. EvalPlus honors XDG_CACHE_HOME, so direct its
                # ground-truth cache into the mounted per-run work directory.
                "XDG_CACHE_HOME=/work/tmp,TMPDIR=/work/tmp,PYTHONNOUSERSITE=1,"
                "PYTHONDONTWRITEBYTECODE=1,CUDA_VISIBLE_DEVICES=,"
                "OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1,MKL_NUM_THREADS=1,"
                "HUMANEVAL_OVERRIDE_PATH=/evalplus_data/humaneval.jsonl,"
                "MBPP_OVERRIDE_PATH=/evalplus_data/mbpp.jsonl"
            ),
            "--bind",
            _bind_spec(work_path, "/work"),
            "--bind",
            _bind_spec(worker_dir, "/ecad_worker", "ro"),
            "--bind",
            _bind_spec(
                resolved_dataset_paths[dataset],
                f"/evalplus_data/{dataset}.jsonl",
                "ro",
            ),
            str(image_path),
            "python",
            "/ecad_worker/evalplus_worker.py",
            "--dataset",
            dataset,
            "--samples",
            f"/work/{sample_name}",
            "--output",
            f"/work/{result_name}",
            "--parallel",
            str(parallel),
            "--scope",
            scope,
        ]
        _secure_directory(work_path / "home")
        _secure_directory(work_path / "tmp")
        completed = _run_contained(
            command,
            cwd=work_path,
            environment=_launcher_environment(work_path),
            timeout_seconds=timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"EvalPlus failed for {dataset} with exit code "
                f"{completed.returncode}:\n{completed.stderr[-8000:]}"
            )
        if not result_path.is_file() or result_path.is_symlink():
            raise RuntimeError(
                f"EvalPlus did not create a regular result file: {result_path}"
            )
        dataset_result = json.loads(result_path.read_text(encoding="utf-8"))
        if dataset_result.get("scope") != scope:
            raise RuntimeError(
                f"EvalPlus worker returned scope {dataset_result.get('scope')!r}; "
                f"expected {scope!r}"
            )
        if dataset_result.get("dataset") != dataset:
            raise RuntimeError(
                f"EvalPlus worker returned dataset "
                f"{dataset_result.get('dataset')!r}; expected {dataset!r}"
            )
        expected_task_ids = {str(output["task_id"]) for output in outputs}
        result_task_ids = {
            str(task["task_id"]) for task in dataset_result.get("tasks", [])
        }
        if result_task_ids != expected_task_ids:
            raise RuntimeError(
                f"EvalPlus {dataset} result task IDs do not match the samples"
            )
        dataset_result["aggregate"] = _aggregate(
            dataset_result["tasks"], scope=scope
        )
        all_results[dataset] = dataset_result

    task_results = [
        result
        for dataset_result in all_results.values()
        for result in dataset_result["tasks"]
    ]
    return {
        "scope": scope,
        "sandbox": {
            "profile": SANDBOX_PROFILE,
            "network": "none",
            "containall": True,
            "clean_environment": True,
            "shell_evaluation": False,
            "privilege_escalation": False,
            "admin_bind_paths": False,
            "gpu_visible": False,
            "timeout_seconds_per_dataset": timeout_seconds,
        },
        "datasets": all_results,
        "aggregate": _aggregate(task_results, scope=scope),
    }
