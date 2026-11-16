"""Masked-row smoothness study: are CADD's masked rows more reusable than DiffuCoder's?

Stock, uncached generation with forward hooks. At every denoising step and
layer the hooks keep the K (``k_proj``, before RoPE), V (``v_proj``) and hidden
state (decoder-layer output) of every position. For each lag k in LAGS the
study measures

    change = ||x_t - x_{t-k}|| / ||x_t||

per row, and averages it over three row groups: masked at step t, committed
(decoded by step t-k) and prompt. The design, the registered metric and the
stop rule are in MASKED_ROW_SMOOTHNESS_STUDY.md.

Arms:

  cadd_on      apple/CADD-Base-7B, CADD sampler (use_cadd, weighted)
  cadd_off     the same weights and code with use_cadd=False (z = 0)
  diffucoder_base  DiffuCoder-7B-Base (the model CADD descends from), native
                   sampler; the registered comparator
  diffucoder   DiffuCoder-7B-cpGRPO, native sampler; secondary

Two steps:

  run --arm A     GPU. Generates the tasks and saves one .npz per task.
                  Finished tasks are skipped on resubmission.
  report          No model. Compares the arms and applies the stop rule.

Results go to <results root>/<run name>/<GPU type>/<arm>/. Arms must be
compared on one GPU type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

LAGS = (1, 4, 8)
GROUPS = ("masked", "committed", "prompt")
KINDS = ("K", "V", "H")
DEEP_LAYERS = range(14, 28)

RUN_NAME = "masked_row_smoothness_v1"
RESULTS_ROOT = Path("/scratch/gqs7zr/ai4mat/ecad/results")
TASKS_FILE = Path(__file__).resolve().parents[2] / "data/code/humaneval_mbpp_64_strict_python_seed0.jsonl"
CADD_SNAPSHOT = Path(
    "/scratch/gqs7zr/cache/huggingface/hub/models--apple--CADD-Base-7B/snapshots/19cf632dc7e0bcf808e8f01aeda988046536f40e"
)
CADD_CODE = Path(__file__).resolve().parents[2] / "ml-cadd"
CADD_OVERLAY = Path("/scratch/gqs7zr/model_checkpoints/cadd_base_7b_overlay")
DIFFUCODER_PATH = Path(
    "/scratch/gqs7zr/model_checkpoints/models--apple--DiffuCoder-7B-cpGRPO/snapshots/98bc42bfd871eeec92fd0a8439ab34f75b405204"
)
DIFFUCODER_BASE_PATH = Path(
    "/scratch/gqs7zr/cache/huggingface/hub/models--apple--DiffuCoder-7B-Base/snapshots/a2c33054cbe99b9ab7af074f09b1d13943157ff1"
)
CADD_CODE_FILES = ("configuration_dream.py", "modeling_dream.py", "generation_utils.py")

ARMS: dict[str, dict[str, Any]] = {
    "cadd_on": {"model": "cadd", "generate": {"use_cadd": True, "cadd_sampling_mode": "weighted"}},
    "cadd_off": {"model": "cadd", "generate": {"use_cadd": False}},
    "diffucoder_base": {"model": "diffucoder_base", "generate": {}},
    "diffucoder": {"model": "diffucoder", "generate": {}},
}
MODEL_PATHS = {"diffucoder": DIFFUCODER_PATH, "diffucoder_base": DIFFUCODER_BASE_PATH}
COMPARATOR = "diffucoder_base"

STEPS = 256
GEN_LENGTH = 256


def build_cadd_overlay() -> Path:
    """The checkpoint's own generation_utils.py has no CADD sampler.

    Link every checkpoint file, except the three code files, which come from
    the unmodified ml-cadd copy.
    """

    CADD_OVERLAY.mkdir(parents=True, exist_ok=True)
    for source in CADD_SNAPSHOT.iterdir():
        if source.name in CADD_CODE_FILES:
            continue
        link = CADD_OVERLAY / source.name
        if not link.is_symlink():
            link.symlink_to(source.resolve())
    for name in CADD_CODE_FILES:
        link = CADD_OVERLAY / name
        target = (CADD_CODE / name).resolve()
        if link.is_symlink() and link.resolve() != target:
            link.unlink()
        if not link.is_symlink():
            link.symlink_to(target)
    return CADD_OVERLAY


def load_tasks(max_tasks: int | None) -> list[dict[str, Any]]:
    tasks = [json.loads(line) for line in TASKS_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
    return tasks[:max_tasks] if max_tasks else tasks


class StepRecorder:
    """Forward hooks plus the token hook: records change statistics online.

    Only the last max(LAGS) forwards are kept, so memory does not grow with
    the number of steps. The token hook runs after each step's unmasking, so
    the mask for the next forward is read from the tokens it returns.
    """

    def __init__(self, model: Any, mask_token_id: int, prompt_length: int, steps: int) -> None:
        import torch

        self.torch = torch
        self.layers = list(model.model.layers)
        self.mask_token_id = mask_token_id
        self.prompt_length = prompt_length
        self.steps = steps
        self.current: dict[str, list[Any]] = {kind: [None] * len(self.layers) for kind in KINDS}
        self.history: dict[int, tuple[Any, dict[str, list[Any]]]] = {}
        self.order: deque[int] = deque()
        self.mask: Any = None
        self.step = 0
        self.rel = np.full((steps, len(LAGS), len(GROUPS), len(KINDS), len(self.layers)), np.nan, np.float32)
        self.counts = np.zeros((steps, len(LAGS), len(GROUPS)), np.int32)
        self.n_masked = np.zeros(steps, np.int32)
        self.handles = []
        for index, layer in enumerate(self.layers):
            self.handles.append(layer.self_attn.k_proj.register_forward_hook(self._store("K", index)))
            self.handles.append(layer.self_attn.v_proj.register_forward_hook(self._store("V", index)))
            self.handles.append(layer.register_forward_hook(self._store_hidden(index)))

    def _store(self, kind: str, index: int):
        def hook(module, inputs, output):
            self.current[kind][index] = output[0].detach()

        return hook

    def _store_hidden(self, index: int):
        def hook(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.current["H"][index] = hidden[0].detach()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def token_hook(self, step, tokens, logits):
        """Called with step None before the first forward and after every step."""

        if step is not None:
            self._finalize(step)
        self.mask = tokens[0] == self.mask_token_id
        return tokens

    def _finalize(self, step: int) -> None:
        torch = self.torch
        mask = self.mask
        self.n_masked[step] = int(mask.sum())
        length = mask.shape[0]
        positions = torch.arange(length, device=mask.device)
        prompt = positions < self.prompt_length
        generated = ~prompt
        snapshot = {kind: list(values) for kind, values in self.current.items()}
        for lag_index, lag in enumerate(LAGS):
            if step - lag < 0:
                continue
            previous_mask, previous = self.history[step - lag]
            groups = torch.stack([mask, generated & ~previous_mask, prompt]).float()
            counts = groups.sum(dim=1)
            self.counts[step, lag_index] = counts.cpu().numpy().astype(np.int32)
            weights = groups / counts.clamp_min(1.0).unsqueeze(1)
            for kind_index, kind in enumerate(KINDS):
                per_layer = []
                for layer in range(len(self.layers)):
                    now = snapshot[kind][layer].float()
                    before = previous[kind][layer].float()
                    change = (now - before).norm(dim=-1) / now.norm(dim=-1).clamp_min(1e-6)
                    per_layer.append(weights @ change)
                self.rel[step, lag_index, :, kind_index, :] = torch.stack(per_layer, dim=1).cpu().numpy()
        self.history[step] = (mask.clone(), snapshot)
        self.order.append(step)
        while len(self.order) > max(LAGS):
            del self.history[self.order.popleft()]


def run_arm(arm: str, max_tasks: int | None, seed: int) -> None:
    import torch

    from language_models.diffucoder_loader import load_diffucoder_model_and_tokenizer

    from dlm_kv_policy.code_correctness_and_latency_evaluator import gpu_folder_name

    spec = ARMS[arm]
    output_dir = RESULTS_ROOT / RUN_NAME / gpu_folder_name() / arm
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir}", flush=True)
    tasks = load_tasks(max_tasks)
    pending = [index for index in range(len(tasks)) if not (output_dir / f"task_{index:03d}.npz").exists()]
    if not pending:
        print("All tasks already finished", flush=True)
        return

    path = build_cadd_overlay() if spec["model"] == "cadd" else MODEL_PATHS[spec["model"]]
    model, tokenizer = load_diffucoder_model_and_tokenizer(path, device="cuda", dtype="bfloat16")
    mask_token_id = int(model.config.mask_token_id)
    for index in pending:
        task = tasks[index]
        prompt_ids = tokenizer(task["original_prompt"], return_tensors="pt", add_special_tokens=False)["input_ids"].cuda()
        recorder = StepRecorder(model, mask_token_id, prompt_ids.shape[1], STEPS)
        torch.manual_seed(seed + index)
        torch.cuda.manual_seed_all(seed + index)
        start = perf_counter()
        try:
            with torch.no_grad():
                output = model.diffusion_generate(
                    prompt_ids,
                    attention_mask=None,
                    max_new_tokens=GEN_LENGTH,
                    steps=STEPS,
                    temperature=0.0,
                    top_p=1.0,
                    alg="entropy",
                    alg_temp=0.0,
                    return_dict_in_generate=True,
                    generation_tokens_hook_func=recorder.token_hook,
                    **spec["generate"],
                )
        finally:
            recorder.close()
        seconds = perf_counter() - start
        sequence = output.sequences[0]
        expected = np.arange(GEN_LENGTH, 0, -1)
        valid = bool(np.array_equal(recorder.n_masked, expected))
        text = tokenizer.decode(sequence[prompt_ids.shape[1] :], skip_special_tokens=True)
        np.savez_compressed(
            output_dir / f"task_{index:03d}.npz",
            rel=recorder.rel,
            counts=recorder.counts,
            n_masked=recorder.n_masked,
            prompt_hash=hashlib.sha256(prompt_ids.cpu().numpy().tobytes()).hexdigest(),
            valid=valid,
            seconds=seconds,
            prompt_length=prompt_ids.shape[1],
            output_ids=sequence.cpu().numpy(),
        )
        (output_dir / f"task_{index:03d}.json").write_text(
            json.dumps({"task_id": task["task_id"], "valid": valid, "seconds": seconds, "output": text}, indent=2),
            encoding="utf-8",
        )
        print(f"task {index:03d} {task['task_id']} valid={valid} {seconds:.1f}s", flush=True)
        del recorder
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# Report (no model)
# --------------------------------------------------------------------------


def load_arm(arm_dir: Path) -> dict[str, np.ndarray]:
    files = sorted(arm_dir.glob("task_*.npz"))
    if not files:
        raise FileNotFoundError(f"No finished tasks in {arm_dir}")
    loaded = [np.load(file) for file in files]
    return {
        "rel": np.stack([item["rel"] for item in loaded]),
        "counts": np.stack([item["counts"] for item in loaded]),
        "n_masked": np.stack([item["n_masked"] for item in loaded]),
        "valid": np.array([bool(item["valid"]) for item in loaded]),
        "prompt_hash": np.array([str(item["prompt_hash"]) for item in loaded]),
        "task_index": np.array([int(file.stem.split("_")[1]) for file in files]),
    }


def weighted_by_task(data: dict[str, np.ndarray], lag: int, group: int, steps: slice | None = None) -> np.ndarray:
    """Per task, per (kind, layer): row-weighted mean change over steps.

    The weight of a step is the number of rows of the group at that step, so
    the value is the mean over every (row, step) pair.
    """

    lag_index = LAGS.index(lag)
    rel = data["rel"][:, :, lag_index, group]
    weight = data["counts"][:, :, lag_index, group].astype(np.float64)
    if steps is not None:
        rel, weight = rel[:, steps], weight[:, steps]
    usable = weight > 0
    weight = np.where(usable, weight, 0.0)
    filled = np.where(usable[:, :, None, None], rel, 0.0)
    total = weight.sum(axis=1)
    return (weight[:, :, None, None] * filled).sum(axis=1) / np.maximum(total, 1.0)[:, None, None]


def primary_per_task(data: dict[str, np.ndarray], lag: int = 1, kinds: tuple[int, ...] = (0, 1)) -> np.ndarray:
    values = weighted_by_task(data, lag, 0)
    return values[:, list(kinds)][:, :, list(DEEP_LAYERS)].mean(axis=(1, 2))


def bootstrap_ratio(numerator: np.ndarray, denominator: np.ndarray, resamples: int = 10_000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    count = len(numerator)
    picks = rng.integers(0, count, size=(resamples, count))
    ratios = numerator[picks].mean(axis=1) / denominator[picks].mean(axis=1)
    return {
        "ratio": float(numerator.mean() / denominator.mean()),
        "ci_low": float(np.percentile(ratios, 2.5)),
        "ci_high": float(np.percentile(ratios, 97.5)),
    }


def arm_summary(data: dict[str, np.ndarray]) -> dict[str, Any]:
    summary: dict[str, Any] = {"tasks": int(len(data["valid"])), "all_valid": bool(data["valid"].all())}
    for lag in LAGS:
        for group_index, group in enumerate(GROUPS):
            per_layer = weighted_by_task(data, lag, group_index).mean(axis=0)
            summary[f"lag{lag}_{group}_per_layer"] = {kind: per_layer[k].round(5).tolist() for k, kind in enumerate(KINDS)}
        summary[f"lag{lag}_masked_primary"] = float(primary_per_task(data, lag).mean())
        summary[f"lag{lag}_masked_primary_hidden"] = float(primary_per_task(data, lag, (2,)).mean())
        shallow = weighted_by_task(data, lag, 0)[:, :2][:, :, :14].mean()
        summary[f"lag{lag}_masked_shallow_KV"] = float(shallow)
    bins = []
    for low in range(0, STEPS, STEPS // 4):
        window = slice(max(low, 1), low + STEPS // 4)
        values = weighted_by_task(data, 1, 0, window)[:, :2][:, :, list(DEEP_LAYERS)].mean()
        rows = data["counts"][:, window, 0, 0].sum() / max(data["counts"][:, 1:, 0, 0].sum(), 1)
        bins.append({"s_high": round(1 - low / STEPS, 2), "s_low": round(1 - (low + STEPS // 4) / STEPS, 2),
                     "deep_KV_change": float(values), "share_of_masked_rows": float(rows)})
    summary["lag1_masked_deep_KV_by_noise_level"] = bins
    return summary


def report(run_dir: Path) -> None:
    arms = {arm: load_arm(run_dir / arm) for arm in ARMS if (run_dir / arm).exists()}
    missing = [arm for arm in ARMS if arm not in arms]
    if missing:
        print(f"Missing arms: {missing}")
    result: dict[str, Any] = {"arms": {arm: arm_summary(data) for arm, data in arms.items()}}

    shared = None
    for data in arms.values():
        indices = set(data["task_index"].tolist())
        shared = indices if shared is None else shared & indices
    shared_list = sorted(shared or [])
    aligned = {
        arm: {key: value[np.isin(data["task_index"], shared_list)] for key, value in data.items()} for arm, data in arms.items()
    }
    hashes = {arm: data["prompt_hash"].tolist() for arm, data in aligned.items()}
    result["prompts_identical_across_arms"] = len({tuple(value) for value in hashes.values()}) <= 1
    result["comparison_tasks"] = len(shared_list)

    comparisons: dict[str, Any] = {}
    pairs = [
        ("cadd_on", COMPARATOR), ("cadd_on", "cadd_off"), ("cadd_on", "diffucoder"),
        ("diffucoder_base", "cadd_off"), ("diffucoder", "diffucoder_base"),
    ]
    for numerator, denominator in pairs:
        if numerator in aligned and denominator in aligned:
            for lag in LAGS:
                comparisons[f"{numerator}/{denominator} lag{lag}"] = bootstrap_ratio(
                    primary_per_task(aligned[numerator], lag), primary_per_task(aligned[denominator], lag)
                )
    result["ratios_masked_deep_KV"] = comparisons

    registered = comparisons.get(f"cadd_on/{COMPARATOR} lag1")
    if registered is not None:
        valid = all(summary["all_valid"] for summary in result["arms"].values()) and result["prompts_identical_across_arms"]
        passes = registered["ratio"] <= 0.80 and registered["ci_high"] < 1.0
        result["decision"] = {
            "registered_ratio": registered,
            "runs_valid": valid,
            "rule": "continue only if ratio <= 0.80 and the 95% interval's upper end < 1.0",
            "verdict": ("CONTINUE" if passes else "STOP") if valid else "INVALID: fix the runs first",
        }

    (run_dir / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"{'arm':<11}{'lag1 primary':>13}{'lag4':>9}{'lag8':>9}{'shallow(l1)':>13}{'hidden(l1)':>12}")
    for arm, summary in result["arms"].items():
        print(f"{arm:<11}{summary['lag1_masked_primary']:>13.4f}{summary['lag4_masked_primary']:>9.4f}"
              f"{summary['lag8_masked_primary']:>9.4f}{summary['lag1_masked_shallow_KV']:>13.4f}"
              f"{summary['lag1_masked_primary_hidden']:>12.4f}")
    for name, ratio in comparisons.items():
        print(f"{name:<30} ratio {ratio['ratio']:.3f}  95% [{ratio['ci_low']:.3f}, {ratio['ci_high']:.3f}]")
    if "decision" in result:
        print(f"Verdict: {result['decision']['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--max-tasks", type=int)
    run.add_argument("--seed", type=int, default=0)
    rep = sub.add_parser("report")
    rep.add_argument("--gpu-folder", required=True)
    args = parser.parse_args()
    if args.command == "run":
        run_arm(args.arm, args.max_tasks, args.seed)
    else:
        report(RESULTS_ROOT / RUN_NAME / args.gpu_folder)


if __name__ == "__main__":
    main()
