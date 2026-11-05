"""Steps 0 and 1 of the masked-row reuse study: where does the dense model commit?

Masked rows can be reused only if they are not about to be committed: a reused
row has no fresh logits. This logs the dense greedy trajectory and measures how
predictable commit positions are, then prices candidate policies from the same
logs. The design, the selection rule, the decision rule and the truncation
baseline are in MASKED_ROW_REUSE_STUDY.md.

  run      GPU. Dense greedy generation that logs, per step, the entropy
           confidence and top-1 token of every masked position and the committed
           positions. Finished tasks are skipped on resubmission. A task whose
           final tokens differ from the earlier forced-trajectory replay is
           flagged and left out of the report.
  report   No model. Locality, stale-confidence rank, coverage of the seven
           protected-set rules at lags 1-8, the pre-registered selection, and
           Step 1 compute accounting (floor, truncation, reuse, reuse+truncation).
           With --timing it also projects wall-clock speedups.

Results go to <run>/<GPU type>/commit_position_study/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

LAGS = tuple(range(1, 9))
M_VALUES = (16, 32)
PADDING_TOP = 4
VARIANTS = ("top", "top±1", "top+recent", "recent", "split", "split±1", "split+recent")
REUSE_WINDOWS = (2, 4, 8)
COVERAGE_TARGET = 0.99
RANK_THRESHOLDS = (1, 4, 16, 32, 64)
CLASSES = ("answer", "padding")


# --------------------------------------------------------------------------
# Step 0 logging (GPU)
# --------------------------------------------------------------------------


def run_task(model, sampler, tokenizer, task, index, generation, device, top_p_logits) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F

    from ecad.language_models.diffucoder_loader import prepare_diffucoder_prompt_batch

    from .code_correctness_and_latency_evaluator import seed_everything
    from .denoising_token_state import DenoisingTokenState
    from .forced_trajectory_replay import entropy_confidence

    seed_everything(generation.seed + index)
    prompt, attention_mask = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], device)
    mask_token_id = int(model.config.mask_token_id)
    _, model_attention_mask, position_ids = sampler._attention_inputs(prompt, attention_mask, generation.gen_length)
    x = F.pad(prompt, (0, generation.gen_length), value=mask_token_id)
    state = DenoisingTokenState.create(prompt.shape[1], x.shape[1], attention_mask, mask_token_id)
    timesteps = torch.linspace(1, 1e-12, generation.steps + 1, device=x.device)
    prompt_length = prompt.shape[1]
    steps, gen_length = generation.steps, generation.gen_length
    confidence = np.full((steps, gen_length), np.nan, np.float32)
    top1 = np.full((steps, gen_length), -1, np.int32)
    committed = np.zeros((steps, gen_length), bool)

    for step in range(steps):
        mask_index = x.eq(mask_token_id)
        logits = sampler.backend.forward_mask_logits(
            input_ids=x, attention_mask=model_attention_mask, position_ids=position_ids,
            state=state, step=step, total_steps=steps,
        ).mask_logits
        positions = mask_index[0].nonzero()[:, 0] - prompt_length
        scores = entropy_confidence(logits, generation.temperature, generation.top_p, top_p_logits)
        confidence[step, positions.cpu().numpy()] = scores.float().cpu().numpy()
        top1[step, positions.cpu().numpy()] = logits.argmax(dim=-1).cpu().numpy().astype(np.int32)
        previous = x.clone()
        sampler.commit_tokens(
            x, mask_index, logits, step=step, steps=steps, timesteps=timesteps,
            temperature=generation.temperature, top_p=generation.top_p, top_k=generation.top_k,
            algorithm=generation.algorithm, algorithm_temperature=generation.algorithm_temperature,
        )
        state.update_after_sampling(previous, x, step)
        changed = (previous[0] != x[0]).nonzero()[:, 0] - prompt_length
        committed[step, changed.cpu().numpy()] = True
    return {
        "confidence": confidence, "top1": top1, "committed": committed,
        "dense_token_ids": x[0, prompt_length:].cpu().numpy().astype(np.int32),
        "prompt_length": prompt_length,
    }


def run(config_path: Path, max_tasks: int | None) -> None:
    from ecad.language_models.code_tasks_and_response_processing import load_code_task_dataset
    from ecad.language_models.diffucoder_loader import load_diffucoder_model_and_tokenizer

    from .code_correctness_and_latency_evaluator import gpu_folder_name
    from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
    from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
    from .forced_trajectory_replay import load_top_p_logits
    from .kv_policy_experiment_config import load_kv_policy_experiment_config

    config = load_kv_policy_experiment_config(config_path)
    generation = config.generation
    if generation.algorithm != "entropy" or generation.algorithm_temperature != 0 or generation.temperature != 0:
        raise ValueError("The study assumes greedy decoding with DiffuCoder's entropy order")
    tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
    if max_tasks:
        tasks = tasks[:max_tasks]
    gpu = gpu_folder_name()
    output_dir = config.run.run_dir / gpu / "commit_position_study"
    replay_dir = config.run.run_dir / gpu / "forced_trajectory_replay" / "tasks"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results: {output_dir}", flush=True)
    pending = [i for i in range(len(tasks)) if not (output_dir / f"task_{i:03d}.npz").exists()]
    if not pending:
        print("All tasks already finished", flush=True)
        return

    model, tokenizer = load_diffucoder_model_and_tokenizer(
        config.model.path, device=config.model.device, dtype=config.model.dtype,
        local_files_only=config.model.local_files_only,
    )
    sampler = DiffuCoderDenoisingSampler(DenseDiffuCoderBaseline(model))
    top_p_logits = load_top_p_logits(model)
    stop_ids = sorted({t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None})
    for index in pending:
        result = run_task(model, sampler, tokenizer, tasks[index], index, generation, config.model.device, top_p_logits)
        saved = replay_dir / f"task_{index:03d}.json"
        matches = -1  # no replay to compare with
        if saved.is_file():
            matches = int(json.loads(saved.read_text())["dense_token_ids"] == result["dense_token_ids"].tolist())
        np.savez_compressed(
            output_dir / f"task_{index:03d}.npz", **result, stop_ids=np.array(stop_ids),
            matches_replay=matches, task_id=tasks[index].task_id,
        )
        print(f"task {index:03d} {tasks[index].task_id} matches_replay={matches}", flush=True)


# --------------------------------------------------------------------------
# Analysis (numpy only)
# --------------------------------------------------------------------------


def answer_length(token_ids: np.ndarray, stop_ids: set[int]) -> int:
    for index, token in enumerate(token_ids.tolist()):
        if token in stop_ids:
            return index
    return len(token_ids)


class TaskLog:
    """One task's dense trajectory, with the masks and classes derived from it."""

    def __init__(self, data: dict[str, Any], index: int) -> None:
        self.index = index
        self.confidence = data["confidence"]
        self.top1 = data["top1"]
        self.committed = data["committed"].astype(bool)
        self.tokens = data["dense_token_ids"]
        self.prompt_length = int(data["prompt_length"])
        self.stop_ids = set(int(t) for t in data["stop_ids"])
        self.steps, self.length = self.committed.shape
        done_before = np.cumsum(self.committed, axis=0) - self.committed
        self.done_before = done_before > 0  # committed strictly before step t
        self.masked = ~self.done_before
        self.answer_end = answer_length(self.tokens, self.stop_ids)
        self.is_stop = np.isin(self.tokens, list(self.stop_ids))

    def events(self):
        """(step, position) of every commit after step 0, where reuse is possible."""

        steps, positions = np.nonzero(self.committed)
        return [(int(t), int(p)) for t, p in zip(steps, positions) if t >= 1]

    def event_class(self, position: int) -> int:
        return 0 if position < self.answer_end else 1


def dilate(flags: np.ndarray) -> np.ndarray:
    out = flags.copy()
    out[1:] |= flags[:-1]
    out[:-1] |= flags[1:]
    return out


def top_positions(candidates: np.ndarray, scores: np.ndarray, count: int) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")[:count]
    return candidates[order]


def protected_set(log: TaskLog, step: int, lag: int, variant: str, m: int) -> np.ndarray:
    """Boolean mask over positions: rows recomputed at ``step`` under the rule."""

    masked = log.masked[step]
    stale = max(step - lag, 0)
    candidates = np.flatnonzero(masked)
    scores = log.confidence[stale][candidates]
    result = np.zeros(log.length, bool)
    recent = np.zeros(log.length, bool)
    if variant in ("top+recent", "recent", "split+recent"):
        recent = dilate(log.committed[stale:step].any(axis=0)) & masked
    if variant in ("top", "top±1", "top+recent"):
        result[top_positions(candidates, scores, m)] = True
    elif variant in ("split", "split±1", "split+recent"):
        padding_like = np.isin(log.top1[stale][candidates], list(log.stop_ids))
        result[top_positions(candidates[~padding_like], scores[~padding_like], m)] = True
        result[top_positions(candidates[padding_like], scores[padding_like], PADDING_TOP)] = True
    if variant in ("top±1", "split±1"):
        result = dilate(result) & masked
    return (result | recent) & masked


def stale_rank(log: TaskLog, step: int, position: int, lag: int) -> int:
    candidates = np.flatnonzero(log.masked[step])
    scores = log.confidence[max(step - lag, 0)][candidates]
    own = log.confidence[max(step - lag, 0)][position]
    return int((scores > own).sum()) + 1


def locality(log: TaskLog, step: int, position: int) -> tuple[int, int]:
    committed = np.flatnonzero(log.done_before[step])
    nearest = int(np.abs(committed - position).min()) if committed.size else position + 1
    nearest = min(nearest, position + 1)  # the prompt's last token counts as committed
    leftmost = int(np.flatnonzero(log.masked[step]).min())
    return nearest, position - leftmost


def analyze_logs(logs: list[TaskLog]) -> dict[str, Any]:
    """Locality, rank and coverage tables, kept per task so halves can be split."""

    tasks = len(logs)
    lags = len(LAGS)
    cover = np.zeros((tasks, lags, len(VARIANTS), len(M_VALUES), 2))
    total = np.zeros((tasks, 2))
    size_sum = np.zeros((tasks, lags, len(VARIANTS), len(M_VALUES)))
    size_count = np.zeros(tasks)
    ranks: list[list[list[int]]] = [[[] for _ in range(lags)] for _ in range(2)]
    near: list[list[int]] = [[], []]
    left: list[list[int]] = [[], []]
    for ti, log in enumerate(logs):
        step_done: set[int] = set()
        for step, position in log.events():
            cls = log.event_class(position)
            total[ti, cls] += 1
            nearest, from_left = locality(log, step, position)
            near[cls].append(nearest)
            left[cls].append(from_left)
            for li, lag in enumerate(LAGS):
                ranks[cls][li].append(stale_rank(log, step, position, lag))
                for vi, variant in enumerate(VARIANTS):
                    for mi, m in enumerate(M_VALUES):
                        prot = protected_set(log, step, lag, variant, m)
                        cover[ti, li, vi, mi, cls] += bool(prot[position])
                        if step not in step_done:
                            size_sum[ti, li, vi, mi] += prot.sum()
            if step not in step_done:
                size_count[ti] += 1
                step_done.add(step)
    return {"cover": cover, "total": total, "size_sum": size_sum, "size_count": size_count,
            "ranks": ranks, "near": near, "left": left}


def coverage(stats: dict[str, Any], task_mask: np.ndarray, lag_index: int, vi: int, mi: int, cls: int | None) -> float:
    cover = stats["cover"][task_mask][:, lag_index, vi, mi]
    total = stats["total"][task_mask]
    if cls is None:
        return float(cover.sum() / max(total.sum(), 1))
    return float(cover[:, cls].sum() / max(total[:, cls].sum(), 1))


def mean_size(stats: dict[str, Any], task_mask: np.ndarray, lag_index: int, vi: int, mi: int) -> float:
    return float(stats["size_sum"][task_mask][:, lag_index, vi, mi].sum() / max(stats["size_count"][task_mask].sum(), 1))


def select_rules(stats: dict[str, Any], indices: np.ndarray) -> dict[int, dict[str, Any]]:
    """The pre-registered rule: smallest set with >=99% answer coverage on even tasks."""

    even, odd = indices % 2 == 0, indices % 2 == 1
    chosen: dict[int, dict[str, Any]] = {}
    for window in REUSE_WINDOWS:
        li = LAGS.index(window)
        options = []
        for vi, variant in enumerate(VARIANTS):
            for mi, m in enumerate(M_VALUES):
                if coverage(stats, even, li, vi, mi, 0) >= COVERAGE_TARGET:
                    options.append((mean_size(stats, even, li, vi, mi), m, vi, mi))
        if not options:
            chosen[window] = {"window": window, "selected": None}
            continue
        size, m, vi, mi = min(options)
        chosen[window] = {
            "window": window, "selected": True, "variant": VARIANTS[vi], "m": m, "mean_size": size,
            "answer_coverage_even": coverage(stats, even, li, vi, mi, 0),
            "answer_coverage_odd": coverage(stats, odd, li, vi, mi, 0),
            "padding_coverage_even": coverage(stats, even, li, vi, mi, 1),
            "padding_coverage_odd": coverage(stats, odd, li, vi, mi, 1),
        }
    return chosen


# -- Step 1: compute accounting ---------------------------------------------


def model_shape(config_json: Path):
    from .compute_cost import ModelShape

    raw = json.loads(config_json.read_text())
    return ModelShape(raw["hidden_size"], raw["intermediate_size"], raw["num_attention_heads"],
                      raw["num_key_value_heads"], raw["num_hidden_layers"], raw["vocab_size"])


def step_rows(log: TaskLog, selection: dict[str, Any] | None, truncate: bool) -> tuple[np.ndarray, np.ndarray]:
    """Rows through the layers and rows through the LM head at every step.

    ``selection`` is None for the floor (every masked row each step); otherwise
    {"variant", "m", "window"}. New rows (committed at the previous step) are
    recomputed once. Step 0 recomputes everything.
    """

    layer_rows = np.zeros(log.steps)
    head_rows = np.zeros(log.steps)
    committed_stop = np.where(log.committed & log.is_stop[None, :], 1, 0)
    for step in range(log.steps):
        masked = log.masked[step].copy()
        new = int(log.committed[step - 1].sum()) if step else 0
        if step == 0:
            layer_rows[0] = log.prompt_length + log.length
            head_rows[0] = masked.sum()
            continue
        if truncate:
            stops = np.flatnonzero(committed_stop[:step].any(axis=0))
            if stops.size:
                cutoff = int(stops.min())
                masked[cutoff + 1 :] = False
                new = int(log.committed[step - 1][: cutoff + 1].sum())
        if selection is None:
            computed = float(masked.sum())
        else:
            prot = protected_set(log, step, selection["window"], selection["variant"], selection["m"]) & masked
            computed = float(prot.sum()) + float((masked & ~prot).sum()) / selection["window"]
        layer_rows[step] = computed + new
        head_rows[step] = computed
    return layer_rows, head_rows


def policy_flops(shape, log: TaskLog, rows: tuple[np.ndarray, np.ndarray]) -> float:
    from .compute_cost import layer_flops, lm_head_flops

    key_length = log.prompt_length + log.length
    per_layer = shape.num_layers * np.array([layer_flops(shape, max(r, 0), key_length) for r in rows[0]], dtype=np.float64)
    head = np.array([lm_head_flops(shape, max(r, 0)) for r in rows[1]], dtype=np.float64)
    return float((per_layer + head).sum())


def dense_flops(shape, log: TaskLog) -> float:
    from .compute_cost import dense_forward_flops

    return float(dense_forward_flops(shape, log.prompt_length + log.length) * log.steps)


def accounting(logs: list[TaskLog], shape, chosen: dict[int, dict[str, Any]], timing: dict[str, Any] | None):
    policies: dict[str, dict[str, Any]] = {"floor (refresh_256)": {"sel": None, "trunc": False},
                                          "truncation": {"sel": None, "trunc": True}}
    for window in REUSE_WINDOWS:
        picked = chosen.get(window, {})
        if not picked.get("selected"):
            continue
        for m in M_VALUES:
            selection = {"variant": picked["variant"], "m": m, "window": window}
            policies[f"reuse r={window} m={m} ({picked['variant']})"] = {"sel": selection, "trunc": False}
            policies[f"reuse+trunc r={window} m={m} ({picked['variant']})"] = {"sel": selection, "trunc": True}
    table: dict[str, Any] = {}
    for name, spec in policies.items():
        fractions, projected = [], {}
        for log in logs:
            rows = step_rows(log, spec["sel"], spec["trunc"])
            fractions.append(policy_flops(shape, log, rows) / dense_flops(shape, log))
            if timing:
                for batch, curve in timing["curves"].items():
                    pts = sorted((int(n), s) for n, s in curve.items())
                    xs, ys = zip(*pts)
                    seconds = float(np.interp(np.maximum(rows[0], 1), xs, ys).sum())
                    projected.setdefault(batch, []).append(log.steps * curve[str(timing["sequence_length"])] / seconds)
        table[name] = {"compute_fraction": float(np.mean(fractions))}
        if timing:
            table[name]["projected_speedup"] = {b: float(np.mean(v)) for b, v in projected.items()}
    return table


# -- report ------------------------------------------------------------------


def load_logs(directory: Path) -> tuple[list[TaskLog], list[str]]:
    logs, excluded = [], []
    for file in sorted(directory.glob("task_*.npz")):
        data = np.load(file)
        index = int(file.stem.split("_")[1])
        if int(data["matches_replay"]) == 0:
            excluded.append(str(data["task_id"]))
            continue
        logs.append(TaskLog({key: data[key] for key in data.files}, index))
    if not logs:
        raise FileNotFoundError(f"No usable tasks in {directory}")
    return logs, excluded


def quantiles(values: list[int]) -> dict[str, float]:
    if not values:
        return {}
    array = np.array(values)
    return {"n": int(array.size), "median": float(np.median(array)), "p95": float(np.percentile(array, 95)),
            "p99": float(np.percentile(array, 99)), "max": float(array.max()),
            **{f"le_{t}": float((array <= t).mean()) for t in (1, 2, 4, 8)}}


def report(directory: Path, config_json: Path | None, timing_path: Path | None) -> None:
    logs, excluded = load_logs(directory)
    indices = np.array([log.index for log in logs])
    stats = analyze_logs(logs)
    chosen = select_rules(stats, indices)
    everyone = np.ones(len(logs), bool)

    result: dict[str, Any] = {"tasks": len(logs), "excluded_not_matching_replay": excluded,
                              "commit_events": {c: int(stats["total"][:, i].sum()) for i, c in enumerate(CLASSES)}}
    result["locality"] = {c: {"distance_to_committed": quantiles(stats["near"][i]),
                              "distance_from_leftmost_masked": quantiles(stats["left"][i])} for i, c in enumerate(CLASSES)}
    result["stale_rank"] = {c: {str(lag): {**quantiles(stats["ranks"][i][li]),
                                           **{f"rank_le_{t}": float((np.array(stats["ranks"][i][li]) <= t).mean())
                                              for t in RANK_THRESHOLDS if stats["ranks"][i][li]}}
                                for li, lag in enumerate(LAGS)} for i, c in enumerate(CLASSES)}
    result["coverage"] = {}
    for li, lag in enumerate(LAGS):
        for vi, variant in enumerate(VARIANTS):
            for mi, m in enumerate(M_VALUES):
                result["coverage"][f"lag{lag} {variant} m={m}"] = {
                    "answer": coverage(stats, everyone, li, vi, mi, 0),
                    "padding": coverage(stats, everyone, li, vi, mi, 1),
                    "mean_size": mean_size(stats, everyone, li, vi, mi),
                }
    result["selection"] = chosen
    result["gate"] = {"window_2_selected": bool(chosen[2].get("selected")),
                      "window_2_holdout_ok": bool(chosen[2].get("selected") and chosen[2]["answer_coverage_odd"] >= COVERAGE_TARGET)}
    result["gate"]["verdict"] = "PASS" if all(result["gate"].values()) else "STOP"
    if config_json is not None:
        timing = json.loads(timing_path.read_text()) if timing_path else None
        result["accounting"] = accounting(logs, model_shape(config_json), chosen, timing)
    (directory / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"{len(logs)} tasks ({len(excluded)} excluded); commits: {result['commit_events']}")
    for cls in CLASSES:
        loc = result["locality"][cls]["distance_to_committed"]
        print(f"[{cls}] distance to nearest committed: median {loc.get('median')} p95 {loc.get('p95')} p99 {loc.get('p99')}")
    print("answer-position coverage by lag (m=16 / m=32):")
    for variant in VARIANTS:
        cells = []
        for lag in (1, 2, 4, 8):
            a, b = (result["coverage"][f"lag{lag} {variant} m={m}"]["answer"] for m in M_VALUES)
            cells.append(f"lag{lag} {a:.3f}/{b:.3f}")
        print(f"  {variant:<13}" + "  ".join(cells))
    for window, pick in chosen.items():
        print(f"select r={window}:", pick if not pick.get("selected") else
              f"{pick['variant']} m={pick['m']} size {pick['mean_size']:.1f} answer cov even {pick['answer_coverage_even']:.4f} odd {pick['answer_coverage_odd']:.4f}")
    if "accounting" in result:
        for name, row in result["accounting"].items():
            extra = f"  speedup {row['projected_speedup']}" if "projected_speedup" in row else ""
            print(f"  {name:<44}{row['compute_fraction']:.3f}{extra}")
    print(f"Step 0 gate: {result['gate']['verdict']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--config", type=Path, required=True)
    run_parser.add_argument("--max-tasks", type=int)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--config", type=Path, required=True)
    report_parser.add_argument("--gpu-folder", required=True)
    report_parser.add_argument("--timing", type=Path, help="batched_row_timing.py output, for projected speedups")
    args = parser.parse_args()
    if args.command == "run":
        run(args.config, args.max_tasks)
        return
    import yaml

    from .kv_policy_experiment_config import load_kv_policy_experiment_config

    config = load_kv_policy_experiment_config(args.config)
    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    report(config.run.run_dir / args.gpu_folder / "commit_position_study", Path(raw["model"]["path"]) / "config.json", args.timing)


if __name__ == "__main__":
    main()
