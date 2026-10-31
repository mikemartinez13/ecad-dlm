"""Stale K/V error study: can the error of a reused K/V row be predicted?

When a cached policy reuses a K or V row instead of recomputing it, the row
differs from what dense computes at that step. This study asks whether that
error, e = dense - cached, can be predicted from the row's cell and from the
cached vector c itself. It only measures predictability; nothing is corrected
during decoding. The design and decision rule are pre-registered in
STALE_KV_ERROR_STUDY.md.

Every policy runs the forced-trajectory replay: each step it sees the dense
model's sequence and keeps its own cache. Keys are compared before RoPE. RoPE
rotates each position differently, so it would hide an error shared across
positions.

A cell is (layer, K or V, KV head, age bucket, row type):

- age: steps since the row was last computed at that layer: 1, 2-3, 4-7,
  8-31 or 32+
- row type: prompt; answer (a response position before the first stop token
  of the final dense output); padding (the first stop token and after)

Methods, each fit per cell on the training folds and written e_hat = A c + b:
M0 none, M1 one scale (A = sI), M2 mean shift (b only), M3 per-dimension
affine (diagonal A), M4 full affine with ridge.

Four steps. Folds split the tasks, balanced between HumanEval+ and MBPP+:

  collect --fold K    GPU. Replays fold K's tasks and saves running sums.
  fit                 Fits every method for each fold from the other folds.
  evaluate --fold K   GPU. Replays fold K's tasks again with fold K's fit
                      and saves one small file of held-out sums per task.
  report              Writes the CSVs and the decision. No model.

Usage (from the repository root; collect and evaluate inside a GPU allocation):

    python -m dlm_kv_policy.stale_kv_error_study collect \\
        --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml --fold 0
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

from ecad.language_models.code_tasks_and_response_processing import load_code_task_dataset
from ecad.language_models.diffucoder_loader import (
    load_diffucoder_model_and_tokenizer,
    prepare_diffucoder_prompt_batch,
)

from .code_correctness_and_latency_evaluator import gpu_folder_name, seed_everything
from .decoding_oracles import answer_length
from .dense_diffucoder_baseline import DenseDiffuCoderBaseline
from .denoising_token_state import DenoisingTokenCategory
from .diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .forced_trajectory_replay import load_top_p_logits, replay_task
from .kv_policy_experiment_config import load_kv_policy_experiment_config

POLICIES = ("refresh_4", "refresh_32", "refresh_256")
# refresh_4 is reported, but its ages are too short to count toward the decision.
DECISION_POLICIES = ("refresh_32", "refresh_256")
# Sanity check 1: recomputes every row each step, so it has no stale rows.
CONTROL_POLICY = "full_refresh_every_step"
FOLDS = 4
FOLD_SEED = 0
# Lower edges of the age buckets after the first; bucket 0 is age 1.
AGE_EDGES = (2, 4, 8, 32)
AGE_BUCKETS = ("1", "2-3", "4-7", "8-31", "32+")
ROW_TYPES = ("prompt", "answer", "padding")
COMBOS = len(AGE_BUCKETS) * len(ROW_TYPES)  # combo = age bucket * 3 + row type
KINDS = ("K", "V")
METHODS = ("M1", "M2", "M3", "M4")  # M0 predicts zero; its residual is the error
RIDGE = 0.01  # M4: lambda = RIDGE * tr(centered covariance of c) / head_dim
SUBSPACE_RANK = 8
# A stale row's layer-0 K/V depends only on its own token, which is fixed once
# committed, so its error is rounding between the two forward paths. Layer 0
# appears in the breakdown but in no score.
SCORED_FIRST_LAYER = 1
MIN_ROWS_FOR_S1 = 100
BOOTSTRAP_RESAMPLES = 10_000
# Decision thresholds from STALE_KV_ERROR_STUDY.md. Do not change them after results exist.
MIN_FER = 0.20
MAX_S1_RELATIVE_ERROR = 0.25
MIN_S2_FRACTION = 0.25
# Sanity check 2: every stale row of one head and two layers, for the first
# two tasks of fold 0, is saved during evaluate.
RAW_SAMPLE_TASKS = 2
RAW_SAMPLE_POLICY = "refresh_32"
RAW_SAMPLE_HEAD = 0
RAW_SAMPLE_MAX_RELATIVE_ERROR = 1e-4


def assign_folds(task_ids: list[str]) -> list[int]:
    """Fold of each task: shuffled with a fixed seed within each dataset, then dealt out."""

    rng = random.Random(FOLD_SEED)
    folds = [0] * len(task_ids)
    for dataset in sorted({task_id.split("/")[0] for task_id in task_ids}):
        indices = [index for index, task_id in enumerate(task_ids) if task_id.split("/")[0] == dataset]
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            folds[index] = position % FOLDS
    return folds


def row_types(prompt_length: int, dense_token_ids: list[int], stop_ids: set[int], device: Any) -> torch.Tensor:
    """ROW_TYPES index of every position."""

    types = torch.ones(prompt_length + len(dense_token_ids), dtype=torch.long, device=device)
    types[:prompt_length] = ROW_TYPES.index("prompt")
    types[prompt_length + answer_length(dense_token_ids, stop_ids) :] = ROW_TYPES.index("padding")
    return types


# Capturing pre-RoPE K/V -------------------------------------------------------


class PreRopeKV:
    """A policy's pre-RoPE K and V of every row, as last computed: [heads, length, head_dim] per layer."""

    def __init__(self, backend: DiffuCoderActiveTokenKVBackend) -> None:
        self.backend = backend
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        backend.kv_observer = self.record

    def reset(self, length: int) -> None:
        parameter = next(self.backend.model.parameters())
        shape = (self.backend.num_kv_heads, length, self.backend.head_dim)
        self.keys = [parameter.new_zeros(shape) for _ in range(self.backend.num_layers)]
        self.values = [parameter.new_zeros(shape) for _ in range(self.backend.num_layers)]

    def record(self, layer: int, positions: torch.Tensor, keys: torch.Tensor, values: torch.Tensor) -> None:
        self.keys[layer].index_copy_(1, positions, keys[0])
        self.values[layer].index_copy_(1, positions, values[0])


class DenseWithPreRopeKV:
    """The dense backend, also recording every layer's pre-RoPE K/V of all rows."""

    def __init__(self, backend: Any, num_kv_heads: int, head_dim: int) -> None:
        self.backend = backend
        self.model = backend.model
        self.head_shape = (num_kv_heads, head_dim)
        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    @property
    def flops(self) -> int:
        return self.backend.flops

    def forward_mask_logits(self, **inputs: Any) -> Any:
        layers = self.model.model.layers
        keys: list[Any] = [None] * len(layers)
        values: list[Any] = [None] * len(layers)

        def recorder(outputs: list[Any], layer_id: int) -> Callable:
            def hook(module, args, output) -> None:
                outputs[layer_id] = output[0].view(output.shape[1], *self.head_shape).transpose(0, 1)

            return hook

        handles = [layer.self_attn.k_proj.register_forward_hook(recorder(keys, i)) for i, layer in enumerate(layers)]
        handles += [layer.self_attn.v_proj.register_forward_hook(recorder(values, i)) for i, layer in enumerate(layers)]
        try:
            result = self.backend.forward_mask_logits(**inputs)
        finally:
            for handle in handles:
                handle.remove()
        if any(rows is None or rows.shape[1] != inputs["input_ids"].shape[1] for rows in keys + values):
            raise RuntimeError("The dense forward did not compute K/V for every row")
        self.keys, self.values = keys, values
        return result


Sink = Callable[[str, int, int, torch.Tensor, torch.Tensor, torch.Tensor], None]


class StaleRowReplay:
    """Forced-trajectory replay that hands each policy's stale K/V rows to a sink.

    The sink gets (policy, layer, kind, c, e, combo): c is the cached pre-RoPE
    row and e = dense - c, both [heads, rows, head_dim] in float64, and combo
    is each row's age bucket and row type.
    """

    def __init__(self, model: Any, dense_backend: Any, policies: dict[str, Any], generation: Any, top_p_logits) -> None:
        attention = model.model.layers[0].self_attn
        self.model = model
        self.dense = DenseWithPreRopeKV(dense_backend, attention.num_key_value_heads, attention.head_dim)
        self.sampler = DiffuCoderDenoisingSampler(self.dense)
        self.backends = {name: DiffuCoderActiveTokenKVBackend(model, policy) for name, policy in policies.items()}
        self.stores = {name: PreRopeKV(backend) for name, backend in self.backends.items()}
        self.generation = generation
        self.top_p_logits = top_p_logits

    def run(self, prompt: torch.Tensor, attention_mask: torch.Tensor, row_type: torch.Tensor, sink: Sink) -> list[int]:
        """Replay one task and return the dense token IDs."""

        for store in self.stores.values():
            store.reset(row_type.numel())
        edges = torch.tensor(AGE_EDGES, device=row_type.device)

        def on_step(step: int, x: torch.Tensor, state: Any) -> None:
            valid = state.classify(x, step)[0].ne(int(DenoisingTokenCategory.PAD))
            for name, backend in self.backends.items():
                store = self.stores[name]
                for layer in range(backend.num_layers):
                    last = backend.cache.layer_last_refresh[layer, 0]
                    if bool((valid & last.lt(0)).any()):
                        raise RuntimeError(f"{name}: a row at layer {layer} was never computed")
                    # Rows computed this step have last == step; the rest are stale.
                    positions = torch.where(valid & last.lt(step))[0]
                    if positions.numel() == 0:
                        continue
                    ages = step - last[positions]
                    combo = torch.bucketize(ages, edges, right=True) * len(ROW_TYPES) + row_type[positions]
                    pairs = ((self.dense.keys[layer], store.keys[layer]), (self.dense.values[layer], store.values[layer]))
                    for kind, (dense_rows, cached_rows) in enumerate(pairs):
                        c = cached_rows[:, positions].double()
                        sink(name, layer, kind, c, dense_rows[:, positions].double() - c, combo)

        result = replay_task(
            self.model, self.sampler, self.backends, prompt, attention_mask, self.generation, self.top_p_logits,
            on_step=on_step,
        )
        return result["dense_token_ids"]


# Running sums, fits and held-out sums -----------------------------------------


def empty_sums(num_layers: int, num_heads: int, head_dim: int, device: Any) -> dict[str, torch.Tensor]:
    cell = (num_layers, len(KINDS), num_heads, COMBOS)
    zeros = lambda *extra: torch.zeros(cell + extra, dtype=torch.float64, device=device)  # noqa: E731
    return {
        "rows": zeros(),
        "c_sum": zeros(head_dim),
        "e_sum": zeros(head_dim),
        "cc": zeros(head_dim, head_dim),  # sum of c c^T
        "ce": zeros(head_dim, head_dim),  # sum of c e^T
        "ee": zeros(head_dim, head_dim),  # sum of e e^T
    }


def add_rows(sums: dict[str, torch.Tensor], layer: int, kind: int, c: torch.Tensor, e: torch.Tensor, combo: torch.Tensor) -> None:
    """Add stale rows to the running sums. c, e: [heads, rows, head_dim]; combo: [rows]."""

    onehot = F.one_hot(combo, COMBOS).to(c.dtype)  # [rows, combos]
    sums["rows"][layer, kind] += onehot.sum(0)
    sums["c_sum"][layer, kind] += torch.einsum("nk,hnd->hkd", onehot, c)
    sums["e_sum"][layer, kind] += torch.einsum("nk,hnd->hkd", onehot, e)
    c_by_cell = c.unsqueeze(2) * onehot[None, :, :, None]  # [heads, rows, combos, head_dim]
    e_by_cell = e.unsqueeze(2) * onehot[None, :, :, None]
    sums["cc"][layer, kind] += torch.einsum("hnkc,hnd->hkcd", c_by_cell, c)
    sums["ce"][layer, kind] += torch.einsum("hnkc,hnd->hkcd", c_by_cell, e)
    sums["ee"][layer, kind] += torch.einsum("hnkc,hnd->hkcd", e_by_cell, e)


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    """numerator / denominator, and 0 where the denominator is not positive."""

    positive = denominator > 0
    return torch.where(positive, numerator / torch.where(positive, denominator, 1.0), 0.0)


def fit_methods(sums: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Closed-form fit of M1-M4 per cell, plus the error RMS and top error subspace."""

    rows = sums["rows"]
    head_dim = sums["c_sum"].shape[-1]
    count = rows.clamp_min(1).unsqueeze(-1)
    mean_c, mean_e = sums["c_sum"] / count, sums["e_sum"] / count
    diagonal_cc = sums["cc"].diagonal(dim1=-2, dim2=-1)
    diagonal_ce = sums["ce"].diagonal(dim1=-2, dim2=-1)
    error_energy = sums["ee"].diagonal(dim1=-2, dim2=-1).sum(-1)

    # M3: per dimension, least squares of e_d on c_d with an intercept.
    slope = safe_divide(diagonal_ce / count - mean_c * mean_e, diagonal_cc / count - mean_c.square())
    # M4: ridge regression of e on c, centered so the intercept is unpenalized.
    cov_cc = sums["cc"] / count.unsqueeze(-1) - mean_c.unsqueeze(-1) * mean_c.unsqueeze(-2)
    cov_ce = sums["ce"] / count.unsqueeze(-1) - mean_c.unsqueeze(-1) * mean_e.unsqueeze(-2)
    ridge = RIDGE * cov_cc.diagonal(dim1=-2, dim2=-1).sum(-1) / head_dim
    # Cells with fewer than two distinct rows have no spread to fit; they keep A = 0.
    solvable = (ridge > 0).unsqueeze(-1).unsqueeze(-1)
    identity = torch.eye(head_dim, dtype=cov_cc.dtype, device=cov_cc.device)
    system = torch.where(solvable, cov_cc + ridge.unsqueeze(-1).unsqueeze(-1) * identity, identity)
    matrix = torch.linalg.solve(system, cov_ce).transpose(-1, -2) * solvable  # rows index e
    _, eigenvectors = torch.linalg.eigh(sums["ee"])  # ascending eigenvalues, vectors in columns
    return {
        "train_rows": rows,
        "train_energy": error_energy,
        "train_rms": safe_divide(error_energy, rows).sqrt(),
        # M1: e = (s - 1) c; this is s - 1.
        "m1_scale": safe_divide(diagonal_ce.sum(-1), diagonal_cc.sum(-1)),
        "m2_shift": mean_e,
        "m3_slope": slope,
        "m3_intercept": mean_e - slope * mean_c,
        "m4_matrix": matrix,
        "m4_intercept": mean_e - (matrix @ mean_c.unsqueeze(-1)).squeeze(-1),
        "subspace": eigenvectors[..., -SUBSPACE_RANK:].flip(-1).transpose(-1, -2),  # [..., rank, head_dim]
    }


PARAMETER_KEYS = ("m1_scale", "m2_shift", "m3_slope", "m3_intercept", "m4_matrix", "m4_intercept", "subspace")


def linear_maps(fit: dict[str, torch.Tensor]) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Each method as (weights, intercept). Weights shaped like the intercept are a per-dimension slope."""

    zeros = torch.zeros_like(fit["m2_shift"])
    return {
        "M1": (fit["m1_scale"].unsqueeze(-1).expand_as(zeros), zeros),
        "M2": (zeros, fit["m2_shift"]),
        "M3": (fit["m3_slope"], fit["m3_intercept"]),
        "M4": (fit["m4_matrix"], fit["m4_intercept"]),
    }


def residual_energy(sums: dict[str, torch.Tensor], weights: torch.Tensor, intercept: torch.Tensor) -> torch.Tensor:
    """Sum over rows of |e - (A c + b)|^2 per cell, from the running sums alone."""

    matrix = torch.diag_embed(weights) if weights.dim() == intercept.dim() else weights
    error_energy = sums["ee"].diagonal(dim1=-2, dim2=-1).sum(-1)
    cross = torch.einsum("...ec,...ce->...", matrix, sums["ce"])
    predicted = torch.einsum("...ec,...cd,...ed->...", matrix, sums["cc"], matrix)
    intercept_terms = (
        -2 * (intercept * sums["e_sum"]).sum(-1)
        + 2 * torch.einsum("...e,...ec,...c->...", intercept, matrix, sums["c_sum"])
        + sums["rows"] * intercept.square().sum(-1)
    )
    return error_energy - 2 * cross + predicted + intercept_terms


def predict(weights: torch.Tensor, intercept: torch.Tensor, c: torch.Tensor, combo: torch.Tensor) -> torch.Tensor:
    """e_hat for rows c [heads, rows, head_dim]; weights and intercept are one (layer, kind)."""

    shift = intercept[:, combo]
    if weights.dim() == intercept.dim():
        return weights[:, combo] * c + shift
    # Apply every cell's matrix to every row, then keep each row's own cell.
    every = torch.einsum("hkoc,hnc->hkno", weights, c)
    return every[:, combo, torch.arange(c.shape[1], device=c.device)] + shift


def empty_held_out(num_layers: int, num_heads: int, device: Any) -> dict[str, torch.Tensor]:
    cell = (num_layers, len(KINDS), num_heads, COMBOS)
    zeros = lambda *lead: torch.zeros(lead + cell, dtype=torch.float64, device=device)  # noqa: E731
    return {"rows": zeros(), "energy": zeros(), "residual": zeros(len(METHODS)), "subspace": zeros()}


def add_held_out(
    sums: dict[str, torch.Tensor],
    fit: dict[str, torch.Tensor],
    layer: int,
    kind: int,
    c: torch.Tensor,
    e: torch.Tensor,
    combo: torch.Tensor,
) -> None:
    """Score stale rows of a held-out task against the fit of its fold."""

    sums["rows"][layer, kind].index_add_(1, combo, torch.ones(c.shape[:2], dtype=c.dtype, device=c.device))
    sums["energy"][layer, kind].index_add_(1, combo, e.square().sum(-1))
    for index, (weights, intercept) in enumerate(linear_maps(fit).values()):
        prediction = predict(weights[layer, kind], intercept[layer, kind], c, combo)
        sums["residual"][index, layer, kind].index_add_(1, combo, (e - prediction).square().sum(-1))
    basis = fit["subspace"][layer, kind][:, combo]  # [heads, rows, rank, head_dim]
    captured = torch.einsum("hnrd,hnd->hnr", basis, e).square().sum(-1)
    sums["subspace"][layer, kind].index_add_(1, combo, captured)


# GPU steps: collect and evaluate ---------------------------------------------


def study_dir(config: Any, gpu_folder: str) -> Path:
    return config.run.run_dir / gpu_folder / "stale_kv_error_study"


class Study:
    """The model, tasks and folds shared by collect and evaluate."""

    def __init__(self, config: Any) -> None:
        if config.generation.temperature != 0:
            raise ValueError("The stale K/V error study is pre-registered for greedy decoding")
        self.config = config
        self.tasks = load_code_task_dataset(config.evaluation.code_tasks_file)[: config.evaluation.max_prompts]
        self.folds = assign_folds([task.task_id for task in self.tasks])
        self.model, self.tokenizer = load_diffucoder_model_and_tokenizer(
            config.model.path,
            device=config.model.device,
            dtype=config.model.dtype,
            local_files_only=config.model.local_files_only,
        )
        self.stop_ids = {token for token in (self.tokenizer.eos_token_id, self.tokenizer.pad_token_id) if token is not None}
        self.output_dir = study_dir(config, gpu_folder_name())
        # The earlier forced-trajectory replay has every task's final dense output.
        self.replay_dir = config.run.run_dir / gpu_folder_name() / "forced_trajectory_replay" / "tasks"
        attention = self.model.model.layers[0].self_attn
        self.shape = (len(self.model.model.layers), attention.num_key_value_heads, attention.head_dim)
        print(f"Results: {self.output_dir}; layers, KV heads, head_dim = {self.shape}", flush=True)

    def replay(self, policy_names: tuple[str, ...]) -> StaleRowReplay:
        policies = {name: self.config.dkv_policies[name] for name in policy_names}
        return StaleRowReplay(
            self.model, DenseDiffuCoderBaseline(self.model), policies, self.config.generation,
            load_top_p_logits(self.model),
        )

    def fold_tasks(self, fold: int) -> list[int]:
        return [index for index, task_fold in enumerate(self.folds) if task_fold == fold]

    def run_task(self, replay: StaleRowReplay, index: int, sink: Sink) -> None:
        task = self.tasks[index]
        seed_everything(self.config.generation.seed + index)
        prompt, attention_mask = prepare_diffucoder_prompt_batch(self.tokenizer, [task.prompt], self.config.model.device)
        saved = json.loads((self.replay_dir / f"task_{index:03d}.json").read_text())
        row_type = row_types(prompt.shape[1], saved["dense_token_ids"], self.stop_ids, prompt.device)
        if replay.run(prompt, attention_mask, row_type, sink) != saved["dense_token_ids"]:
            raise RuntimeError(f"Task {index}: dense output differs from the earlier replay")

    def write_job_info(self, step: str, fold: int) -> None:
        def git(*args: str) -> str:
            return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()

        info = {
            "commit": git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(git("status", "--porcelain", "--untracked-files=no")),
            "gpu": gpu_folder_name(),
            "layers_kv_heads_head_dim": self.shape,
            "policies": {name: self.config.dkv_policies[name].to_dict() for name in POLICIES},
            "generation": dict(self.config.generation.__dict__),
            "folds": {task.task_id: fold for task, fold in zip(self.tasks, self.folds)},
        }
        path = self.output_dir / "jobs" / f"{step}_fold{fold}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def collect(config: Any, fold: int) -> None:
    study = Study(config)
    study.write_job_info("collect", fold)
    replay = study.replay(POLICIES)
    device = config.model.device
    fold_sums = {name: empty_sums(*study.shape, device) for name in POLICIES}
    task_sums = {name: empty_sums(*study.shape, device) for name in POLICIES}
    first_task_sums = None
    indices = study.fold_tasks(fold)
    for count, index in enumerate(indices):
        for sums in task_sums.values():
            for tensor in sums.values():
                tensor.zero_()
        study.run_task(replay, index, lambda name, *rows: add_rows(task_sums[name], *rows))
        for name in POLICIES:
            for key, tensor in task_sums[name].items():
                fold_sums[name][key] += tensor
        if first_task_sums is None:
            first_task_sums = {name: {key: t.cpu() for key, t in sums.items()} for name, sums in task_sums.items()}
        print(f"fold {fold}: task {count + 1}/{len(indices)} {study.tasks[index].task_id} collected", flush=True)
    for name, sums in fold_sums.items():
        path = study.output_dir / "sums" / name / f"fold{fold}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: tensor.cpu() for key, tensor in sums.items()}, path)
    del fold_sums, task_sums

    if fold == 0:
        # Sanity checks 1 and 3: replay the first task again, with the control policy added.
        names = POLICIES + (CONTROL_POLICY,)
        repeat = {name: empty_sums(*study.shape, device) for name in names}
        study.run_task(study.replay(names), indices[0], lambda name, *rows: add_rows(repeat[name], *rows))
        check = {
            "task_id": study.tasks[indices[0]].task_id,
            "control_policy": CONTROL_POLICY,
            "control_stale_rows": int(repeat[CONTROL_POLICY]["rows"][:, 0, 0].sum().item()),
            "repeat_sums_identical": {
                name: all(torch.equal(first_task_sums[name][key], repeat[name][key].cpu()) for key in repeat[name])
                for name in POLICIES
            },
        }
        path = study.output_dir / "checks" / "repeat_and_control.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(check, indent=2), encoding="utf-8")
        print(f"Checks 1 and 3: {check}", flush=True)


def load_fit(path: Path, device: Any) -> dict[str, torch.Tensor]:
    with np.load(path) as saved:
        return {key: torch.from_numpy(saved[key]).to(device=device, dtype=torch.float64) for key in PARAMETER_KEYS}


def evaluate(config: Any, fold: int) -> None:
    study = Study(config)
    study.write_job_info("evaluate", fold)
    replay = study.replay(POLICIES)
    device = config.model.device
    fits = {name: load_fit(study.output_dir / "fit" / name / f"fold{fold}.npz", device) for name in POLICIES}
    num_layers, num_heads, _ = study.shape
    sample_layers = (SCORED_FIRST_LAYER, num_layers - 1)
    indices = study.fold_tasks(fold)
    for count, index in enumerate(indices):
        held_out = {name: empty_held_out(num_layers, num_heads, device) for name in POLICIES}
        sampling = fold == 0 and count < RAW_SAMPLE_TASKS
        raw: dict[str, list[torch.Tensor]] = {}

        def sink(name: str, layer: int, kind: int, c: torch.Tensor, e: torch.Tensor, combo: torch.Tensor) -> None:
            add_held_out(held_out[name], fits[name], layer, kind, c, e, combo)
            if sampling and name == RAW_SAMPLE_POLICY and layer in sample_layers:
                prefix = f"layer{layer}_{KINDS[kind]}"
                raw.setdefault(f"{prefix}_c", []).append(c[RAW_SAMPLE_HEAD].float().cpu())
                raw.setdefault(f"{prefix}_e", []).append(e[RAW_SAMPLE_HEAD].float().cpu())
                raw.setdefault(f"{prefix}_combo", []).append(combo.cpu())

        study.run_task(replay, index, sink)
        for name, sums in held_out.items():
            path = study.output_dir / "eval" / name / f"task_{index:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            arrays = {key: tensor.cpu().numpy() for key, tensor in sums.items()}
            np.savez(path, task_id=study.tasks[index].task_id, fold=fold, **arrays)
        if sampling:
            path = study.output_dir / "checks" / f"raw_sample_task_{index:03d}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, layers=np.array(sample_layers), **{key: torch.cat(parts).numpy() for key, parts in raw.items()})
        print(f"fold {fold}: task {count + 1}/{len(indices)} {study.tasks[index].task_id} evaluated", flush=True)


# CPU steps: fit and report ----------------------------------------------------


def fit(output_dir: Path) -> None:
    for name in POLICIES:
        fold_sums = [torch.load(output_dir / "sums" / name / f"fold{k}.pt", weights_only=True) for k in range(FOLDS)]
        for held_out in range(FOLDS):
            train = {key: sum(fold_sums[k][key] for k in range(FOLDS) if k != held_out) for key in fold_sums[0]}
            fitted = fit_methods(train)
            # Saved in float32; in-sample residuals use the saved values, as evaluate does.
            saved = {key: fitted[key].float() for key in PARAMETER_KEYS}
            in_sample = torch.stack([
                residual_energy(train, weights, intercept)
                for weights, intercept in linear_maps({key: value.double() for key, value in saved.items()}).values()
            ])
            path = output_dir / "fit" / name / f"fold{held_out}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                path,
                in_sample_residual=in_sample.numpy(),
                **{key: fitted[key].numpy() for key in ("train_rows", "train_energy", "train_rms")},
                **{key: value.numpy() for key, value in saved.items()},
            )
            pairs = int(train["rows"][:, 0, 0].sum())
            print(f"{name} fold {held_out}: fit from {pairs} stale (row, layer) pairs", flush=True)


def equal_weight_fer(energy: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Mean over (layer, kind) of 1 - residual / energy.

    energy: [..., layers, kinds]; residual: [..., methods, layers, kinds]. Pairs
    with no error energy are skipped.
    """

    energy = np.expand_dims(energy, -3)
    fer = 1 - residual / np.where(energy > 0, energy, np.nan)
    return np.nanmean(fer, axis=(-2, -1))


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    samples = rng.integers(0, len(values), size=(BOOTSTRAP_RESAMPLES, len(values)))
    means = values[samples].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def raw_sample_check(output_dir: Path) -> dict[str, Any]:
    """Sanity check 2: held-out sums and the sums formula both match a direct recomputation."""

    worst = 0.0
    tasks = []
    for path in sorted((output_dir / "checks").glob("raw_sample_task_*.npz")):
        index = int(path.stem.rsplit("_", 1)[1])
        raw = np.load(path)
        held = np.load(output_dir / "eval" / RAW_SAMPLE_POLICY / f"task_{index:03d}.npz")
        fitted = load_fit(output_dir / "fit" / RAW_SAMPLE_POLICY / f"fold{int(held['fold'])}.npz", "cpu")
        maps = linear_maps(fitted)
        for layer in raw["layers"].tolist():
            for kind, kind_name in enumerate(KINDS):
                c = torch.from_numpy(raw[f"layer{layer}_{kind_name}_c"]).double()
                e = torch.from_numpy(raw[f"layer{layer}_{kind_name}_e"]).double()
                combo = torch.from_numpy(raw[f"layer{layer}_{kind_name}_combo"]).long()
                sums = empty_sums(1, 1, c.shape[-1], "cpu")
                add_rows(sums, 0, 0, c[None], e[None], combo)
                sums = {key: tensor[0, 0, 0] for key, tensor in sums.items()}  # [combos, ...]
                for m, (weights, intercept) in enumerate(maps.values()):
                    weights = weights[layer, kind, RAW_SAMPLE_HEAD]
                    intercept = intercept[layer, kind, RAW_SAMPLE_HEAD]
                    direct = 0.0
                    for cell in combo.unique().tolist():
                        rows = combo == cell
                        w, b = weights[cell], intercept[cell]
                        prediction = c[rows] * w + b if w.dim() == 1 else c[rows] @ w.T + b
                        direct += float((e[rows] - prediction).square().sum())
                    from_sums = float(residual_energy(sums, weights, intercept).sum())
                    from_held_out = float(held["residual"][m, layer, kind, RAW_SAMPLE_HEAD].sum())
                    worst = max(worst, abs(from_sums - direct) / direct, abs(from_held_out - direct) / direct)
        tasks.append(str(held["task_id"]))
    return {"tasks": tasks, "max_relative_error": worst, "passed": bool(tasks) and worst <= RAW_SAMPLE_MAX_RELATIVE_ERROR}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(output_dir: Path, task_count: int) -> None:
    rng = np.random.default_rng(0)
    scored = slice(SCORED_FIRST_LAYER, None)
    method_names = ("M0",) + METHODS
    results, breakdown, second_moment, policy_reports = [], [], [], {}
    for name in POLICIES:
        tasks = [np.load(path) for path in sorted((output_dir / "eval" / name).glob("task_*.npz"))]
        if len(tasks) != task_count:
            raise SystemExit(f"{name}: {len(tasks)} evaluated tasks, expected {task_count}. Finish evaluate first.")
        fits = [np.load(output_dir / "fit" / name / f"fold{k}.npz") for k in range(FOLDS)]
        task_fold = np.array([int(task["fold"]) for task in tasks])
        rows = np.stack([task["rows"] for task in tasks])  # [tasks, layers, kinds, heads, combos]
        energy = np.stack([task["energy"] for task in tasks])
        # M0's residual is the error itself.
        residual = np.stack([np.concatenate([task["energy"][None], task["residual"]]) for task in tasks])
        subspace = np.stack([task["subspace"] for task in tasks])
        head_dim = fits[0]["m2_shift"].shape[-1]
        parameters = {"M0": 0, "M1": 1, "M2": head_dim, "M3": 2 * head_dim, "M4": head_dim * head_dim + head_dim}

        # First moment. Pool heads and cells within each (layer, kind).
        layer_energy = energy[:, scored].sum(axis=(3, 4))  # [tasks, layers, kinds]
        layer_residual = residual[:, :, scored].sum(axis=(4, 5))  # [tasks, methods, layers, kinds]
        per_task = equal_weight_fer(layer_energy, layer_residual)  # [tasks, methods]
        held_out_pooled = np.mean([
            equal_weight_fer(layer_energy[task_fold == k].sum(0), layer_residual[task_fold == k].sum(0))
            for k in range(FOLDS)
        ], axis=0)
        in_sample = np.mean([
            equal_weight_fer(
                fold_fit["train_energy"][scored].sum(axis=(2, 3)),
                np.concatenate([fold_fit["train_energy"][None], fold_fit["in_sample_residual"]])[:, scored].sum(axis=(3, 4)),
            )
            for fold_fit in fits
        ], axis=0)
        energy_weighted = 1 - layer_residual.sum(axis=(0, 2, 3)) / layer_energy.sum()
        policy_rows = {}
        for m, method in enumerate(method_names):
            scores = per_task[:, m]
            low, high = bootstrap_interval(scores, rng)
            row = {
                "policy": name,
                "method": method,
                "parameters_per_cell": parameters[method],
                "fer_mean": float(scores.mean()),
                "ci_low": low,
                "ci_high": high,
                **{f"fold{k + 1}_mean": float(scores[task_fold == k].mean()) for k in range(FOLDS)},
                "energy_weighted_fer": float(energy_weighted[m]),
                "in_sample_fer": float(in_sample[m]),
                "held_out_pooled_fer": float(held_out_pooled[m]),
            }
            results.append(row)
            policy_rows[method] = row

        # Breakdown per cell, pooled over held-out tasks and heads; includes layer 0.
        cell_energy = energy.sum(axis=(0, 3))  # [layers, kinds, combos]
        cell_residual = residual.sum(axis=(0, 4))  # [methods, layers, kinds, combos]
        cell_rows = rows[:, :, :, 0].sum(axis=0)
        for layer, kind, combo in np.ndindex(*cell_energy.shape):
            for m, method in enumerate(method_names):
                total = cell_energy[layer, kind, combo]
                breakdown.append({
                    "policy": name,
                    "method": method,
                    "layer": layer,
                    "kind": KINDS[kind],
                    "age_bucket": AGE_BUCKETS[combo // len(ROW_TYPES)],
                    "row_type": ROW_TYPES[combo % len(ROW_TYPES)],
                    "fer": float(1 - cell_residual[m, layer, kind, combo] / total) if total > 0 else "",
                    "n_rows": int(cell_rows[layer, kind, combo]),
                })

        # Second moment. S1: per held-out (task, cell) with enough rows.
        relative_errors = []
        for task_index, task in enumerate(tasks):
            train_rms = fits[int(task["fold"])]["train_rms"][scored]
            task_rows = rows[task_index, scored]
            enough = (task_rows >= MIN_ROWS_FOR_S1) & (train_rms > 0)
            task_rms = np.sqrt(energy[task_index, scored][enough] / task_rows[enough])
            relative_errors.append(np.abs(task_rms - train_rms[enough]) / train_rms[enough])
        relative_errors = np.concatenate(relative_errors)
        layer_subspace = subspace[:, scored].sum(axis=(0, 3, 4))
        second = {
            "policy": name,
            "s1_median_relative_error": float(np.median(relative_errors)) if relative_errors.size else float("nan"),
            "s1_task_cells": int(relative_errors.size),
            "s2_captured_fraction": float(subspace[:, scored].sum() / energy[:, scored].sum()),
            "s2_captured_fraction_equal_weight": float(np.mean(layer_subspace / layer_energy.sum(axis=0))),
            "random_baseline": SUBSPACE_RANK / head_dim,
        }
        second_moment.append(second)

        # Sanity 4: mean error energy per row and head, by age bucket.
        by_age = energy[:, scored].reshape(*energy[:, scored].shape[:-1], len(AGE_BUCKETS), len(ROW_TYPES))
        age_rows = rows[:, scored].reshape(by_age.shape)
        age_energy = by_age.sum(axis=(0, 1, 2, 3, 5)) / np.maximum(age_rows.sum(axis=(0, 1, 2, 3, 5)), 1)
        present = age_rows.sum(axis=(0, 1, 2, 3, 5)) > 0
        increasing = bool(np.all(np.diff(age_energy[present]) >= 0))

        m4 = policy_rows["M4"]
        if m4["ci_low"] > 0 and m4["fer_mean"] >= MIN_FER and all(m4[f"fold{k + 1}_mean"] > 0 for k in range(FOLDS)):
            first_moment = "systematic"
        elif m4["ci_low"] > 0:
            first_moment = "weak"
        else:
            first_moment = "not systematic"
        policy_reports[name] = {
            "first_moment": first_moment,
            "second_moment_systematic": bool(
                second["s1_median_relative_error"] <= MAX_S1_RELATIVE_ERROR
                and second["s2_captured_fraction"] >= MIN_S2_FRACTION
            ),
            "m4_fer": m4["fer_mean"],
            "m4_ci": [m4["ci_low"], m4["ci_high"]],
            "m4_in_sample_fer": m4["in_sample_fer"],
            "m4_held_out_pooled_fer": m4["held_out_pooled_fer"],
            "error_energy_by_age_bucket": dict(zip(AGE_BUCKETS, age_energy.tolist())),
            "error_energy_increases_with_age": increasing,
            "stale_row_layer_pairs": int(rows[:, :, 0, 0].sum()),
            **{key: value for key, value in second.items() if key != "policy"},
        }

    first = all(policy_reports[name]["first_moment"] == "systematic" for name in DECISION_POLICIES)
    second = all(policy_reports[name]["second_moment_systematic"] for name in DECISION_POLICIES)
    outcome = "A: first-moment systematic" if first else "B: second-moment only" if second else "C: neither"
    repeat_path = output_dir / "checks" / "repeat_and_control.json"
    summary = {
        "outcome": outcome,
        "first_moment_systematic": first,
        "second_moment_systematic": second,
        "decision_policies": DECISION_POLICIES,
        "sanity_checks": {
            "control_and_repeat": json.loads(repeat_path.read_text()) if repeat_path.is_file() else None,
            "raw_sample": raw_sample_check(output_dir),
        },
        "policies": policy_reports,
    }
    write_csv(output_dir / "stale_kv_error_results.csv", results)
    write_csv(output_dir / "stale_kv_error_breakdown.csv", breakdown)
    write_csv(output_dir / "stale_kv_error_second_moment.csv", second_moment)
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'policy':<12} {'method':<6} {'FER':>7} {'95% CI':>17} {'in-sample':>9} {'pooled':>7} {'energy-wt':>9}")
    for row in results:
        print(
            f"{row['policy']:<12} {row['method']:<6} {row['fer_mean']:>7.3f} "
            f"[{row['ci_low']:>7.3f}, {row['ci_high']:>6.3f}] {row['in_sample_fer']:>9.3f} "
            f"{row['held_out_pooled_fer']:>7.3f} {row['energy_weighted_fer']:>9.3f}"
        )
    for name, policy_report in policy_reports.items():
        print(
            f"{name}: first moment {policy_report['first_moment']}; S1 {policy_report['s1_median_relative_error']:.3f}, "
            f"S2 {policy_report['s2_captured_fraction']:.3f} (random {policy_report['random_baseline']:.3f}); "
            f"error energy rises with age: {policy_report['error_energy_increases_with_age']}"
        )
    print(f"Sanity checks: {json.dumps(summary['sanity_checks'])}")
    print(f"Outcome: {outcome}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("collect", "fit", "evaluate", "report"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Required for collect and evaluate")
    parser.add_argument("--gpu-folder", help="For fit and report, e.g. NVIDIA_A100-SXM4-80GB")
    args = parser.parse_args()
    config = load_kv_policy_experiment_config(args.config)
    if args.step in ("collect", "evaluate"):
        if args.fold is None:
            parser.error(f"{args.step} needs --fold")
        (collect if args.step == "collect" else evaluate)(config, args.fold)
        return
    if not args.gpu_folder:
        parser.error(f"{args.step} needs --gpu-folder")
    output_dir = study_dir(config, args.gpu_folder)
    if args.step == "fit":
        fit(output_dir)
    else:
        report(output_dir, config.evaluation.max_prompts)


if __name__ == "__main__":
    main()
