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

Amendment 2 adds a rounding check, run after the registered result:

  rounding --fold K --part P
                      GPU. Measures the rounding-only error of rows each
                      policy recomputes in partial forwards, for half of
                      fold K's tasks. Any A100 may run it (see STUDY_GPU).
  rounding-check      Asks whether the registered classification survives the
                      worst-case contribution of that rounding. No model.

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
import warnings
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
from .dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from .fixed_token_refresh_policy import TokenRefreshDecision
from .forced_trajectory_replay import load_top_p_logits, replay_task
from .kv_policy_experiment_config import load_kv_policy_experiment_config

POLICIES = ("refresh_4", "refresh_32", "refresh_256")
# refresh_4 is reported, but its ages are too short to count toward the decision.
DECISION_POLICIES = ("refresh_32", "refresh_256")
# Sanity check 1: recomputes every row each step, so it has no stale rows.
CONTROL_POLICY = "full_refresh_every_step"
FOLDS = 4
FOLD_SEED = 0
# collect and evaluate ran on this GPU type. The follow-ups (rounding and the
# correction pilot) read its replay and fits from any GPU and write here too.
STUDY_GPU = "NVIDIA_A100-SXM4-80GB"
# The follow-ups split each fold's tasks into this many jobs.
PARTS = 2
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
# Amendment of 2026-10-31: the decision reads only prompt and answer rows;
# padding is reported separately, and so is the score over all rows.
ROW_SETS = {"prompt_and_answer": ("prompt", "answer"), "padding": ("padding",), "all": ROW_TYPES}
DECISION_ROWS = "prompt_and_answer"
# Amendment of 2026-10-31: pipeline checks. If either fails, the outcome is not trusted.
# Path check: at step 0 both paths compute every row fresh, so their difference
# is the paths themselves. Its energy per row, averaged over scored (layer, K/V)
# as a share of the stale error energy per row, bounds how much FER a consistent
# path offset could add.
MAX_PATH_SHARE = 0.02
MAX_LAYER0_PATH_DIFFERENCE = 1e-4  # step-0 layer-0 difference energy / dense energy
# Layer-0 check on held-out stale rows.
MAX_LAYER0_ENERGY_SHARE = 0.01  # layer-0 error energy per row / that of layers 1-27
MAX_LAYER0_FER = 0.05  # held-out FER of M2 and M4 at layer 0
# Sanity check 2: every stale row of one head and two layers, for the first
# two tasks of fold 0, is saved during evaluate.
RAW_SAMPLE_TASKS = 2
RAW_SAMPLE_POLICY = "refresh_32"
RAW_SAMPLE_HEAD = 0
RAW_SAMPLE_MAX_RELATIVE_ERROR = 1e-4
# Amendment 2 (2026-11-01, after the registered result): the rounding check.
# At every probe step, a stale-free cache is built with one full forward, and
# the rows each policy recomputed that step are recomputed again in the same
# partial forward, reading every other row from that cache. The difference
# from dense is rounding alone. Probe steps are 1, 9, 17, ...
ROUNDING_PROBE_EVERY = 8
ROUNDING_PROBE_OFFSET = 1
# Per-step fields that must match the earlier replay exactly when a policy is
# replayed again (rounding on another GPU type; the pilot's uncorrected runs).
COMPARED_FIELDS = ("token_tv", "kl", "top1_flip", "position_flip")


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

    def run(
        self,
        prompt: torch.Tensor,
        attention_mask: torch.Tensor,
        row_type: torch.Tensor,
        sink: Sink,
        first_step_sink: Callable[[str, int, int, torch.Tensor, torch.Tensor], None] | None = None,
    ) -> list[int]:
        """Replay one task and return the dense token IDs.

        ``first_step_sink(policy, layer, kind, dense, cached)`` gets every valid
        row at step 0, when both paths have just computed everything.
        """

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
                    pairs = ((self.dense.keys[layer], store.keys[layer]), (self.dense.values[layer], store.values[layer]))
                    if step == 0 and first_step_sink is not None:
                        fresh = torch.where(valid)[0]
                        for kind, (dense_rows, cached_rows) in enumerate(pairs):
                            first_step_sink(name, layer, kind, dense_rows[:, fresh].double(), cached_rows[:, fresh].double())
                    # Rows computed this step have last == step; the rest are stale.
                    positions = torch.where(valid & last.lt(step))[0]
                    if positions.numel() == 0:
                        continue
                    ages = step - last[positions]
                    combo = torch.bucketize(ages, edges, right=True) * len(ROW_TYPES) + row_type[positions]
                    for kind, (dense_rows, cached_rows) in enumerate(pairs):
                        c = cached_rows[:, positions].double()
                        sink(name, layer, kind, c, dense_rows[:, positions].double() - c, combo)

        result = replay_task(
            self.model, self.sampler, self.backends, prompt, attention_mask, self.generation, self.top_p_logits,
            on_step=on_step,
        )
        return result["dense_token_ids"]


class RecomputeRows:
    """A policy that recomputes exactly the given rows, at every layer."""

    def __init__(self, active: torch.Tensor) -> None:
        self.active = active

    def validate(self) -> None:
        pass

    def decide(self, *, states: torch.Tensor, **_: Any) -> TokenRefreshDecision:
        depth = torch.ones_like(states, dtype=torch.long)
        return TokenRefreshDecision(active=self.active, refresh_depth=depth, reasons={"probe": int(self.active.sum())})


def empty_rounding(num_layers: int, num_heads: int, device: Any) -> dict[str, torch.Tensor]:
    cell = (num_layers, len(KINDS))
    return {
        "rows": torch.zeros(cell + (len(ROW_TYPES),), dtype=torch.float64, device=device),
        "energy": torch.zeros(cell + (num_heads, len(ROW_TYPES)), dtype=torch.float64, device=device),
    }


def rounding_probe(
    replay: StaleRowReplay,
    probe: DiffuCoderActiveTokenKVBackend,
    prompt: torch.Tensor,
    attention_mask: torch.Tensor,
    row_type: torch.Tensor,
    sums: dict[str, dict[str, torch.Tensor]],
) -> tuple[dict[str, Any], bool]:
    """Replay one task; at probe steps, add each policy's rounding-only error to sums.

    Only committed rows are measured, since only they go stale. Returns the
    replay result and whether every stale-free cache matched dense exactly.
    """

    generation = replay.generation
    _, model_attention_mask, position_ids = replay.sampler._attention_inputs(prompt, attention_mask, generation.gen_length)
    mask_token_id = int(replay.model.config.mask_token_id)
    probe_store = PreRopeKV(probe)
    for store in replay.stores.values():
        store.reset(row_type.numel())
    exact = True

    def on_step(step: int, x: torch.Tensor, state: Any) -> None:
        nonlocal exact
        if step % ROUNDING_PROBE_EVERY != ROUNDING_PROBE_OFFSET:
            return
        valid = state.classify(x, step)[0].ne(int(DenoisingTokenCategory.PAD))
        committed = valid & row_type.ne(ROW_TYPES.index("prompt")) & x[0].ne(mask_token_id)
        inputs = dict(
            input_ids=x, attention_mask=model_attention_mask, position_ids=position_ids,
            state=state, step=step, total_steps=generation.steps,
        )
        for name, backend in replay.backends.items():
            recomputed = valid & backend.cache.layer_last_refresh[0, 0].eq(step)
            rows = torch.where(recomputed & committed)[0]
            # A full refresh has no partial forward, so no rounding to measure.
            if rows.numel() == 0 or bool(recomputed.eq(valid).all()):
                continue
            probe.set_policy(DKVDecodeRefreshPolicy(cache_reloading_step=1))
            probe.start_generation(x.shape[1])
            probe_store.reset(x.shape[1])
            probe.forward_mask_logits(**inputs)
            for layer in range(probe.num_layers):
                exact &= torch.equal(probe_store.keys[layer][:, valid], replay.dense.keys[layer][:, valid])
                exact &= torch.equal(probe_store.values[layer][:, valid], replay.dense.values[layer][:, valid])
            probe.set_policy(RecomputeRows(recomputed[None]))
            probe.forward_mask_logits(**inputs)
            types = row_type[rows]
            for layer in range(probe.num_layers):
                pairs = ((probe_store.keys[layer], replay.dense.keys[layer]), (probe_store.values[layer], replay.dense.values[layer]))
                for kind, (probed, dense) in enumerate(pairs):
                    difference = (probed[:, rows] - dense[:, rows]).double()
                    sums[name]["rows"][layer, kind].index_add_(0, types, torch.ones_like(types, dtype=torch.float64))
                    sums[name]["energy"][layer, kind].index_add_(1, types, difference.square().sum(-1))

    result = replay_task(
        replay.model, replay.sampler, replay.backends, prompt, attention_mask, generation, replay.top_p_logits,
        on_step=on_step,
    )
    return result, exact


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


def empty_first_step(num_layers: int, num_heads: int, head_dim: int, device: Any) -> dict[str, torch.Tensor]:
    cell = (num_layers, len(KINDS), num_heads)
    zeros = lambda *extra: torch.zeros(cell + extra, dtype=torch.float64, device=device)  # noqa: E731
    return {"rows": zeros(), "difference_energy": zeros(), "dense_energy": zeros(), "difference_sum": zeros(head_dim)}


def add_first_step(sums: dict[str, torch.Tensor], layer: int, kind: int, dense: torch.Tensor, cached: torch.Tensor) -> None:
    """Add step-0 rows, which both paths computed fresh. dense, cached: [heads, rows, head_dim]."""

    difference = dense - cached
    sums["rows"][layer, kind] += dense.shape[1]
    sums["difference_energy"][layer, kind] += difference.square().sum(dim=(1, 2))
    sums["dense_energy"][layer, kind] += dense.square().sum(dim=(1, 2))
    sums["difference_sum"][layer, kind] += difference.sum(dim=1)


def path_check(first_step: dict[str, np.ndarray], stale_energy: np.ndarray, stale_rows: np.ndarray) -> dict[str, Any]:
    """Can the difference between the dense and cached code paths pass for stale error?

    first_step: summed step-0 sums. stale_energy, stale_rows: [layers, kinds]
    stale error energy (summed over heads) and stale rows.
    """

    rows = first_step["rows"][..., 0]
    path_energy = first_step["difference_energy"].sum(-1) / np.maximum(rows, 1)  # per row, all heads
    stale_per_row = stale_energy / np.maximum(stale_rows, 1)
    share = path_energy[SCORED_FIRST_LAYER:] / np.where(stale_per_row > 0, stale_per_row, np.nan)[SCORED_FIRST_LAYER:]
    offset = (np.square(first_step["difference_sum"]).sum(-1) / np.maximum(first_step["rows"], 1)).sum()
    total = first_step["difference_energy"].sum()
    layer0 = float(first_step["difference_energy"][0].sum() / first_step["dense_energy"][0].sum())
    mean_share = float(np.nanmean(share))
    return {
        "mean_share_of_stale_error": mean_share,
        "max_share_of_stale_error": float(np.nanmax(share)),
        "relative_difference_by_layer": (
            first_step["difference_energy"].sum(axis=(1, 2)) / first_step["dense_energy"].sum(axis=(1, 2))
        ).tolist(),
        "layer0_relative_difference": layer0,
        # Share of the step-0 difference that is one consistent offset per (layer, K/V, head).
        "consistent_offset_fraction": float(offset / total) if total > 0 else 0.0,
        "passed": bool(mean_share <= MAX_PATH_SHARE and layer0 <= MAX_LAYER0_PATH_DIFFERENCE),
    }


def row_set_combos(row_set: str) -> list[int]:
    return [combo for combo in range(COMBOS) if ROW_TYPES[combo % len(ROW_TYPES)] in ROW_SETS[row_set]]


# GPU steps: collect and evaluate ---------------------------------------------


def study_dir(config: Any, gpu_folder: str) -> Path:
    return config.run.run_dir / gpu_folder / "stale_kv_error_study"


def same_steps(rows: list[dict[str, Any]], other: list[dict[str, Any]]) -> bool:
    """Whether two replays of one policy have exactly the same per-step fields."""

    return [{key: row[key] for key in COMPARED_FIELDS} for row in rows] == [
        {key: row[key] for key in COMPARED_FIELDS} for row in other
    ]


class Study:
    """The model, tasks and folds shared by collect and evaluate."""

    def __init__(self, config: Any, gpu_folder: str | None = None) -> None:
        """gpu_folder is where the study's data are; by default, this GPU's."""

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
        gpu_folder = gpu_folder or gpu_folder_name()
        self.output_dir = study_dir(config, gpu_folder)
        # The earlier forced-trajectory replay has every task's final dense output.
        self.replay_dir = config.run.run_dir / gpu_folder / "forced_trajectory_replay" / "tasks"
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

    def part_tasks(self, fold: int, part: int) -> list[int]:
        return self.fold_tasks(fold)[part::PARTS]

    def task_inputs(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Seeds the task and returns its prompt, attention mask, row types and saved replay."""

        seed_everything(self.config.generation.seed + index)
        prompt, attention_mask = prepare_diffucoder_prompt_batch(
            self.tokenizer, [self.tasks[index].prompt], self.config.model.device
        )
        saved = json.loads((self.replay_dir / f"task_{index:03d}.json").read_text())
        row_type = row_types(prompt.shape[1], saved["dense_token_ids"], self.stop_ids, prompt.device)
        return prompt, attention_mask, row_type, saved

    def run_task(self, replay: StaleRowReplay, index: int, sink: Sink, first_step_sink: Callable | None = None) -> None:
        prompt, attention_mask, row_type, saved = self.task_inputs(index)
        if replay.run(prompt, attention_mask, row_type, sink, first_step_sink) != saved["dense_token_ids"]:
            raise RuntimeError(f"Task {index}: dense output differs from the earlier replay")

    def write_job_info(self, step: str, fold: int, part: int | None = None) -> None:
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
        name = f"{step}_fold{fold}" if part is None else f"{step}_fold{fold}_part{part}"
        path = self.output_dir / "jobs" / f"{name}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(info, indent=2), encoding="utf-8")


def collect(config: Any, fold: int) -> None:
    study = Study(config)
    study.write_job_info("collect", fold)
    replay = study.replay(POLICIES)
    device = config.model.device
    fold_sums = {name: empty_sums(*study.shape, device) for name in POLICIES}
    task_sums = {name: empty_sums(*study.shape, device) for name in POLICIES}
    first_step = {name: empty_first_step(*study.shape, device) for name in POLICIES}
    decision_combos = row_set_combos(DECISION_ROWS)
    first_task_sums = None
    indices = study.fold_tasks(fold)
    for count, index in enumerate(indices):
        for sums in task_sums.values():
            for tensor in sums.values():
                tensor.zero_()
        task_first_step = {name: empty_first_step(*study.shape, device) for name in POLICIES}
        study.run_task(
            replay, index,
            lambda name, *rows: add_rows(task_sums[name], *rows),
            lambda name, *rows: add_first_step(task_first_step[name], *rows),
        )
        for name in POLICIES:
            for key, tensor in task_sums[name].items():
                fold_sums[name][key] += tensor
            for key, tensor in task_first_step[name].items():
                first_step[name][key] += tensor
        if first_task_sums is None:
            first_task_sums = {name: {key: t.cpu() for key, t in sums.items()} for name, sums in task_sums.items()}
        # The path check for this task alone, against refresh_32's stale error.
        stale = task_sums["refresh_32"]
        check = path_check(
            {key: tensor.cpu().numpy() for key, tensor in task_first_step["refresh_32"].items()},
            stale["ee"].diagonal(dim1=-2, dim2=-1).sum(-1)[..., decision_combos].sum(dim=(2, 3)).cpu().numpy(),
            stale["rows"][:, :, 0, decision_combos].sum(-1).cpu().numpy(),
        )
        print(
            f"fold {fold}: task {count + 1}/{len(indices)} {study.tasks[index].task_id} collected; "
            f"path check: share of stale error {check['mean_share_of_stale_error']:.2e} "
            f"(max {check['max_share_of_stale_error']:.2e}), layer-0 difference {check['layer0_relative_difference']:.2e}, "
            f"{'passed' if check['passed'] else 'FAILED'}",
            flush=True,
        )
    for name, sums in fold_sums.items():
        path = study.output_dir / "sums" / name / f"fold{fold}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: tensor.cpu() for key, tensor in sums.items()}, path)
        torch.save({key: tensor.cpu() for key, tensor in first_step[name].items()}, path.with_name(f"fold{fold}_first_step.pt"))
    del fold_sums, task_sums

    if fold == 0:
        # Sanity checks 1 and 3: replay the first task again, with the control policy added.
        names = POLICIES + (CONTROL_POLICY,)
        repeat = {name: empty_sums(*study.shape, device) for name in names}
        study.run_task(study.replay(names), indices[0], lambda name, *rows: add_rows(repeat[name], *rows))
        check = {
            # Each policy's rerun is compared with that same policy's first run.
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


def rounding(config: Any, fold: int, part: int) -> None:
    """Amendment 2: rounding-only error of the rows each policy recomputes in partial forwards.

    Rounding depends on the GPU's kernels. On a GPU type other than STUDY_GPU,
    the job stops unless every policy reproduces the study GPU's replay exactly.
    """

    study = Study(config, STUDY_GPU)
    gpu = gpu_folder_name()
    study.write_job_info("rounding", fold, part)
    replay = study.replay(POLICIES)
    probe = DiffuCoderActiveTokenKVBackend(study.model, DKVDecodeRefreshPolicy(cache_reloading_step=1))
    num_layers, num_heads, _ = study.shape
    sums = {name: empty_rounding(num_layers, num_heads, config.model.device) for name in POLICIES}
    exact = True
    indices = study.part_tasks(fold, part)
    for count, index in enumerate(indices):
        prompt, attention_mask, row_type, saved = study.task_inputs(index)
        result, task_exact = rounding_probe(replay, probe, prompt, attention_mask, row_type, sums)
        if result["dense_token_ids"] != saved["dense_token_ids"]:
            raise RuntimeError(f"Task {index}: dense output differs from the earlier replay")
        if gpu != STUDY_GPU and not all(same_steps(result["policies"][name], saved["policies"][name]) for name in POLICIES):
            raise RuntimeError(f"Task {index}: {gpu} does not reproduce the {STUDY_GPU} replay exactly")
        exact &= task_exact
        answer_rows = int(sums["refresh_256"]["rows"][0, 0, ROW_TYPES.index("answer")].item())
        print(
            f"fold {fold} part {part} on {gpu}: task {count + 1}/{len(indices)} {study.tasks[index].task_id} probed; "
            f"stale-free cache equals dense: {task_exact}; refresh_256 answer rows so far {answer_rows}",
            flush=True,
        )
    for name, policy_sums in sums.items():
        path = study.output_dir / "rounding" / name / f"fold{fold}_part{part}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({key: tensor.cpu() for key, tensor in policy_sums.items()}, path)
    path = study.output_dir / "rounding" / f"fold{fold}_part{part}.json"
    path.write_text(json.dumps({"gpu": gpu, "stale_free_cache_equals_dense": exact}), encoding="utf-8")


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


def first_moment_class(m4: dict[str, float]) -> str:
    """The registered first-moment rule, applied to M4's held-out scores."""

    if m4["ci_low"] > 0 and m4["fer_mean"] >= MIN_FER and all(m4[f"fold{k + 1}_mean"] > 0 for k in range(FOLDS)):
        return "systematic"
    return "weak" if m4["ci_low"] > 0 else "not systematic"


def score_row_set(held: dict[str, np.ndarray], fits: list[dict[str, np.ndarray]], combos: list[int], rng) -> dict[str, Any]:
    """First- and second-moment scores of one policy, using only the cells in combos."""

    scored = slice(SCORED_FIRST_LAYER, None)
    rows, energy = held["rows"][..., combos], held["energy"][..., combos]
    residual, subspace = held["residual"][..., combos], held["subspace"][..., combos]
    task_fold = held["fold"]

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
            fold_fit["train_energy"][scored][..., combos].sum(axis=(2, 3)),
            fold_fit["residual"][:, scored][..., combos].sum(axis=(3, 4)),
        )
        for fold_fit in fits
    ], axis=0)
    energy_weighted = 1 - layer_residual.sum(axis=(0, 2, 3)) / layer_energy.sum()
    methods = {}
    for m, method in enumerate(("M0",) + METHODS):
        scores = per_task[:, m]
        low, high = bootstrap_interval(scores, rng)
        methods[method] = {
            "fer_mean": float(scores.mean()),
            "ci_low": low,
            "ci_high": high,
            **{f"fold{k + 1}_mean": float(scores[task_fold == k].mean()) for k in range(FOLDS)},
            "energy_weighted_fer": float(energy_weighted[m]),
            "in_sample_fer": float(in_sample[m]),
            "held_out_pooled_fer": float(held_out_pooled[m]),
        }

    # Second moment. S1: per held-out (task, cell) with enough rows.
    relative_errors = []
    for task_index in range(len(task_fold)):
        train_rms = fits[task_fold[task_index]]["train_rms"][scored][..., combos]
        task_rows = rows[task_index, scored]
        enough = (task_rows >= MIN_ROWS_FOR_S1) & (train_rms > 0)
        task_rms = np.sqrt(energy[task_index, scored][enough] / task_rows[enough])
        relative_errors.append(np.abs(task_rms - train_rms[enough]) / train_rms[enough])
    relative_errors = np.concatenate(relative_errors)
    layer_subspace = subspace[:, scored].sum(axis=(0, 3, 4))
    second = {
        "s1_median_relative_error": float(np.median(relative_errors)) if relative_errors.size else float("nan"),
        "s1_task_cells": int(relative_errors.size),
        "s2_captured_fraction": float(subspace[:, scored].sum() / energy[:, scored].sum()),
        "s2_captured_fraction_equal_weight": float(np.mean(layer_subspace / layer_energy.sum(axis=0))),
        "random_baseline": SUBSPACE_RANK / fits[0]["head_dim"],
    }

    # Sanity 4 and the precondition: stale rows and error energy per row and head, by age bucket.
    ages = np.array([combo // len(ROW_TYPES) for combo in combos])
    age_rows = np.array([rows[:, scored, :, 0][..., ages == bucket].sum() for bucket in range(len(AGE_BUCKETS))])
    age_energy = np.array([energy[:, scored][..., ages == bucket].sum() for bucket in range(len(AGE_BUCKETS))])
    heads = rows.shape[3]
    energy_per_row = age_energy / np.maximum(age_rows * heads, 1)
    present = age_rows > 0

    # Layer-0 check: no real staleness there, so tiny error and no structure.
    layer0 = {}
    for kind, kind_name in enumerate(KINDS):
        layer0_energy = energy[:, 0, kind].sum()
        layer0_per_row = layer0_energy / max(rows[:, 0, kind, 0].sum(), 1)
        deeper_per_row = energy[:, scored, kind].sum() / max(rows[:, scored, kind, 0].sum(), 1)
        layer0[kind_name] = {
            "energy_share": float(layer0_per_row / deeper_per_row) if deeper_per_row > 0 else float("nan"),
            **{
                f"{method}_fer": float(1 - residual[:, m, 0, kind].sum() / layer0_energy) if layer0_energy > 0 else None
                for m, method in ((2, "M2"), (4, "M4"))
            },
        }
    layer0_passed = all(
        entry["energy_share"] <= MAX_LAYER0_ENERGY_SHARE
        and all(entry[key] is None or entry[key] <= MAX_LAYER0_FER for key in ("M2_fer", "M4_fer"))
        for entry in layer0.values()
    )

    return {
        "methods": methods,
        "second_moment": second,
        "first_moment": first_moment_class(methods["M4"]),
        "second_moment_systematic": bool(
            second["s1_median_relative_error"] <= MAX_S1_RELATIVE_ERROR and second["s2_captured_fraction"] >= MIN_S2_FRACTION
        ),
        "stale_rows_by_age_bucket": dict(zip(AGE_BUCKETS, age_rows.tolist())),
        "error_energy_by_age_bucket": dict(zip(AGE_BUCKETS, energy_per_row.tolist())),
        "error_energy_increases_with_age": bool(np.all(np.diff(energy_per_row[present]) >= 0)),
        "layer0": layer0,
        "layer0_passed": bool(layer0_passed),
    }


def by_age_bucket(held: dict[str, np.ndarray], combos: list[int], rng) -> list[dict[str, Any]]:
    """Held-out FER of every method within each age bucket, scored like the primary score."""

    scored = slice(SCORED_FIRST_LAYER, None)
    rows = []
    for bucket, bucket_name in enumerate(AGE_BUCKETS):
        cells = [combo for combo in combos if combo // len(ROW_TYPES) == bucket]
        layer_energy = held["energy"][:, scored][..., cells].sum(axis=(3, 4))
        layer_residual = held["residual"][:, :, scored][..., cells].sum(axis=(4, 5))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # tasks with no rows of this age
            per_task = equal_weight_fer(layer_energy, layer_residual)
        present = np.isfinite(per_task[:, 0])
        stale_rows = int(held["rows"][:, scored, :, 0][..., cells].sum())
        for m, method in enumerate(("M0",) + METHODS):
            scores = per_task[present, m]
            low, high = bootstrap_interval(scores, rng) if scores.size else (float("nan"), float("nan"))
            total = layer_energy.sum()
            rows.append({
                "age_bucket": bucket_name,
                "method": method,
                "tasks": int(scores.size),
                "fer_mean": float(scores.mean()) if scores.size else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "energy_weighted_fer": float(1 - layer_residual[:, m].sum() / total) if total > 0 else float("nan"),
                "stale_row_layer_pairs": stale_rows,
            })
    return rows


def load_held_out(output_dir: Path, name: str, task_count: int) -> dict[str, np.ndarray]:
    tasks = [np.load(path) for path in sorted((output_dir / "eval" / name).glob("task_*.npz"))]
    if len(tasks) != task_count:
        raise SystemExit(f"{name}: {len(tasks)} evaluated tasks, expected {task_count}. Finish evaluate first.")
    return {
        "fold": np.array([int(task["fold"]) for task in tasks]),
        "rows": np.stack([task["rows"] for task in tasks]),  # [tasks, layers, kinds, heads, combos]
        "energy": np.stack([task["energy"] for task in tasks]),
        # M0's residual is the error itself.
        "residual": np.stack([np.concatenate([task["energy"][None], task["residual"]]) for task in tasks]),
        "subspace": np.stack([task["subspace"] for task in tasks]),
    }


def load_fits(output_dir: Path, name: str) -> list[dict[str, Any]]:
    fits = []
    for k in range(FOLDS):
        with np.load(output_dir / "fit" / name / f"fold{k}.npz") as saved:
            fits.append({
                "train_energy": saved["train_energy"],
                "train_rms": saved["train_rms"],
                "residual": np.concatenate([saved["train_energy"][None], saved["in_sample_residual"]]),
                "head_dim": saved["m2_shift"].shape[-1],
            })
    return fits


def report(output_dir: Path, task_count: int) -> None:
    rng = np.random.default_rng(0)
    # Added after the registered result; its own generator leaves the registered numbers unchanged.
    age_rng = np.random.default_rng(0)
    method_names = ("M0",) + METHODS
    results, breakdown, second_moment, age_rows, policy_reports = [], [], [], [], {}
    for name in POLICIES:
        held = load_held_out(output_dir, name, task_count)
        fits = load_fits(output_dir, name)
        head_dim = fits[0]["head_dim"]
        parameters = {"M0": 0, "M1": 1, "M2": head_dim, "M3": 2 * head_dim, "M4": head_dim * head_dim + head_dim}

        policy_reports[name] = {}
        for row_set in ROW_SETS:
            scores = score_row_set(held, fits, row_set_combos(row_set), rng)
            policy_reports[name][row_set] = scores
            for method, row in scores["methods"].items():
                results.append({"policy": name, "rows": row_set, "method": method, "parameters_per_cell": parameters[method], **row})
            second_moment.append({"policy": name, "rows": row_set, **scores["second_moment"]})
            age_rows += [
                {"policy": name, "rows": row_set, **row} for row in by_age_bucket(held, row_set_combos(row_set), age_rng)
            ]

        # Path check: step-0 differences against this policy's held-out stale error on the decision rows.
        first_step_paths = [output_dir / "sums" / name / f"fold{k}_first_step.pt" for k in range(FOLDS)]
        if all(path.is_file() for path in first_step_paths):
            parts = [torch.load(path, weights_only=True) for path in first_step_paths]
            first_step = {key: sum(part[key] for part in parts).numpy() for key in parts[0]}
            combos = row_set_combos(DECISION_ROWS)
            policy_reports[name]["path_check"] = path_check(
                first_step,
                held["energy"][..., combos].sum(axis=(0, 3, 4)),
                held["rows"][:, :, :, 0][..., combos].sum(axis=(0, 3)),
            )
        else:
            policy_reports[name]["path_check"] = None

        # Breakdown per cell, pooled over held-out tasks and heads; includes layer 0.
        cell_energy = held["energy"].sum(axis=(0, 3))  # [layers, kinds, combos]
        cell_residual = held["residual"].sum(axis=(0, 4))  # [methods, layers, kinds, combos]
        cell_rows = held["rows"][:, :, :, 0].sum(axis=0)
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

    decision = {name: policy_reports[name][DECISION_ROWS] for name in DECISION_POLICIES}
    first = all(scores["first_moment"] == "systematic" for scores in decision.values())
    second = all(scores["second_moment_systematic"] for scores in decision.values())
    outcome = "A: first-moment systematic" if first else "B: second-moment only" if second else "C: neither"
    repeat_path = output_dir / "checks" / "repeat_and_control.json"
    control_and_repeat = json.loads(repeat_path.read_text()) if repeat_path.is_file() else None
    raw_sample = raw_sample_check(output_dir)
    pipeline = {
        "path_check": all(policy_reports[name]["path_check"] and policy_reports[name]["path_check"]["passed"] for name in DECISION_POLICIES),
        "layer0": all(scores["layer0_passed"] for scores in decision.values()),
        "control_has_no_stale_rows": bool(control_and_repeat and control_and_repeat["control_stale_rows"] == 0),
        "repeat_sums_identical": bool(control_and_repeat and all(control_and_repeat["repeat_sums_identical"].values())),
        "raw_sample": raw_sample["passed"],
    }
    trusted = all(pipeline.values())
    summary = {
        "outcome": outcome if trusted else f"UNTRUSTED (a pipeline check failed): {outcome}",
        "pipeline_checks": pipeline,
        "first_moment_systematic": first,
        "second_moment_systematic": second,
        "decision_policies": DECISION_POLICIES,
        "decision_rows": ROW_SETS[DECISION_ROWS],
        "sanity_checks": {"control_and_repeat": control_and_repeat, "raw_sample": raw_sample},
        "policies": policy_reports,
    }
    write_csv(output_dir / "stale_kv_error_results.csv", results)
    write_csv(output_dir / "stale_kv_error_breakdown.csv", breakdown)
    write_csv(output_dir / "stale_kv_error_second_moment.csv", second_moment)
    write_csv(output_dir / "stale_kv_error_by_age.csv", age_rows)
    (output_dir / "report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'policy':<12} {'rows':<18} {'method':<6} {'FER':>7} {'95% CI':>17} {'in-sample':>9} {'pooled':>7} {'energy-wt':>9}")
    for row in results:
        print(
            f"{row['policy']:<12} {row['rows']:<18} {row['method']:<6} {row['fer_mean']:>7.3f} "
            f"[{row['ci_low']:>7.3f}, {row['ci_high']:>6.3f}] {row['in_sample_fer']:>9.3f} "
            f"{row['held_out_pooled_fer']:>7.3f} {row['energy_weighted_fer']:>9.3f}"
        )
    for name in POLICIES:
        scores = policy_reports[name][DECISION_ROWS]
        print(
            f"{name} ({DECISION_ROWS}): first moment {scores['first_moment']}; "
            f"S1 {scores['second_moment']['s1_median_relative_error']:.3f}, "
            f"S2 {scores['second_moment']['s2_captured_fraction']:.3f} "
            f"(random {scores['second_moment']['random_baseline']:.3f}); "
            f"error energy rises with age: {scores['error_energy_increases_with_age']}; layer 0 passed: {scores['layer0_passed']}"
        )
    print(f"Pipeline checks: {pipeline}")
    print(f"Outcome: {summary['outcome']}")
    print(f"\nM4 by age bucket ({DECISION_ROWS}): FER [95% CI], energy-weighted")
    for row in age_rows:
        if row["rows"] == DECISION_ROWS and row["method"] == "M4" and row["tasks"]:
            print(
                f"{row['policy']:<12} age {row['age_bucket']:<5} {row['fer_mean']:>7.3f} "
                f"[{row['ci_low']:>7.3f}, {row['ci_high']:>6.3f}] {row['energy_weighted_fer']:>7.3f} ({row['tasks']} tasks)"
            )


def worst_case_fer_gain(rho: np.ndarray) -> np.ndarray:
    """Most FER a fit can gain by learning a component holding a share rho of the error energy.

    If e = s + r with |r|^2 = rho |e|^2, then |s| >= (1 - sqrt(rho)) |e|, so
    removing r exactly lowers the residual energy by at most 2 sqrt(rho) - rho
    of |e|^2. That is the case where r points along s; uncorrelated, it is rho.
    """

    rho = np.clip(rho, 0.0, 1.0)
    return 2 * np.sqrt(rho) - rho


def rounding_check(output_dir: Path, task_count: int) -> dict[str, Any]:
    """Amendment 2: does the registered classification survive the worst-case rounding contribution?

    Rounding per row comes from the probes, for answer rows. Prompt rows are
    only computed in full forwards, which match dense exactly. Every stale
    answer row is assumed to carry one partial forward's rounding, which
    overstates it for rows last computed at a full refresh.
    """

    registered = json.loads((output_dir / "report.json").read_text())
    jobs = [f"fold{k}_part{p}" for k in range(FOLDS) for p in range(PARTS)]
    exact = all(
        json.loads((output_dir / "rounding" / f"{job}.json").read_text())["stale_free_cache_equals_dense"]
        for job in jobs
    )
    combos = row_set_combos(DECISION_ROWS)
    answer = ROW_TYPES.index("answer")
    answer_combos = [combo for combo in combos if combo % len(ROW_TYPES) == answer]
    scored = slice(SCORED_FIRST_LAYER, None)
    policies = {}
    for name in POLICIES:
        parts = [torch.load(output_dir / "rounding" / name / f"{job}.pt", weights_only=True) for job in jobs]
        probe_rows = sum(part["rows"] for part in parts).numpy()[..., answer]  # [layers, kinds]
        probe_energy = sum(part["energy"] for part in parts).numpy()[..., answer]  # [layers, kinds, heads]
        rounding_per_row = probe_energy / np.maximum(probe_rows, 1)[..., None]

        held = load_held_out(output_dir, name, task_count)
        stale_answer_rows = held["rows"][..., answer_combos].sum(axis=(0, 4))  # [layers, kinds, heads]
        rounding_energy = stale_answer_rows * rounding_per_row
        error_energy = held["energy"][..., combos].sum(axis=(0, 4))
        rho = rounding_energy.sum(-1)[scored] / error_energy.sum(-1)[scored]  # [layers, kinds]
        fer_bound = float(worst_case_fer_gain(rho).mean())

        scores = registered["policies"][name][DECISION_ROWS]
        m4 = scores["methods"]["M4"]
        adjusted = {key: m4[key] - fer_bound for key in ("fer_mean", "ci_low", *(f"fold{k + 1}_mean" for k in range(FOLDS)))}

        # S2: |P s| >= |P e| - |r| and |s| <= |e| + |r|, pooled over rows.
        captured = held["subspace"][:, scored][..., combos].sum()
        total, rounded = error_energy[scored].sum(), rounding_energy[scored].sum()
        s2_worst = max(np.sqrt(captured) - np.sqrt(rounded), 0.0) ** 2 / (np.sqrt(total) + np.sqrt(rounded)) ** 2

        # S1: a cell's RMS moves by at most the rounding RMS q, in the task and in training alike.
        fits = load_fits(output_dir, name)
        q = np.zeros(rounding_per_row.shape + (COMBOS,))
        q[..., answer_combos] = np.sqrt(rounding_per_row)[..., None]
        q = q[scored][..., combos]
        relative, worst = [], []
        for task in range(len(held["fold"])):
            train_rms = fits[held["fold"][task]]["train_rms"][scored][..., combos]
            task_rows = held["rows"][task, scored][..., combos]
            enough = (task_rows >= MIN_ROWS_FOR_S1) & (train_rms > 0)
            a = np.sqrt(held["energy"][task, scored][..., combos][enough] / task_rows[enough])
            b, cell_q = train_rms[enough], q[enough]
            relative.append(np.abs(a - b) / b)
            worst.append(np.where(b > cell_q, (np.abs(a - b) + 2 * cell_q) / np.maximum(b - cell_q, 1e-300), np.inf))
        s1 = float(np.median(np.concatenate(relative)))
        s1_worst = float(np.median(np.concatenate(worst)))

        second_worst = s1_worst <= MAX_S1_RELATIVE_ERROR and s2_worst >= MIN_S2_FRACTION
        policies[name] = {
            "probed_answer_rows": int(probe_rows[0, 0]),
            "rounding_share_by_layer": {KINDS[k]: rho[:, k].tolist() for k in range(len(KINDS))},
            "rounding_share_mean": float(rho.mean()),
            "rounding_share_max": float(rho.max()),
            "uncorrelated_fer_bound": float(np.clip(rho, 0, 1).mean()),
            "worst_case_fer_bound": fer_bound,
            "m4_registered": {key: m4[key] for key in adjusted},
            "m4_minus_worst_case": adjusted,
            "first_moment_registered": scores["first_moment"],
            "first_moment_worst_case": first_moment_class(adjusted),
            "s1_registered": scores["second_moment"]["s1_median_relative_error"],
            "s1_recomputed": s1,
            "s1_worst_case": s1_worst,
            "s2_registered": scores["second_moment"]["s2_captured_fraction"],
            "s2_worst_case": float(s2_worst),
            "second_moment_registered": scores["second_moment_systematic"],
            "second_moment_worst_case": bool(second_worst),
        }

    def outcome(first: bool, second: bool) -> str:
        return "A: first-moment systematic" if first else "B: second-moment only" if second else "C: neither"

    worst_outcome = outcome(
        all(policies[name]["first_moment_worst_case"] == "systematic" for name in DECISION_POLICIES),
        all(policies[name]["second_moment_worst_case"] for name in DECISION_POLICIES),
    )
    registered_outcome = outcome(registered["first_moment_systematic"], registered["second_moment_systematic"])
    unchanged = all(
        entry["first_moment_worst_case"] == entry["first_moment_registered"]
        and entry["second_moment_worst_case"] == entry["second_moment_registered"]
        for entry in policies.values()
    )
    probed = all(entry["probed_answer_rows"] > 0 for entry in policies.values())
    result = {
        "stale_free_cache_equals_dense": exact,
        "registered_outcome": registered["outcome"],
        "worst_case_outcome": worst_outcome,
        "every_policy_keeps_its_classification": unchanged,
        "passed": bool(exact and probed and unchanged and worst_outcome == registered_outcome),
        "policies": policies,
    }
    (output_dir / "rounding_check.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name, entry in policies.items():
        print(
            f"{name}: rounding share mean {entry['rounding_share_mean']:.2e} (max {entry['rounding_share_max']:.2e}); "
            f"M4 FER {entry['m4_registered']['fer_mean']:.3f} -> worst case {entry['m4_minus_worst_case']['fer_mean']:.3f} "
            f"({entry['first_moment_registered']} -> {entry['first_moment_worst_case']}); "
            f"S1 {entry['s1_registered']:.3f} -> {entry['s1_worst_case']:.3f}, S2 {entry['s2_registered']:.3f} -> {entry['s2_worst_case']:.3f}"
        )
    print(f"Worst-case outcome: {worst_outcome}; rounding check passed: {result['passed']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=("collect", "fit", "evaluate", "report", "rounding", "rounding-check"))
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int, choices=range(FOLDS), help="Required for collect, evaluate and rounding")
    parser.add_argument("--part", type=int, choices=range(PARTS), help="Required for rounding")
    parser.add_argument("--gpu-folder", help="For fit, report and rounding-check, e.g. NVIDIA_A100-SXM4-80GB")
    args = parser.parse_args()
    config = load_kv_policy_experiment_config(args.config)
    gpu_steps = {"collect": collect, "evaluate": evaluate}
    if args.step in gpu_steps or args.step == "rounding":
        if args.fold is None:
            parser.error(f"{args.step} needs --fold")
        if args.step == "rounding":
            if args.part is None:
                parser.error("rounding needs --part")
            rounding(config, args.fold, args.part)
        else:
            gpu_steps[args.step](config, args.fold)
        return
    if not args.gpu_folder:
        parser.error(f"{args.step} needs --gpu-folder")
    output_dir = study_dir(config, args.gpu_folder)
    if args.step == "fit":
        fit(output_dir)
    elif args.step == "report":
        report(output_dir, config.evaluation.max_prompts)
    else:
        rounding_check(output_dir, config.evaluation.max_prompts)


if __name__ == "__main__":
    main()
