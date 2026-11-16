"""Teacher forcing of several tasks at once, one batched cache per policy.

``teacher_force_task`` (teacher_forcing.py) scores every policy along one task's dense
trajectory at batch one. Here each policy's ``BatchedDiffuCoderKVBackend`` holds the
caches of B tasks, so one forward per policy and step covers B tasks and the weights
are read once for all of them.

The dense trajectory is forced, not regenerated. Dense at batch B rounds differently
from dense at batch one, and on a near-tie it would commit a different position and
leave the batch-one dense reference. So the batch-one dense run is recorded once
(``record_trajectory``: the step at which each generated position was committed),
and every step here commits exactly those positions with dense's tokens. The token
sequences each policy sees are then the ones it sees at batch one; only the logits
it is scored on, and dense's logits it is compared with, can round differently.

With one unpadded task the batched backend runs the batch-one arithmetic, so the
results equal ``teacher_force_task``'s (tests/test_batched_teacher_forcing.py). With
padding they agree to rounding; experiments/batched_teacher_forcing_check.py measures
how much that moves EAR and the search's decisions on the GPU.

FLOPs are counted per task as the batch-one backends count them, from each task's
own rows and unpadded length, so the search's cost objective does not depend on B.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from utils.compute_cost import ModelShape, dense_forward_flops, layer_flops, lm_head_flops

from .denoising_token_state import DenoisingTokenState
from .diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from .entropy_confidence import entropy_confidence
from .teacher_forcing import step_metrics

# commit_steps value of a position dense never unmasked (it emitted the mask token).
NEVER = -1


def commit_steps(history: list[torch.Tensor], prompt_width: int, mask_token_id: int) -> list[int]:
    """The step at which dense unmasked each generated position, from ``generate``'s history.

    ``history[step]`` is the batch-one sequence after that step. Tokens never change
    once unmasked, so the first step a position is not the mask token is its commit.
    """

    steps = [NEVER] * (history[0].shape[1] - prompt_width)
    for step, sequence in enumerate(history):
        unmasked = sequence[0, prompt_width:].ne(mask_token_id).tolist()
        for position, done in enumerate(unmasked):
            if done and steps[position] == NEVER:
                steps[position] = step
    return steps


def record_trajectory(sampler: DiffuCoderDenoisingSampler, prompt: torch.Tensor, attention_mask: torch.Tensor,
                      generation: Any) -> dict[str, list[int]]:
    """Generate one task densely at batch one and keep its tokens and commit steps."""

    result = sampler.generate(
        prompt, attention_mask=attention_mask, steps=generation.steps, gen_length=generation.gen_length,
        temperature=generation.temperature, top_p=generation.top_p, top_k=generation.top_k,
        algorithm=generation.algorithm, algorithm_temperature=generation.algorithm_temperature,
        collect_history=True,
    )
    mask_token_id = int(sampler.model.config.mask_token_id)
    return {"dense_token_ids": result.sequences[0, prompt.shape[1]:].tolist(),
            "commit_steps": commit_steps(result.history, prompt.shape[1], mask_token_id)}


def _start(backend: Any, sequence_length: int, batch_size: int) -> None:
    if getattr(backend, "supports_batch", False):
        backend.start_generation(sequence_length, batch_size=batch_size)
    elif batch_size == 1:
        backend.start_generation(sequence_length)
    else:
        raise ValueError(f"{type(backend).__name__} runs one task at a time")


def teacher_force_batch(
    model: torch.nn.Module,
    dense_backend: Any,
    cached_backends: dict[str, Any],
    prompts: torch.Tensor,
    attention_mask: torch.Tensor,
    trajectories: list[dict[str, list[int]]],
    generation: Any,
    top_p_logits,
    answer_ends: list[int | None] | None = None,
) -> list[dict[str, Any]]:
    """Teacher-force B left-padded tasks along their recorded dense trajectories.

    ``dense_backend`` gives dense's logits (BatchedDenseDiffuCoderBaseline; at B=1 also
    DenseDiffuCoderBaseline). ``cached_backends`` are batched backends, one per policy.
    Returns one result per task in ``teacher_force_task``'s format, plus ``rows``: per
    policy and step, the rows the policy recomputed, as the search records them.
    """

    if generation.algorithm != "entropy" or generation.algorithm_temperature != 0:
        raise ValueError("Teacher forcing supports DiffuCoder's entropy ordering with algorithm_temperature 0")
    batch, width = prompts.shape
    if len(trajectories) != batch or (answer_ends is not None and len(answer_ends) != batch):
        raise ValueError("One trajectory and answer end per task")
    gen_length, steps = generation.gen_length, generation.steps
    if any(len(t["commit_steps"]) != gen_length or len(t["dense_token_ids"]) != gen_length for t in trajectories):
        raise ValueError("Trajectories must cover gen_length positions")
    mask_token_id = int(model.config.mask_token_id)
    device = prompts.device
    shape = ModelShape.from_model(model)
    _, model_attention_mask, position_ids = DiffuCoderDenoisingSampler._attention_inputs(
        prompts, attention_mask, gen_length)
    x = F.pad(prompts, (0, gen_length), value=mask_token_id)
    state = DenoisingTokenState.create(width, x.shape[1], attention_mask, mask_token_id, batch_size=batch, device=device)
    for backend in (dense_backend, *cached_backends.values()):
        _start(backend, x.shape[1], batch)
    lengths = attention_mask.sum(-1).tolist()  # each task's unpadded prompt
    names = list(cached_backends)
    per_task = [{name: [] for name in names} for _ in range(batch)]
    # Per policy and step: [B, 3] of active rows, full-depth rows, layer-rows (rows times
    # layers computed). Read once at the end, so the loop does not wait on them.
    counts: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    layers = shape.num_layers

    for step in range(steps):
        # Known before the forward: which generated positions are masked, and which of
        # them dense commits now. Row order is each task's masked positions, ascending.
        masked = [[p for p, s in enumerate(t["commit_steps"]) if s >= step or s == NEVER] for t in trajectories]
        committed = [[row for row, p in enumerate(rows) if t["commit_steps"][p] == step]
                     for rows, t in zip(masked, trajectories)]
        sizes = [len(rows) for rows in masked]
        forward_inputs = dict(input_ids=x, attention_mask=model_attention_mask, position_ids=position_ids,
                              state=state, step=step, total_steps=steps)
        dense_logits = dense_backend.forward_mask_logits(**forward_inputs).mask_logits.split(sizes)
        scored = [b for b in range(batch) if committed[b]]
        rows_on_device = {b: torch.tensor(committed[b], device=device) for b in scored}
        positions = {b: torch.tensor(masked[b], device=device) for b in scored}
        answer_rows = {b: None if answer_ends is None or answer_ends[b] is None else positions[b] < answer_ends[b]
                       for b in scored}
        dense_confidence = {b: entropy_confidence(dense_logits[b], generation.temperature, generation.top_p,
                                                  top_p_logits) for b in scored}
        # One policy at a time, so only one policy's logits for B tasks are held.
        for name in names:
            result = cached_backends[name].forward_mask_logits(**forward_inputs)
            decision = result.decision
            depth_layers = layers - (decision.refresh_depth - 1).clamp(0, layers)
            counts[name].append(torch.stack([
                decision.active.sum(-1), (decision.active & decision.refresh_depth.le(1)).sum(-1),
                (depth_layers * decision.active).sum(-1)], dim=-1))
            logits = result.mask_logits.split(sizes)
            candidates = None if result.candidates is None else result.candidates.split(sizes)
            for b in scored:
                confidence = entropy_confidence(logits[b], generation.temperature, generation.top_p, top_p_logits)
                if candidates is not None:
                    confidence = confidence.masked_fill(~candidates[b], -torch.inf)
                metrics = step_metrics(
                    dense_logits[b], logits[b], rows_on_device[b], confidence,
                    dense_confidence=dense_confidence[b], temperature=generation.temperature,
                    top_p=generation.top_p, top_p_logits=top_p_logits, answer_rows=answer_rows[b],
                )
                metrics["committed_positions"] = [masked[b][row] for row in committed[b]]
                if candidates is not None:
                    allowed = candidates[b].tolist()
                    metrics["outside_positions"] = [masked[b][row] for row in committed[b] if not allowed[row]]
                per_task[b][name].append({"step": step, **metrics})
        # Dense's commits, from the recorded trajectory.
        previous = x.clone()
        task_index = [b for b in range(batch) for _ in committed[b]]
        if task_index:
            columns = [width + masked[b][row] for b in range(batch) for row in committed[b]]
            tokens = [trajectories[b]["dense_token_ids"][masked[b][row]] for b in range(batch) for row in committed[b]]
            x[torch.tensor(task_index, device=device), torch.tensor(columns, device=device)] = torch.tensor(
                tokens, device=device, dtype=x.dtype)
        state.update_after_sampling(previous, x, step)

    results = []
    totals = {name: torch.stack(counts[name]).cpu() for name in names}  # [steps, B, 3]
    for b in range(batch):
        length = lengths[b] + gen_length
        dense_ids = x[b, width:].tolist()
        if dense_ids != trajectories[b]["dense_token_ids"]:
            raise RuntimeError(f"Task {b}: forced sequence differs from its trajectory")
        flops, rows = {}, {}
        for name in names:
            per_step = totals[name][:, b].tolist()
            masked_rows = [sum(s >= step or s == NEVER for s in trajectories[b]["commit_steps"]) for step in range(steps)]
            # layer_flops and lm_head_flops are linear in rows, so totals can be multiplied out.
            flops[name] = (sum(row[2] for row in per_step) * layer_flops(shape, 1, length)
                           + sum(masked_rows) * lm_head_flops(shape, 1))
            rows[name] = [{"active": active, "masked": count, "full_depth": full}
                          for (active, full, _), count in zip(per_step, masked_rows)]
        results.append({"dense_token_ids": dense_ids, "dense_flops": steps * dense_forward_flops(shape, length),
                        "flops": flops, "rows": rows, "policies": per_task[b]})
    return results
