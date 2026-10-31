# Stale K/V error study

Registered 2026-10-31, before any data was collected. Code:
`stale_kv_error_study.py`. Every job records its commit in
`stale_kv_error_study/jobs/`.

## Question

Training-free corrections for cached image diffusion models assume that the
error from reusing a stale feature is systematic enough to predict and
remove. Does that hold for DiffuCoder's K/V cache?

When a cached policy reuses a K or V row instead of recomputing it, the row
differs from what dense computes at that step. We ask whether that error is
predictable from information a cache already has: the row's layer, its age,
its row type and the cached vector itself. Two senses of predictable:

- **First moment:** can we predict the error itself, its direction and size?
  That is what fixing individual decisions would need.
- **Second moment:** can we predict how large the error is and which
  directions it spreads in, without predicting the error itself? That is
  what rescaling or temperature-style corrections need.

This study only measures predictability. Nothing is corrected during
decoding.

## Definitions

- **Stale row:** a cached K or V row of a prompt or committed token that the
  policy did not recompute at this layer in this step. Masked rows are
  recomputed every step, so they are never stale.
- **Age:** the step minus the step at which that row was last computed at
  that layer.
- **Error:** e = x_dense − x_cached for x in {K, V}, at the same step of the
  same forced trajectory. **Keys are compared before RoPE**: both sides are
  the `k_proj` output (bias included), recorded directly rather than undoing
  the rotation. RoPE rotates each position differently, so comparing rotated
  keys would hide an error shared across positions. Values need no
  adjustment.
- **Row type:** prompt; answer (a response position before the first stop
  token of the final dense output); padding (the first stop token and every
  position after it). Padding is split from the answer because the model is
  very confident about padding. Its stale rows are many and probably very
  regular, and they could make the error look predictable on their own.
- **Cell:** (layer, K or V, KV head, age bucket, row type). Age buckets: 1,
  2–3, 4–7, 8–31, 32+. Every correction is fit separately per cell.

## Setup

- DiffuCoder-7B-cpGRPO, greedy (temperature 0, top_p 1), 256 steps × 256
  tokens, the 64-task HumanEval+/MBPP+ subset of
  `configs/diffucoder7b_output_variance_greedy_64tasks.yaml`.
- Forced-trajectory replay against the native dense forward, on an A100 80GB
  (the GPU type of the existing replay, whose dense outputs define the row
  types; each task's dense output must match it exactly).
- Policies: `refresh_4`, `refresh_32` and `refresh_256` as defined in that
  config (full refresh every 4, 32 or 256 steps; newly decoded rows
  recomputed for one more step).
- The model is expected to have 28 layers, 4 KV heads and head_dim 128; the
  jobs record the actual values.
- **Precondition, checked in the code before registration:** each policy
  keeps its own cache across steps (`replay_task` resets it once per task),
  so rows go stale on the policy's own schedule. The cache is not rebuilt
  from dense.

## Folds

Four folds by task, with seed 0: the HumanEval+ tasks and the MBPP+ tasks are
shuffled separately and dealt out in turn, giving 8 + 8 per fold. Each task is
held out exactly once. Folds never split a task by step or token.

## Methods

Each method predicts ê from the cell and the cached vector c, fit on the
training folds only. All have closed-form fits from running sums.

| ID | Method | Prediction | Parameters per cell |
|---|---|---|---|
| M0 | None | ê = 0 | 0 |
| M1 | One scale | ê = (s − 1)·c | 1 |
| M2 | Mean shift | ê = μ | 128 |
| M3 | Per-dimension affine | ê = a ⊙ c + b | 256 |
| M4 | Full affine | ê = A·c + b | 128² + 128 |

- M1 is least squares through the origin.
- M3 is least squares with an intercept per dimension.
- M4 is ridge regression on centered c and e, so b is not penalized. λ = 0.01
  · tr(Σ_cc) / 128, where Σ_cc is the covariance of c in the cell.
- A cell with no spread in c, including one with fewer than two training
  rows, keeps zero slopes and predicts its training mean, or zero if it has
  no training rows.

**M4 is the primary method.** M1–M3 show how much each added parameter buys.

## Procedure

1. **Collect** (GPU, one job per fold). Replay the fold's tasks for all three
   policies. Add every stale row into per-cell sums in float64: n, Σc, Σe,
   Σccᵀ, Σceᵀ, Σeeᵀ.
2. **Fit.** For each fold, add up the other three folds' sums and fit M1–M4.
   Also record the training error RMS per cell (for S1) and the top 8
   eigenvectors of Σeeᵀ (for S2). Parameters are saved in float32; in-sample
   residuals use the saved values.
3. **Evaluate** (GPU, one job per fold). Replay the fold's tasks again with
   that fold's fit. Per task and cell, accumulate n, Σ‖e‖², Σ‖e − ê‖² for
   each method, and Σ‖Pe‖², where P projects onto the training top-8
   subspace. Write one file per task.
4. **Report.** Compute the metrics and the decision below.

## Metrics

**Layer 0 is left out of every score.** A stale row's layer-0 K/V depends only
on its own token, which is fixed once committed. Its error is therefore only
rounding between the two forward paths. It appears in the breakdown as a
rounding floor.

**First moment: fraction of error energy removed (FER), on held-out tasks.**
For each task and each (layer, K or V), pooling heads and cells:

FER = 1 − Σ‖e − ê‖² / Σ‖e‖²

- **Primary per-task score:** the average of FER over (layer, K or V) pairs,
  layers 1–27, each pair weighted equally. Equal weights stop a few
  high-norm layers or sink tokens from dominating.
- FER can be negative, meaning the correction made things worse; the sign is
  kept.
- Reported:
  - the mean over the 64 held-out tasks, with a 95% bootstrap interval
    (10,000 resamples of tasks, seed 0);
  - each fold's mean;
  - the energy-weighted FER pooled over all held-out rows;
  - the in-sample FER next to the held-out FER pooled the same way, so a
    large gap shows overfitting.

**Second moment** (layers 1–27):

- **S1, magnitude predictability:** for every held-out (task, cell) with at
  least 100 stale rows, the relative error between that task's error RMS and
  the training-fold RMS for the cell. The median is reported.
- **S2, subspace consistency:** the fraction of held-out error energy inside
  the training-fold top-8 subspace of its cell, pooled over rows. A random
  8-dimensional subspace captures 8/128 = 6.25%. The average over (layer, K
  or V) is also reported, but not used in the decision.

## Decision rule

These thresholds are fixed now and are also constants in the code.

- **First-moment systematic** requires all of the following, on **both
  refresh_32 and refresh_256**:
  - M4's held-out FER has a 95% interval whose lower bound is above 0;
  - its point estimate is at least 0.20;
  - every fold mean is above 0.
- **Weak:** the interval is above 0, but the point estimate is below 0.20.
  **Not systematic:** anything else.
- **Second-moment systematic** requires, on both refresh_32 and refresh_256,
  a median S1 relative error of at most 0.25 and S2 of at least 0.25 (4× the
  random baseline).
- refresh_4 is reported, but does not count. Its ages are too short to be
  informative.

Outcomes:

- **A: first-moment systematic.** Corrections for individual decisions are
  plausible. Next, apply the corrections during decoding and measure
  divergence events.
- **B: second-moment only.** Only rescaling or temperature-style corrections
  are plausible; these features cannot fix individual decisions.
- **C: neither.** A fixed correction is not viable; record a negative result.

## Sanity checks (before trusting any numbers)

1. `full_refresh_every_step` produces zero stale rows. It is replayed on the
   first task of fold 0.
2. On the first two tasks of fold 0, every stale row of refresh_32 for head 0,
   layers 1 and 27, is saved. Recomputing each method's residual directly
   from these rows must match two other results within 1e-4 relative error:
   - the held-out sums;
   - the formula that computes residuals from running sums (used for
     in-sample FER).
3. Replaying one task twice gives identical sums.
4. Mean error energy per row should generally rise with age bucket. It is
   flagged if it does not.
5. M4's in-sample FER is reported next to its held-out FER.

## Outputs

All outputs go to `<run>/NVIDIA_A100-SXM4-80GB/stale_kv_error_study/`:

- `sums/<policy>/fold<k>.pt`: the running sums from collect
- `fit/<policy>/fold<k>.npz`: the fit for held-out fold k
- `eval/<policy>/task_<index>.npz`: the held-out sums for each task
- `stale_kv_error_results.csv`, with columns:
  - policy, method, parameters per cell;
  - FER mean, CI low, CI high;
  - fold 1–4 means;
  - energy-weighted FER, in-sample FER, held-out pooled FER
- `stale_kv_error_breakdown.csv`: policy, method, layer, K/V, age bucket, row
  type, FER, rows
- `stale_kv_error_second_moment.csv`: policy, S1 median, S2 fraction, random
  baseline
- `report.json`: the outcome, the sanity checks and per-policy details
- `checks/`: the data behind sanity checks 1–3

## Out of scope

- plots;
- corrections conditioned on features of the current state;
- first-order attention updates;
- applying corrections during decoding;
- wall-clock timing;
- other models or tasks.

## Results

Added after the run.
