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

## Amendment (2026-10-31, before any data was collected)

Made after registration, but before any collect job had started. No result
of this study had been seen.

### 1. The decision reads prompt and answer rows only

Answers are about 55 tokens, so most response rows are padding. Padding is
one token repeated, and its staleness may be very regular. Pooled into each
(task, layer, K/V), padding could raise FER without saying anything about
the rows that matter.

The decision rule above (FER, its interval and fold means, S1 and S2) is
therefore evaluated on **prompt and answer rows only**. Two other row sets
are reported with the same metrics but do not count:
- padding rows alone;
- all rows (the readout as first registered).

Row type is part of the cell, so the fits are unchanged; only the readout
differs.

### 2. Pipeline checks gate the outcome

The dense K/V and the cached K/V come from different code paths (a `k_proj`
hook and a backend callback). If the paths differed consistently, say in
precision, every error would carry the same offset. M2 and M4 would learn
it and report a false "systematic". Two checks guard against this. **If
either fails on refresh_32 or refresh_256, the report marks the outcome
untrusted**, and so it does if sanity check 1, 2 or 3 fails.

- **Path check (step 0).** At step 0 both paths compute every row fresh.
  Their difference is therefore the paths themselves, not staleness. collect
  records it for every task at every layer. It passes if both hold:
  - the step-0 difference energy per row, as a share of the held-out stale
    error energy per row (prompt and answer rows), averaged over (layer
    1–27, K/V), is at most **0.02**. A consistent offset can add at most its
    energy share to FER, so a pass bounds that inflation at 0.02;
  - at layer 0, the step-0 difference energy is at most **1e-4** of the
    dense energy. Both paths apply the same modules to the same rows there.

  The share of the step-0 difference that is one consistent offset per
  (layer, K/V, head) is also reported.
- **Layer-0 check (held-out stale rows).** A committed token's layer-0 K/V
  cannot truly go stale. Both conditions must hold for K and for V:
  - layer-0 error energy per row is at most **1%** of that in layers 1–27;
  - layer-0 held-out FER of M2 and of M4 is at most **0.05**, or there is no
    layer-0 error at all.

  Structure found at layer 0 would mean the pipeline, not staleness.

### 3. Clarifications

- **Precondition.** It is confirmed in code and covered by tests:
  - each policy has its own backend and cache, reset once per task in
    `replay_task` and never rebuilt from dense
    (`tests/test_replay_compounding.py` checks that the cache drifts between
    the policy's own refreshes);
  - the report also lists stale rows by age bucket, so refresh_256 rows of
    age 32+ show the cache going stale on its own schedule.
- **Repeat check (sanity check 3).** It compares each policy's sums on the
  rerun with that same policy's sums on the first run, never one policy with
  another.
- **Null control.** The synthetic pipeline test plants pure-noise errors,
  independent of c and across tasks, in one 128-dimensional cell with 1,800
  training rows per fold. M4 fits +0.072 in-sample but scores −0.077 held
  out (95% interval below 0), and the outcome is C. So cross-validation does
  catch overfitting at this matrix size. A planted affine map is recovered
  with held-out FER above 0.9.

## Results

Run on 2026-11-01, A100 80GB, Slurm jobs 22230649–22230658. Outputs are in
`results/dlm_kv_policy/diffucoder7b_output_variance_greedy_v1/NVIDIA_A100-SXM4-80GB/stale_kv_error_study/`.

### Outcome as registered

**UNTRUSTED (a pipeline check failed): B, second-moment only.**

The layer-0 check failed on refresh_32 and refresh_256, so by the
amendment's rule the outcome is not trusted. The other pipeline checks and
sanity checks passed. Even if every check had passed, the outcome would be B:
refresh_32 is only weak on the first moment.

### First moment (prompt and answer rows, held out)

| Policy | M1 | M2 | M3 | M4 FER [95% CI] | M4 fold means | M4 energy-weighted | M4 in-sample / held-out pooled |
|---|---|---|---|---|---|---|---|
| refresh_4 | 0.001 | 0.001 | 0.003 | 0.022 [0.019, 0.024] | 0.023, 0.023, 0.022, 0.019 | 0.008 | 0.026 / 0.019 |
| refresh_32 | 0.008 | 0.011 | 0.028 | 0.145 [0.133, 0.157] | 0.160, 0.146, 0.145, 0.131 | 0.068 | 0.167 / 0.132 |
| refresh_256 | 0.046 | 0.074 | 0.147 | 0.409 [0.389, 0.429] | 0.424, 0.412, 0.411, 0.390 | 0.259 | 0.454 / 0.388 |

- refresh_32 is **weak**: the interval is above 0, but the estimate is below
  0.20. refresh_256 is **systematic**. The rule needs both, so the first
  moment is not systematic.
- Most of the predictable part needs the full 128×128 map. One scale or a
  mean shift removes almost nothing.
- Energy-weighted FER is about half the equal-weight score: the
  highest-energy layers are the least predictable.
- Answer rows are much more predictable than prompt rows. For refresh_256 at
  age 32+, M4 removes 0.50–0.78 of answer-row K error but 0.14–0.64 of
  prompt-row K error. Predictability falls with depth in both.

### Second moment (prompt and answer rows)

| Policy | S1 median relative error | S2 captured fraction (random 0.0625) |
|---|---|---|
| refresh_4 | 0.124 | 0.317 |
| refresh_32 | 0.126 | 0.358 |
| refresh_256 | 0.111 | 0.456 |

Both decision policies meet S1 ≤ 0.25 and S2 ≥ 0.25: **second-moment
systematic**. A cell's error size is predictable to about 12%, and its top-8
directions hold 36–46% of the held-out error energy, 6–7× a random subspace.

### Padding and all rows (reported, not used)

M4 FER on padding is 0.886 (refresh_32) and 0.966 (refresh_256); on all
rows it is 0.518 and 0.707. On the readout as first registered, padding alone
would have made both policies first-moment systematic. Amendment 1 is what
kept that out of the decision.

### Checks

| Check | Result |
|---|---|
| Path check (step 0) | Passed. The two paths are bitwise identical at every layer of every task. |
| Layer 0: energy share | Passed. Layer-0 error per row is about 1e-4 of layers 1–27 for K and 3e-7 for V. |
| Layer 0: FER | **Failed.** M2 is at most 0.046, but M4 removes 0.106 (K) and 0.064 (V) of layer-0 error on refresh_32, and 0.483 and 0.592 on refresh_256. The limit was 0.05. |
| 1. Control has no stale rows | Passed (0 rows) |
| 2. Raw sample | Passed (max relative error 2.5e-9) |
| 3. Repeat gives identical sums | Passed for each policy |
| 4. Error rises with age | Yes, for every policy |

### What the layer-0 failure probably means (interpretation, not tested)

- **Only answer rows have layer-0 error.** Prompt rows have none: they are
  last computed in the full forward at step 0, the same shape as dense.
  Answer rows are last computed in a forward over only a few rows, so their
  bf16 matrix products round differently from dense.
- **Why M4 can predict it.** At layer 0, c is the token's own projection, so
  c identifies the token. Rounding for a given token and forward shape is
  deterministic, so a full affine map can learn part of it. M2, the
  consistent offset the check was designed to catch, finds almost nothing.
- **This is numerics, not a bug.** It is a real difference between how a
  cache and dense compute a row, but it has nothing to do with staleness.
- **Probably too small to matter.** At layer 0 it is about 1e-4 of the error
  energy, so it cannot account for the deeper-layer FER there. The study has
  not measured how this rounding grows through deeper layers, though. Answer
  rows are both the rows that carry it and the most predictable rows, so it
  cannot yet be ruled out.

The check as registered assumed rounding is unpredictable noise. That
assumption was wrong: rounding is a deterministic function of the token, so
a model given c can learn part of it.

## Amendment 2 (2026-11-01, after the registered result, before the rounding check ran)

The registered outcome above stays as it is. This amendment defines one
follow-up check that can upgrade it to trusted. It was written after the
result was seen and before the check ran.

### What the rounding check measures

A stale row differs from dense for two reasons:
- **staleness**, which this study is about;
- **rounding**: the cached run recomputes rows in small partial forwards,
  so bf16 sums are added in a different order than in dense's full forward.

To isolate rounding:
1. At steps 1, 9, 17, … (every 8th step), for each policy, build a
   stale-free cache with one full forward. That cache must equal dense
   bitwise at every layer, as at step 0.
2. Recompute exactly the rows the policy recomputed that step, in the same
   partial forward, reading every other row from the stale-free cache. Steps
   where the policy did a full refresh have no partial forward and are
   skipped.
3. Nothing is stale, so the difference from dense is rounding alone. It is
   recorded for committed rows at every (layer, K/V, head), by row type.

### The bound

- **ρ** is rounding's share of the error energy, for each (policy, layer
  1–27, K/V), on the decision rows (prompt and answer, held out). Prompt
  rows are only ever computed in full forwards, so their rounding is zero.
  Every stale answer row is assumed to carry the rounding of one partial
  forward, at the probed average per row and head. This overstates rounding
  for rows last computed at a full refresh.
- Learning a component that holds a share ρ of the error energy can add at
  most **2√ρ − ρ** to FER: the worst case, where rounding points along the
  staleness error. If the two are uncorrelated, it adds ρ.
- **B**, the FER bound, is the mean of 2√ρ − ρ over (layer 1–27, K/V),
  averaged the same way as the primary score.
- **Worst-case first moment:** subtract B from M4's held-out FER mean, its
  interval's lower bound and each fold mean, then apply the registered rule.
- **Worst-case second moment.** The outcome also rests on the second moment,
  so its bounds are included too:
  - S1: in each (task, cell), both RMS values move by at most the rounding
    RMS q, so the relative error becomes at most (|a − b| + 2q) / (b − q);
    the median is taken over cells.
  - S2: |Pe| − |r| ≤ |Ps| and |s| ≤ |e| + |r| give the worst-case captured
    fraction (√captured − √rounding)² / (√total + √rounding)², pooled over
    rows.

### Criterion

**The registered outcome is upgraded to trusted** if all of these hold:
1. Every stale-free cache equals dense bitwise.
2. Every policy (refresh_4, refresh_32 and refresh_256) keeps its
   first-moment class and its second-moment flag under the worst case.
3. The outcome (A, B or C) is unchanged.

**If it passes,** the record reads: "UNTRUSTED as registered; upgraded to
trusted by a follow-up numerics check, defined after the original result."
**If it fails,** the outcome stays untrusted. The definitive version would
rerun the study with full-shape forwards that keep only the policy's rows,
which removes rounding instead of bounding it. That is done only if it is
needed.

Code: `stale_kv_error_study.py rounding --fold K --part P`, then
`rounding-check`. Output: `rounding_check.json`.

**Change before it ran (2026-11-02): which GPUs may measure rounding.** A100
80GB GPUs were not available, so any A100 may now run the rounding jobs, and
each fold is split into two jobs. A100 40GB is the same chip, but rounding
depends on the kernels, so a job on any GPU type other than the study's
stops unless every policy reproduces the A100 80GB replay's per-step token
TV, KL and flips exactly. A rounding measurement therefore counts only if
it was made with the study GPU's exact arithmetic. Each job records its GPU.

### Rounding check result (2026-11-02)

Slurm jobs 22330605–22330612: 48 tasks on A100 80GB, 16 on A100 40GB. Every
stale-free cache equalled dense bitwise. The 40GB tasks reproduced the 80GB
replay exactly; otherwise those jobs would have stopped.

| Policy | Rounding share ρ (mean, max over layers) | Worst-case FER term | M4 FER → worst case | First moment | Second-moment flag |
|---|---|---|---|---|---|
| refresh_4 | 0.0076, 0.049 | 0.160 | 0.022 → −0.138 | weak → **not systematic** | yes → **no** |
| refresh_32 | 0.0010, 0.0063 | 0.061 | 0.145 → 0.084 | weak → weak | yes → yes |
| refresh_256 | 0.0002, 0.0010 | 0.026 | 0.409 → 0.383 | systematic → systematic | yes → yes |

Worst-case S1 and S2: refresh_4 0.307 and 0.206; refresh_32 0.199 and 0.302;
refresh_256 0.139 and 0.425. If rounding were uncorrelated with the stale
error, the FER terms would be ρ instead: 0.008, 0.001 and 0.0002.

**The check fails.** Criterion 2 required every policy to keep its
classification, and refresh_4 does not: its worst-case rounding term is
larger than its whole FER. The outcome under the worst case is still B, and
both decision policies keep both classifications.

**Record: "B, untrusted as registered. The follow-up rounding check failed,
only on refresh_4, which is not a decision policy. On refresh_32 and
refresh_256 the classification survives worst-case rounding."** Limiting the
criterion to the decision policies would pass, but that choice would be made
after seeing the result, so the outcome is not upgraded. The full-shape
rerun described above would remove rounding instead of bounding it.

## Follow-up readouts (2026-11-01, from the registered data)

These come from the held-out files the registered run already wrote. They
are not part of any decision.

- Energy-weighted M4 FER on prompt and answer rows (also in the results
  table above): 0.008 for refresh_4, 0.068 for refresh_32 and 0.259 for
  refresh_256.
- Held-out M4 FER by age bucket on prompt and answer rows, with a 95%
  bootstrap interval over tasks. All 64 tasks have rows in every bucket
  shown. Full table: `stale_kv_error_by_age.csv`, all methods and row sets.

| Policy | Age 1 | 2–3 | 4–7 | 8–31 | 32+ |
|---|---|---|---|---|---|
| refresh_4 | 0.016 [0.014, 0.019] | 0.023 [0.021, 0.026] | | | |
| refresh_32 | 0.102 [0.088, 0.116] | 0.120 [0.107, 0.133] | 0.137 [0.123, 0.151] | 0.149 [0.137, 0.160] | |
| refresh_256 | 0.225 [0.203, 0.247] | 0.238 [0.219, 0.256] | 0.270 [0.249, 0.290] | 0.351 [0.328, 0.374] | 0.423 [0.403, 0.443] |

FER rises with age within each policy. It also differs between policies at
the same age. A refresh_256 row of age 1 (0.225) is about twice as
predictable as a refresh_32 row of age 1 (0.102). So predictability seems to
depend on how stale the rest of the cache is, not only on the row's own age.

## Exploratory decoding pilot (criteria fixed 2026-11-01, before it ran)

**Question.** Does subtracting the fitted error change decisions? Removing
41% of the error energy matters only if it also lowers decision flips.

**Setup:**
- Forced-trajectory replay on the 64 tasks, on an A100 80GB.
- refresh_4, refresh_32 and refresh_256, each run uncorrected and corrected
  at strengths 0.5 and 1.0.
- A corrected policy adds strength × (A·c + b) from M4 to every stale K and V
  row before attention reads it, in layers 1–27; layer 0 is left alone. Key
  corrections are rotated by the row's RoPE. The cache itself is never
  changed.
- Each task uses the fit of its own held-out fold.
- **Row types are computed as a decoder could:** prompt by position, padding
  if the committed token is a stop token, answer otherwise. On these 64 dense
  outputs this agrees with the study's definition (first stop token onward)
  on all 16,384 response rows, with no disagreements. Tokens are never
  re-masked, so the two definitions also agree at every step.
- **Exploratory:** the fits come from uncorrected trajectories, and a
  corrected cache drifts differently. Strength 0.5 separates "the correction
  does not help" from "it over-corrects".

**Primary metric.** Divergence events per task at refresh_256, as in the
replay proxy: token TV plus counted position changes. The comparison is
paired: corrected minus uncorrected, per task.

**Success.** At strength 0.5 or 1.0, the mean paired change has a 97.5%
bootstrap interval entirely below zero (10,000 resamples of tasks, seed 0).
Each interval is 97.5% because two strengths are tested (Bonferroni).

**Other outcomes:**
- **Over-correction:** an interval entirely above zero.
- **No success at either strength:** a fixed correction does not reduce
  divergence events even at the longest staleness. The fixed-correction
  direction is then finished at this generation length.

**Reported but not part of the criterion:**
- refresh_32 and refresh_4;
- top-1 flips, large-margin decision changes, summed token TV and mean KL.

**Pipeline requirement.** The uncorrected policies must reproduce the
earlier replay's per-step token TV, KL and flips exactly, or the pilot is
not trusted.

**Change before it ran (2026-11-02): any GPU type.** A100 80GB GPUs were
not available. Every comparison is between runs of one task on one GPU, so
the pilot now runs on A100, A40 or A6000. Each fold is split into two jobs,
and each task records its GPU. A different GPU type rounds differently, so
it cannot reproduce the A100 80GB replay exactly. The pipeline requirement
becomes:
- every policy is also run at strength 0, through the correction path with
  nothing added, and must match its uncorrected run exactly, on every task;
- tasks that ran on an A100 80GB must also match the earlier replay, as
  before.

The success criterion is unchanged.

Code: `stale_kv_correction_pilot.py run --fold K --part P`, then `report`.
Outputs: `<run>/NVIDIA_A100-SXM4-80GB/stale_kv_correction_pilot/`, for
tasks from every GPU.

### Pilot result (2026-11-02)

Slurm jobs 22330613–22330620: 16 tasks on A100 80GB, 48 on A40. **Pipeline
passed:** strength 0 matched the uncorrected run exactly on all 64 tasks,
and the A100 80GB tasks reproduced the earlier replay.

Paired change per task, corrected minus uncorrected, with 97.5% intervals:

| Policy | Uncorrected divergence events | @0.5 | @1.0 | Top-1 flips: uncorrected, @0.5, @1.0 |
|---|---|---|---|---|
| refresh_256 | 11.22 | −0.83 [−2.02, 0.22] | +1.70 [−0.22, 3.75] | 2.12, +0.00 [−0.30, 0.30], **+0.59 [0.20, 1.00]** |
| refresh_32 | 8.42 | +0.12 [−0.55, 0.86] | +0.95 [−0.03, 2.08] | 1.81, +0.00 [−0.11, 0.11], −0.03 [−0.19, 0.14] |
| refresh_4 | 2.56 | +0.19 [−0.02, 0.39] | +0.19 [−0.12, 0.50] | 0.55, +0.02 [0.00, 0.06], +0.05 [0.00, 0.11] |

**No success:** neither strength reduces divergence events at refresh_256.
Neither is harmful on the primary metric either, since both intervals
include zero. At strength 1.0, top-1 flips at refresh_256 rise, with an
interval above zero. By the rule above, a fixed correction does not reduce
divergence events even at the longest staleness. Full results:
`pilot_report.json`, `pilot_results.csv`.

## Correction diagnostic (criteria fixed 2026-11-02, before it ran)

**Question.** M4 removes part of the stale error energy on held-out data, so
why did the correction not reduce divergence? Three explanations:
- **compounding:** once corrections in earlier layers and steps change what
  the cache holds, the corrected K/V are no longer closer to dense;
- **not decision-relevant:** the corrected K/V are closer to dense, but each
  layer's attention output is not;
- **residual:** both are closer, but decisions still turn on the error M4
  cannot predict. This does not separate unpredictable error from
  second-moment structure.

**Setup.** The pilot's replay, with each policy at strengths 0 (uncorrected),
0.5 and 1.0. Per run and layer, summed over steps and tasks:
- **K/V error:** stale rows' K/V at read time against the same step's dense
  K/V, before and after the correction, before RoPE (RoPE preserves norms).
- **Attention error:** attention outputs (the o_proj input) of the rows
  computed that step, against dense.

**Ratios to the uncorrected run** (layers 1–27, 97.5% bootstrap over tasks;
below 1 means closer to dense):
- **kv:** corrected K/V error after correction ÷ uncorrected K/V error;
  keys and values are also reported separately.
- **attention:** corrected attention error ÷ uncorrected.
- **Descriptive decomposition,** kv = fit × drift. Fit is error after ÷
  before within the corrected run. Drift is the corrected run's error before
  correction ÷ the uncorrected run's.

**Classification at refresh_256, separately at each strength:**
1. kv interval not entirely below 1 → **compounding**.
2. Otherwise, attention interval not entirely below 1 → **not
   decision-relevant**.
3. Otherwise → **residual**.

refresh_32, refresh_4 and the per-layer ratios are reported, not classified.

**Pipeline requirement.** The recorders only read. So on tasks that run on
the same GPU type as the pilot did, every run's per-step token TV, KL and
flips must equal the pilot's exactly. At least one task must be checked.
The test suite also checks that the uncorrected K/V error equals the
study's own stale error measurement on a small fake model.

Code: `stale_kv_correction_diagnostic.py run --fold K --part P`, then
`report`. Outputs:
`<run>/NVIDIA_A100-SXM4-80GB/stale_kv_correction_diagnostic/`.
