# Masked-row smoothness study: CADD versus DiffuCoder

Registered 2026-11-02, before any data was collected. Code:
`masked_row_smoothness_study.py`. Upstream CADD code: `ml-cadd/` (copied
unmodified; commit in `ml-cadd/SOURCE.json`).

## Question

DiffuCoder's cache policies always recompute masked rows, because a masked
row's context changes every step. CADD feeds masked rows `embed(MASK) + z`,
where `z` is a flow-matching state that is re-estimated every step. Does that
make a masked row's intermediate K, V and hidden state change **less** from one
step to the next than in DiffuCoder, so that masked rows could be cached?

This is an open comparison, not a confirmation test. CADD's masked-row input
changes every step (`z = (1-s)·ẑ₀ + s·fresh noise`), and early on it is mostly
noise, so CADD could just as well be **less** stable than DiffuCoder, above all
in early layers. DiffuCoder's masked rows are not uncacheable either
(Fast-dLLM's DualCache reuses them). The fair question is whether CADD's masked
rows are more reusable than DiffuCoder's.

Nothing is cached or corrected. Each arm is stock uncached generation with
forward hooks.

## Arms

| Arm | Weights | Sampler |
| --- | --- | --- |
| `cadd_on` | `apple/CADD-Base-7B` | CADD, `use_cadd=True`, `weighted` |
| `cadd_off` | `apple/CADD-Base-7B` | same code, `use_cadd=False`: `z = 0`, masked input is the plain MASK embedding |
| `diffucoder_base` | `apple/DiffuCoder-7B-Base`, the model CADD was initialized from | native DiffuCoder |
| `diffucoder` | DiffuCoder-7B-cpGRPO (existing checkpoint) | native DiffuCoder |

`cadd_off` is the attribution control: same weights, same prompts, no
continuous latent. `cadd_on` against `cadd_off` isolates the effect of `z`;
`cadd_on` against `diffucoder_base` is the registered comparison. CADD was
trained for 95k further steps from DiffuCoder-7B-Base, so `diffucoder_base`
against `cadd_off` reports what that extra training changed with no latent.
`diffucoder` (the RL-tuned cpGRPO checkpoint used by the other studies in this
package) is a secondary arm; `cadd_on` against it is reported but not gating.

Amendment 1 (2026-11-02, before any data): the registered comparator was first
written as the cpGRPO checkpoint, the only DiffuCoder on disk. DiffuCoder-7B-Base
was then downloaded and replaces it, since it is the model CADD descends from.
Thresholds and metric are unchanged.

## Setup

- The 64-task HumanEval+/MBPP+ subset of
  `data/code/humaneval_mbpp_64_strict_python_seed0.jsonl`. The prompt is the
  task's `original_prompt` (function signature and docstring), tokenized
  as plain text with no chat template, which is the completion-style prompt a
  Base model expects. All three arms see identical prompt token ids.
- 256 steps, 256 new tokens, entropy order, one token committed per step,
  temperature 0, top_p 1. Noise seed `seed + task index` for `cadd_on`.
  The recorded number of masked rows at every step must equal
  `256 - step` for every arm; a run that violates it is invalid.
- Same GPU type for every arm (A100 80GB), bf16.
- Hooks record, for every layer, the `k_proj` output (pre-RoPE; RoPE is a
  per-position rotation, so for the same position the norm of a step-to-step
  difference is identical before and after it), the `v_proj` output, and the
  decoder layer's output hidden state.

## Measure

For a row at step `t` and lag `k` (k = 1, 4, 8) that has the same position at
step `t-k`:

    change = ‖x_t − x_{t−k}‖₂ / ‖x_t‖₂      (per layer, per kind K / V / H)

Rows are grouped as **masked** (masked at step `t`, hence also at `t-k`),
**committed** (a generated position already decoded at `t-k`) and **prompt**.
Only masked is registered. The others give context: prompt change is the drift
that existing stale-K/V policies already reuse across.

The saved per-step statistic is the mean of `change` over rows of the group,
together with the row count `n_t` (the number of masked rows). The
**compute-weighted** value for a task is `Σ_t n_t · mean_t / Σ_t n_t`, so a
smooth late phase with few rows cannot hide a rough early phase that holds
most of the rows and most of the cost. An unweighted per-step mean and a
breakdown by noise level `s = 1 − t/256` (bins of 0.25) are reported for
inspection.

## Registered metric and decision

**Deep layers** are layers 14–27 (the second half of 28). **Primary metric**:
for a task, the lag-1 compute-weighted `change` of masked rows, averaged over K
and V and over the deep layers. The arm value is the mean over the 64 tasks.

Comparison: `R = primary(cadd_on) / primary(diffucoder_base)`, with a 95%
bootstrap interval from 10,000 resamples of the 64 tasks (the same tasks are
resampled in both arms).

- **Continue** only if `R ≤ 0.80` **and** the upper end of its 95% interval is
  below 1.0.
- Otherwise **stop**: the "smooth continuous component" direction has nothing
  to stand on, and no backend port is built.

Reported but not gating: the same ratio at lags 4 and 8; the hidden-state
version (H); every layer separately; the shallow layers 0–13; `cadd_on`
against `cadd_off`; `cadd_on` against `diffucoder`; `diffucoder_base` against `cadd_off`; and the row groups other
than masked. If the registered comparison passes but `cadd_on / cadd_off` is
not also below 1, the gain belongs to CADD's training, not to its latent, and
the finding is reported that way.

## Sanity checks before the comparison counts

1. `cadd_on` generations must be coherent code (inspected on a sample), and
   CADD must reproduce its reported scores on a slice of HumanEval and MBPP at
   its own settings (512 steps, 512 tokens, temperature 0.1, `entropy`,
   `weighted`) before any comparison is read.
2. Masked-row counts equal `256 - step` at every step of every task.
3. `cadd_off` must still produce coherent code. If the model collapses
   without `z`, that arm is a poor control and the report says so.

## Not covered

Fixed rather than fresh noise per step would make `z` smoother, but it is a
different sampler from the one CADD was trained for and is not measured here.
The cost of the `probs @ embed.weight` product in weighted mode (about 1 GB of
reads per step at batch size 1) does not affect this diagnostic and must be
counted in any later cost model.
