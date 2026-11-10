# Masked-row reuse study: can masked rows be reused without changing decisions?

Registered 2026-11-02, before any data for it was collected. Code:
`commit_position_study.py` (Steps 0 and 1), `batched_row_timing.py`
(Step 1b). Step 2 code does not exist yet and is written only if Step 0 passes.
Model: DiffuCoder-7B-cpGRPO, greedy, 256 steps × 256 tokens, the 64-task
subset of `configs/diffucoder7b_output_variance_greedy_64tasks.yaml`, A100 80GB
(the GPU type of the existing replay, whose dense outputs this study must
reproduce exactly).

## Question

Every committed-row policy in this package recomputes all masked rows at every
step, and those rows are the compute floor: `refresh_256` uses 0.325 of dense
FLOPs, and almost all of that is masked rows. Can masked rows be reused for
`r` steps **without changing which position is committed or which token goes
there**?

A reused row has no fresh logits, so it cannot be chosen that step. Reuse is
safe only for rows the dense model would not commit in the window. So the first
question is how predictable commit positions are.

## Definitions

- **Commit event:** one position committed at one step by the dense trajectory.
- **Answer / padding commit:** by the final dense output. A commit at a
  position before the first stop token (end-of-text or pad id) is an answer
  commit; the first stop token and every position after it are padding commits.
  The mean answer is about 55 of 256 tokens, so most commits are padding.
- **Stale confidence at lag k:** the entropy confidence (the sampler's own
  negative entropy) that a position had at step `max(t-k, 0)`. A row reused for
  `r` steps has a confidence up to `r-1` steps old, so lag `k = r` is the
  relevant worst case. Positions are ranked among those still masked at step `t`.
- **Stale answer-like / padding-like:** a masked position whose stale top-1
  prediction is not / is a stop token.
- **Protected set** (recomputed every step; everything else masked is reused
  for `r` steps). Seven candidate rules, each with `m ∈ {16, 32}`:
  1. `top`: top-m by stale confidence.
  2. `top±1`: `top` plus positional neighbors (±1) of its members.
  3. `top+recent`: `top` plus positions within ±1 of any position committed in
     the last `k` steps (staleness risk: the biggest context change).
  4. `recent`: only the recent-commit neighbors (diagnostic, no top-m).
  5. `split`: top-m of the stale answer-like positions, plus top-4 of the
     stale padding-like positions (so padding cannot crowd out the code).
  6. `split±1`: `split` plus ±1 neighbors.
  7. `split+recent`: `split` plus recent-commit neighbors.
- **Coverage:** the fraction of commit events whose position is in the protected
  set at that step. Reported separately for answer and padding commits.
- **Set size:** mean number of protected positions per step (unique, masked).

## Step 0: where does the model commit? (one dense logging run, no cache)

The existing replay files keep only the final tokens and per-step aggregates, so
Step 0 needs a dense greedy run that logs, per step, the confidence and top-1
token of every masked position and the committed positions. It must reproduce
the replay's dense token ids exactly (same sampler, seeds and GPU type); a task
that does not is reported and excluded.

Reported: locality (distance of each commit to the nearest committed token, with
the prompt's last token counting as committed, and to the leftmost masked
position); the stale-confidence rank of each committed position at lags 1–8;
and coverage for the seven rules at lags 1–8.

**Selection rule (frozen here, applied to data):** tasks are split by index
parity. On the even tasks, for each reuse window `r ∈ {2, 4, 8}`, choose among
the 7 rules × `m ∈ {16, 32}` the candidate with the **smallest set size whose
answer-position coverage at lag r is ≥ 99%**. The chosen rule and `m` are
frozen for Step 2. Its coverage on the odd tasks is the held-out check and must
also be ≥ 99%.

**Step 0 gate:** stop the whole effort if, for `r = 2`, no candidate reaches
≥ 99% answer-position coverage on both halves. If only larger `r` fails, the
reuse windows that fail are dropped.

## Step 1: compute accounting (from the Step 0 logs, no GPU)

FLOPs from row counts with `compute_cost.layer_flops` (key length L, LM head on
masked rows), validated first against the replay's recorded `refresh_256` FLOPs.
For every task and policy:

- **Floor:** `refresh_256`, masked rows plus the newly committed row.
- **Truncation baseline:** once an end-of-text token has been committed at
  position `p`, rows after `p` are frozen: not recomputed, key length unchanged.
  This is the quick win and must be beaten, not just the floor.
- **Masked reuse** (selected rule, `m ∈ {16, 32}`, `r ∈ {2, 4, 8}`): protected
  rows every step, other masked rows an expected `1/r` of the time.
- **Reuse + truncation:** both.

Truncation and reuse both change decisions; Step 1 says only what they would
cost. Step 2 says whether they change decisions.

## Step 1b: batched time projection (GPU, no policy)

`batched_row_timing.py` times a plain forward of `n` recomputed rows per request
against all L keys, at batch 1, 8 and 16, plus the native dense forward. Combined
with Step 1's per-step row counts it projects wall-clock speedups. It builds no
batched backend. At batch 1 a full forward costs only about 1.2× a one-row forward
(`decoding_oracles`), so no row saving can appear as meaningful time there.

## Step 2: fidelity in forced-trajectory replay (only if Step 0 passes)

Greedy, batch 1, 64 tasks. Policies: the frozen protected rule with `m ∈ {16, 32}` ×
`r ∈ {2, 4, 8}` (six), plus the truncation baseline, each paired with
`refresh_256` run without masked reuse. Measured: divergence events per task (the
validated proxy), counted position flips reported **separately** (the new risk),
and live coverage: how often the dense model committed outside the protected set.

## Decision rule (frozen before Step 2)

A masked-reuse policy is worth pursuing only if **all** hold for at least one policy:

1. Compute fraction ≤ 0.22 (floor 0.325), **and** at most 0.85× the truncation
   baseline's compute, so it beats truncation, not just the floor.
2. Divergence events per task no worse than `refresh_256` without masked reuse:
   the upper end of the 95% paired-bootstrap interval (over tasks) on the
   difference is ≤ 0. It must also be no worse than truncation alone, by the same test.
3. Live protected-set coverage of answer commits ≥ 99%.
4. **Deferred.** Saving ≥ 15% wall-clock at batch 8–16 needs a batched backend,
   built only if 1–3 hold. Batch-1 timing is reported, not gated. Step 1b's
   projection is an indicator only: criterion 4 is plausible if projected speedup is
   ≥ 1/0.85 = 1.18 at batch 8–16.

**Stop** if Step 0 fails its gate, or if in Step 2 counted position flips exceed the
baseline's (paired 95% interval excluding zero) and grow with `r`: masked-row
stability then does not help, because the safe rows cannot be told apart.

## Caveats

- Coverage selection and the Step 2 policies use the same 64 tasks. The parity
  split gives a held-out coverage number, not a held-out fidelity number.
- Thresholds in criteria 1 and 4 (0.22, 0.85×, 1.18) and the top-4 padding set
  are proposals made before the data, not derived from it.
- The row-count FLOP model ignores norms and softmax and counts a row's attention
  at full key length even when its keys are mostly stale.

## Step 0 and Step 1 results (2026-11-03)

64 tasks, all reproducing the replay's dense outputs exactly (3,477 answer and
12,843 padding commits). The 2-task smoke run had written the first report; the
64-task report was run on 2026-11-03.

- **Commits are local.** Answer commits: distance to the nearest committed
  token median 1, p95 4, p99 10. Padding commits: median 1, p95 1.
- **Answer-position coverage** (m=16 / m=32):

  | Rule | lag 2 | lag 4 | lag 8 |
  | --- | --- | --- | --- |
  | `top` | 0.979 / 0.985 | 0.947 / 0.965 | 0.911 / 0.949 |
  | `split` | 0.998 / 1.000 | 0.992 / 0.999 | 0.980 / 0.999 |
  | `split+recent` | 0.999 / 1.000 | 0.996 / 0.999 | 0.990 / 0.999 |

- **Selection (as registered):** r=2 `top+recent` m=16 (15.6 rows, even 0.992,
  odd **0.981**); r=4 `split` m=16 (even 0.997, odd 0.985); r=8 `split` m=16
  (even 0.990, odd 0.967). **Step 0 gate: STOP**, because the r=2 choice
  misses 99% on the held-out odd tasks.
- **Step 1 compute fraction** (FLOPs / dense; projected speedup at batch 1 / 8 / 16):
  floor 0.317 (1.53 / 2.72 / 2.74); truncation 0.315 (1.53 / 2.73 / 2.76);
  reuse r=2 m=16 0.181; r=4 m=16 0.118 (1.57 / 5.75 / 6.30); r=8 m=16 0.085
  (1.57 / 6.87 / 8.13). Truncation saves almost nothing: once the end-of-text
  token is committed, the rows after it are mostly committed padding, which
  refresh_256 already reuses.

## Amendment 1 (2026-11-03, written after the Step 0 data)

Everything in this amendment was decided after seeing the Step 0 results above,
so Step 2 is not a pre-registered test of the original rule. The coverage
numbers it relies on were selected on the same 64 tasks Step 2 replays.

1. **Selection rule.** The smallest set with ≥ 99% answer coverage on *both*
   halves. Applied, it picks r=2 `split` m=16, r=4 `split+recent` m=16, r=8
   `split` m=32.
2. **Fixed instead: `split+recent` m=32 at every r** (primary), plus `split`
   m=32 at every r (secondary, the rule named in the plan). Both pass rule 1 at
   every r. Reasons, all from Step 0 data:
   - The selection rule looks only at answer commits, but padding commits
     outside the set also count as divergence events. `split` leaves 15–57
     padding commits per task outside the set. `split+recent` leaves about 5,
     at 0.1 more rows per step.
   - Estimated from the dense margins, the padding commits that `split` m=32
     misses would add about 1.6 counted position changes per task at r=2
     (0.75 at r=8). For `split+recent` m=32 the estimate is 0.00 padding and
     at most 0.03 answer position changes per task. The baseline refresh_256
     has 11.7 divergence events per task.
   - One rule at every r makes r the only variable.
3. **Truncation is dropped from Step 2**, along with its clauses in criteria 1
   and 2. It saves 0.6% of compute, so it is not a baseline worth beating.

## Step 2 design (2026-11-03, before any Step 2 data)

Code: `masked_row_reuse_replay.py`.

- **Policies.** refresh_256 is the baseline and is unchanged. Each reuse
  policy is refresh_256 plus masked-row reuse:
  - Every r steps, and at refresh_256's full steps 0 and 1, all masked rows
    are recomputed. The refreshes are synchronized, so in a batch every
    request does the same work.
  - In between, the protected positions P are computed from the policy's
    *own* last confidences and top-1 tokens, which are stale for reused rows.
    The policy recomputes the masked rows in P ∪ (P − 1), because DiffuCoder
    predicts position p from row p − 1.
  - Other masked rows keep their K/V, hidden states and logits.
  - Only positions in P can be committed. When the dense model commits
    outside P, the replay records a position flip, and the position is logged.
- **Runs.** 64 tasks in 4 parts, any allowed GPU. Each task's comparison with
  refresh_256 is paired within the same run.
- **Pipeline checks.**
  - `reuse_check_r1`, reuse with r = 1, makes refresh_256's decisions. It must
    equal refresh_256 exactly on every task, or the run stops.
  - On A100 80GB, the dense output and refresh_256 must also reproduce the
    earlier replay exactly.
- **Measured.**
  - Divergence events per task and counted position changes.
  - Top-1 flips, large-margin decision changes and KL.
  - Live coverage of answer and padding commits, over all commits.
  - Compute fraction from the backend's FLOP counter.
  - Rows per step, with projected speedups on Step 1b's curves.
- **Decision rule** (otherwise as frozen above). A policy passes if all hold:
  1. Compute fraction ≤ 0.22.
  2. The upper end of the 95% paired-bootstrap interval on its change in
     divergence events versus refresh_256 is ≤ 0.
  3. Live answer coverage ≥ 99%.

  Six policies are tested, each at 95% with no correction for multiplicity;
  criterion 2 can only be harder to meet with a correction.
  Criterion 4 stays deferred. **Stop** if, for both rules, counted position
  changes exceed refresh_256's (the interval's lower end above 0) at every r
  and grow with r.

## What the Step 1b projection does and does not include

The projection is an upper bound on batched speedup:

- **Included:**
  - each recomputed row's attention over all L keys, through every layer;
  - the LM head on those rows;
  - writing their K/V into a cache.
- **Not included:**
  - **Per-request rows.** The benchmark uses the same positions for every
    request, and there is no gather of per-request rows or scatter of K/V and
    logits back.
  - **Uneven requests.** Prompt lengths and row counts differ between
    requests, so a batch pads to its largest request. The projection averages
    each task's speedup as if every request did the same work.
  - **Choosing the protected set.** Ranking the masked rows by confidence for
    each request at each step is not timed.
  - **The sampler.** Its per-step overhead is not timed.
  - **Refresh steps as they would fall in a batch.** The projection reads
    Step 1's per-step averages, in which the other masked rows are recomputed
    1/r of the time at every step. Step 2's synchronized refreshes instead
    alternate cheap steps with full masked steps. The report recomputes the
    projection from Step 2's actual row counts.
- **Baseline.** The bar is not dense DiffuCoder. At batch 8 the comparisons
  that count are Qwen2.5-Coder and Fast-dLLM DualCache on the same GPU.

## Step 2 result (2026-11-03)

64 tasks: 32 on A100 80GB, 16 on A100 40GB, 16 on A40. **Pipeline passed**:
`reuse_check_r1` equals refresh_256 on every task, and on A100 80GB the dense
output and refresh_256 reproduce the earlier replay exactly.

Divergence change is reuse minus refresh_256, as the mean per task with its 95%
paired-bootstrap interval. Compute is FLOPs as a fraction of dense. Speedup is
projected on Step 1b's curves from the measured rows per step.

| Policy | Compute | Rows/step | Divergence/task | Change [95%] | Answer cov. | Padding cov. | Speedup b1 / b8 / b16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| refresh_256 | 0.311 | 131.8 | 11.25 | | | | 1.53 / 2.71 / 2.73 |
| `split+recent` r=2 | 0.210 | 85.6 | 11.17 | −0.08 [−0.31, +0.17] | 1.0000 | 0.990 | 1.55 / 3.82 / 4.02 |
| `split+recent` r=4 | 0.159 | 62.1 | 11.39 | +0.14 [−0.16, +0.50] | 1.0000 | 0.984 | 1.55 / 4.85 / 5.28 |
| `split+recent` r=8 | 0.133 | 50.3 | 11.67 | +0.42 [−0.08, +1.06] | 0.9992 | 0.978 | 1.56 / 5.59 / 6.27 |
| `split` r=2 | 0.210 | 85.6 | 11.33 | +0.08 [−0.12, +0.30] | 1.0000 | 0.961 | 1.55 / 3.83 / 4.02 |
| `split` r=4 | 0.159 | 62.0 | 11.33 | +0.08 [−0.20, +0.39] | 1.0000 | 0.946 | 1.55 / 4.85 / 5.29 |
| `split` r=8 | 0.133 | 50.2 | 11.81 | +0.56 [−0.03, +1.30] | 0.9989 | 0.940 | 1.56 / 5.60 / 6.29 |

**Decision as registered: STOP.**

- Every policy meets criteria 1 (compute) and 3 (coverage). None meets
  criterion 2: every interval's upper end is above 0.
- The stop condition on position changes is not met.
- Changes in top-1 flips (≤ +0.09 per task), large-margin decision changes and
  counted position changes are inside their intervals, which all include 0, at
  every r.

**Reading (not part of the decision).** Criterion 2 needs the upper end ≤ 0.
That is a superiority test, so a policy that changes nothing detectable cannot
pass it. What the data show:

- r=2 has no detectable effect. The interval bounds any harm at +0.17 events
  per task, or 1.5% of the baseline.
- r=4 bounds it at +0.50 (4.4%).
- r=8 shows a trend of about +0.4 to +0.6 events per task.
- `split+recent` and `split` do about equally well on divergence. The extra
  padding commits that `split` leaves outside the set rarely count as events.
  That fits Step 0's estimate that most of them are near-ties below the
  position margin, but this run does not test it directly.

Measured compute is higher than Step 1 projected: 0.159 against 0.118 at r=4.
The rows that predict protected positions (P − 1) add rows, as do the
synchronized refreshes. Continuing would need a non-inferiority margin. Fixing
it now would be after the data, and would need its own amendment.

## Amendment 2 (2026-11-03, written after the Step 2 data)

**Why it exists.** Criterion 2 was mis-specified. As written ("the upper end of
the 95% interval on the difference is ≤ 0") it is a superiority test: only a
policy that makes decisions *better* can pass. A policy with no effect cannot.
The intent was non-inferiority, which needs a margin, and none was set. This
amendment sets one after seeing the Step 2 data. Both verdicts are reported.

**Margin, anchored to a measurement from before Step 2.**
- The 2026-11-02 correction pilot ran refresh_256, uncorrected, on A40 for 48
  tasks. Compared with the same tasks' A100 80GB replay, GPU rounding alone
  changes a task's divergence events by 1.58 on average (SD 2.44). 33 of the
  48 tasks change, and the mean change is 0.00.
- The margin is the 95% half-width that this rounding noise alone gives a
  64-task mean: **δ = 1.96 × 2.44 / √64 = 0.60 events per task** (5.3% of
  refresh_256's 11.25). A mean change smaller than δ cannot be told apart from
  running the same policy on another GPU.
- The measurement predates Step 2. The formula that turns it into a margin
  was chosen afterwards.

**Corrected criterion 2:** the upper end of the 95% paired-bootstrap interval
on the change in divergence events versus refresh_256 is **< δ**. Criteria 1
and 3 and the stop condition are unchanged.

**Result.**

| Policy | Upper end | Registered rule | Amendment 2 |
| --- | --- | --- | --- |
| `split+recent` r=2 | +0.17 | fail | **pass** |
| `split+recent` r=4 | +0.50 | fail | **pass** |
| `split+recent` r=8 | +1.06 | fail | fail |
| `split` r=2 | +0.30 | fail | **pass** |
| `split` r=4 | +0.39 | fail | **pass** |
| `split` r=8 | +1.30 | fail | fail |

**Verdict as registered: STOP. Verdict under Amendment 2: PURSUE**, at r=2 and
r=4 under either rule. r=4 is the cheapest passing window: 0.159 of dense FLOPs
against refresh_256's 0.311.

### End-to-end check (set before it runs)

The proxy was validated on committed-row caching, not masked-row reuse, so
pass@1 is checked directly.

- **Method.** One job generates dense, refresh_256, and `split+recent` m=32 at
  r=2 and r=4, on one GPU type. Greedy decoding, the same 64 tasks, EvalPlus
  base tests. Code: `masked_row_reuse_replay.py generate`.
- **Reported for each reuse policy, against refresh_256 and against dense:**
  pass@1, tasks with changed code, and pass→fail and fail→pass counts.
- **Reading.** This is a check, not a gate. The scale is E3: swapping the
  attention kernel on dense changed 27 tasks' code, with 2 pass→fail and 2
  fail→pass, so pass@1 was unchanged.
  - **"Consistent with the proxy"** if, against refresh_256, the net change in
    passing tasks is within ±2.
  - **"Disagrees with the proxy"** if the net change is a loss of 3 or more
    tasks.
- **Check.** On A100 80GB, dense and refresh_256 must reproduce the output
  variance study's outputs exactly.

### End-to-end check result (2026-11-03)

One job on an A40 (job 22436512), so the check against the A100 80GB variants
does not apply. All four variants share the GPU type. Greedy decoding,
64 tasks, EvalPlus base tests.

| Variant | pass@1 | Compute | vs dense: changed / P→F / F→P | vs refresh_256: changed / P→F / F→P |
| --- | --- | --- | --- | --- |
| dense | 0.609 | 1.000 | | 52 / 2 / 14 |
| refresh_256 | 0.422 | 0.311 | 52 / 14 / 2 | |
| `split+recent` r=2 | 0.422 | 0.211 | 53 / 13 / 1 | 20 / 1 / 1 |
| `split+recent` r=4 | 0.422 | 0.160 | 51 / 13 / 1 | 23 / 1 / 1 |

**Consistent with the proxy.** Against refresh_256 both reuse policies change
the code of 20–23 tasks, with one pass→fail and one fail→pass each, so the net
change is 0, within the pre-stated ±2. That is the scale of E3's kernel swap
(27 changed, 2 / 2). Measured compute matches the replay (0.211 and 0.160).

On A40, dense scores 0.609 and refresh_256 0.422 (0.641 and 0.438 on A100
80GB). The gap to dense belongs to refresh_256, not to masked reuse.

## Step 3: masked reuse on depth_22 (registered 2026-11-04, before any Step 3 data)

**Why.** End to end, refresh_256 loses about 20 points of pass@1 against dense
(0.438 against 0.641 on A100 80GB). Masked reuse matched refresh_256's
fidelity, but on a lossy base. depth_22 is a near-dense base:
- pass@1 0.656 at 0.449 compute;
- 2.44 divergence events per task in the earlier replay (refresh_256 11.22,
  refresh_2 1.58).

It still recomputes every masked row at every step. Step 3 asks whether masked
reuse on top of it keeps its fidelity and cuts its compute.

**Base: `depth_22` as configured.**
- Full refresh every 8 steps, plus steps 0 and 1.
- Delay 1.
- On other steps, decoded rows recompute layers 22–28 from their cached
  layer-21 state; masked and new rows recompute every layer.

**Reuse policies.** `split+recent` m=32 at r=2 and r=4, the windows that
passed Step 2 under Amendment 2. They are otherwise unchanged from Step 2:
- On reuse steps, only masked rows outside P ∪ (P − 1) are skipped.
- Decoded rows keep depth_22's layers 22–28 recomputation.
- The base's full refreshes stay full.
- Both windows divide 8, so reuse refreshes line up with the base's.

**Runs.**
- **Replay:** `masked_row_reuse_replay.py run --base depth_22`, 64 tasks in 4
  parts, any GPU.
  - Check: `reuse_check_r1` must equal depth_22 exactly.
  - On A100 80GB, depth_22 and dense must also reproduce the earlier replay.
- **End to end:** `generate --base depth_22`, one job, generating dense,
  depth_22, r=2 and r=4.
- Compute is measured by the backend's FLOP counter, so no separate Step 1
  accounting is needed.
- The projected speedup now times layers 1–21 at the full-depth row count and
  layers 22–28 at the total row count.

**Decision (Amendment 3).** A window holds on depth_22 if all three are met:
1. Replay compute is at most 0.85 × depth_22's compute.
2. Live answer coverage is at least 99%.
3. **End to end, paired per problem against depth_22:** the net change in
   passing tasks (fail→pass minus pass→fail) is at least −2. A loss of 3 or
   more fails. This is the E3 scale: the kernel swap changed 27 tasks, 2 / 2.

**Reported, not gating.** The replay divergence change against depth_22, with
its 95% interval, against δ = 0.60. δ was anchored on refresh_256's rounding
noise and is 25% of depth_22's 2.44 events, so on this base it is loose. The
pass/fail comparison is the primary fidelity test, as discussed.

**Then.**
- If either window holds, the batched spec targets depth_22 plus reuse.
  Supporting depth_22 needs cached layer-21 hidden states for decoded rows,
  plus per-request depth bookkeeping; the spec will scope that cost.
- If neither holds, refresh_256 plus reuse is built as a mechanism
  demonstration only.

## Step 3 result (2026-11-04)

**Replay.** 64 tasks, 16 each on A100 80GB, A100 40GB, A40 and RTX A6000.
Pipeline passed: `reuse_check_r1` equals depth_22 on every task, and on A100
80GB depth_22 and dense reproduce the earlier replay.

| Policy | Compute | Share of depth_22 | Divergence/task | Change [95%] | Answer cov. | Padding cov. | Speedup b1 / b8 / b16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| depth_22 | 0.449 | | 2.66 | | | | 1.42 / 1.97 / 1.96 |
| + `split+recent` r=2 | 0.349 | 0.78 | 2.77 | +0.11 [−0.06, +0.31] | 1.0000 | 0.991 | 1.44 / 2.50 / 2.55 |
| + `split+recent` r=4 | 0.297 | 0.66 | 2.78 | +0.12 [−0.05, +0.31] | 1.0000 | 0.985 | 1.45 / 2.90 / 3.00 |

Top-1 flips are unchanged (+0.00 and +0.02 per task). The small increase is in
position changes.

**End to end.** One job on an RTX A6000 (job 22444833), so the check against
the A100 80GB variants does not apply.

| Variant | pass@1 | Compute | vs depth_22: changed / P→F / F→P | Net |
| --- | --- | --- | --- | --- |
| dense | 0.609 | 1.000 | 32 / 6 / 2 | −4 |
| depth_22 | 0.672 | 0.449 | | |
| + r=2 | 0.656 | 0.349 | 12 / 1 / 0 | −1 |
| + r=4 | 0.609 | 0.297 | 14 / 5 / 1 | −4 |

**Decision (Amendment 3).**
- **r=2 holds:** 0.78× depth_22's compute, 100% answer coverage, and a net
  change of −1.
- **r=4 fails criterion 3:** a net change of −4, against the pre-stated
  failure line of 3 or more lost tasks. It passes criteria 1 and 2.

Reported, not part of the decision:
- r=4's loss rests on 6 discordant tasks (5 to 1; two-sided sign test
  p ≈ 0.22).
- 4 of its 5 lost tasks also change code under E3's kernel swap.
- 3 of the 5 fail under dense on this GPU.
- Dense itself is at a net −4 against depth_22 on this GPU.

So the loss may be noise on fragile tasks, but the registered rule counts it,
and r=4 is not carried forward on depth_22.

**Outcome.** The batched spec targets **depth_22 + `split+recent` m=32 r=2**.
Its projected speedup at batch 8 is 2.50× over dense, against 1.97× for
depth_22 alone, so reuse adds about 1.27×. Most of what depth_22 still spends
is outside masked reuse's reach:
- decoded rows recomputed in layers 22–28 at every step (about 208–232 rows
  per step in total);
- a full forward every 8 steps.

## Amendment 4: batched evaluation primary (2026-11-04, after Step 3)

Approved by Michael after reading Step 3: the primary batched fidelity comparison
is **depth_22 + split+recent m=32 r=2 versus depth_22**. This replaces r=4 as
the proposed primary for the batched study because r=4 failed Step 3's registered
rule. Step 3's decision and results remain unchanged. Backend support for r=4
and other positive integer windows is retained; support is not evidence of
fidelity and does not restore r=4 as a passing policy.

The full-dataset gate uses the paired per-problem drop in base-test pass@1,
pooled over HumanEval (164) and MBPP (378): the one-sided 97.5% upper confidence
bound must be <= 0.03. Use a paired percentile bootstrap over problems (100,000
resamples, seed 0). Per-dataset results, other windows, comparisons to dense,
and EvalPlus plus-test results are descriptive. The 64-task Step 3 rule remains
separate. Report dense batched/unbatched discordance as the empirical numerical
noise reference; the E3 discordance is a planning estimate, not an assumed result.

Batch manifests sort by tokenized prompt length (task ID breaks length ties),
then chunk by batch size. Every arm uses the same manifest. The last batch may
be smaller; report its actual size and do not count it as a full-batch timing
sample. Keep batch sizes separate in inference and reporting.

Spec corrections, not changes to the validated policy: newly decoded rows get
one subsequent full-depth forward; preserve the float32 transfer schedule,
including zero/multiple-commit steps; stable sorting applies to protected-set
ranking, while sampler topk remains unchanged. Both arms run the output head
on all masked-position predecessor states. Persistent hidden caches retain only
final states and the configured depth boundary; intermediate workspace is transient.

**Speed gate remains pending registration before GPU timing.** Step 3 projects
2.50x dense at batch 8 for r=2, versus 1.97x for depth_22 alone. These projections
do not establish a measured target. No GPU timing or fidelity validation of the
new backend has been run as part of implementation. Batch-1 CPU tests do not
satisfy the real-model A100 bf16 correctness ladder.
