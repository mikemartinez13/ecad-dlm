# Masked-row reuse study: can masked rows be reused without changing decisions?

Designed 2026-11-02, before any data for it was collected. Code: `commit_position_study.py` (Steps 0 and 1), `batched_row_timing.py` (Step 1b) and `masked_row_reuse_teacher_forcing.py` (Steps 2 and 3).

Model and data: DiffuCoder-7B-cpGRPO, greedy, 256 steps × 256 tokens, on the 64-task subset of `configs/diffucoder7b_output_variance_greedy_64tasks.yaml`. GPU: A100 80GB, the GPU type of the existing teacher forcing, whose dense outputs this study reproduces exactly.

## Question

Every committed-row policy in this package recomputes all masked rows at every step, and those rows are the compute floor: `refresh_256` uses 0.325 of dense FLOPs, almost all of it masked rows. Can masked rows be reused for `r` steps **without changing which position is committed or which token goes there**?

A reused row has no fresh logits, so it cannot be chosen that step. Reuse is safe only for rows the dense model would not commit in the window. So the first question is how predictable commit positions are.

## Definitions

- **Commit event:** one position committed at one step by the dense trajectory.
- **Answer and padding commits:** defined by the final dense output. A commit at a position before the first stop token (end-of-text or pad id) is an answer commit. The first stop token and every position after it are padding commits. The mean answer is about 55 of 256 tokens, so most commits are padding.
- **Stale confidence at lag k:** the entropy confidence (the sampler's own negative entropy) that a position had at step `max(t-k, 0)`. A row reused for `r` steps has a confidence up to `r-1` steps old, so lag `k = r` is the relevant worst case. Positions are ranked among those still masked at step `t`.
- **Stale answer-like and padding-like positions:** masked positions whose stale top-1 prediction is not, or is, a stop token.
- **Protected set:** the masked positions recomputed every step; every other masked position is reused for `r` steps. Seven candidate rules, each with `m ∈ {16, 32}`:
  1. `top`: top-m by stale confidence.
  2. `top±1`: `top` plus the positional neighbors (±1) of its members.
  3. `top+recent`: `top` plus positions within ±1 of any position committed in the last `k` steps, where the context changed most.
  4. `recent`: only the recent-commit neighbors (diagnostic, no top-m).
  5. `split`: top-m of the stale answer-like positions, plus top-4 of the stale padding-like positions, so padding cannot crowd out the code.
  6. `split±1`: `split` plus ±1 neighbors.
  7. `split+recent`: `split` plus recent-commit neighbors.
- **Coverage:** the fraction of commit events whose position is in the protected set at that step, reported separately for answer and padding commits.
- **Set size:** the mean number of protected positions per step (unique, masked).

## Step 0: where does the model commit? (one dense logging run, no cache)

The existing teacher forcing files keep only the final tokens and per-step aggregates. Step 0 therefore needs a dense greedy run that logs, per step, the confidence and top-1 token of every masked position and the committed positions. It reproduces the teacher forcing's dense token ids exactly (same sampler, seeds and GPU type). A task that does not is reported and excluded.

**Reported:**

- locality: the distance of each commit to the nearest committed token (the prompt's last token counts as committed), and to the leftmost masked position;
- the stale-confidence rank of each committed position at lags 1–8;
- coverage for the seven rules at lags 1–8.

**Selection rule (fixed here, applied to data).** Tasks are split by index parity. On the even tasks, for each reuse window `r ∈ {2, 4, 8}`, the candidate chosen among the 7 rules × `m ∈ {16, 32}` is the one with the **smallest set size whose answer-position coverage at lag r is ≥ 99%**. Its coverage on the odd tasks is the held-out number.

## Step 1: compute accounting (from the Step 0 logs, no GPU)

FLOPs come from row counts with `utils.compute_cost.layer_flops` (key length L, LM head on masked rows), checked first against the teacher forcing's recorded `refresh_256` FLOPs. For every task and policy:

- **Floor:** `refresh_256`: masked rows plus the newly committed row.
- **Truncation baseline:** once an end-of-text token has been committed at position `p`, rows after `p` are frozen: not recomputed, key length unchanged. This is the quick win to compare against, besides the floor.
- **Masked reuse** (selected rule, `m ∈ {16, 32}`, `r ∈ {2, 4, 8}`): protected rows every step, other masked rows an expected `1/r` of the time.
- **Reuse + truncation:** both.

Truncation and reuse both change decisions; Step 1 says only what they would cost, and Step 2 measures whether they change decisions.

## Step 1b: batched time projection (GPU, no policy)

`batched_row_timing.py` times a plain forward of `n` recomputed rows per request against all L keys, at batch 1, 8 and 16, plus the native dense forward. Combined with Step 1's per-step row counts, it projects wall-clock speedups. It builds no batched backend. At batch 1 a full forward costs only about 1.2× a one-row forward (`decoding_oracles`), so no row saving can show up as meaningful time there.

## Step 2: fidelity in teacher forcing

**Setup.** Greedy, batch 1, 64 tasks.

**Policies.** The protected rule with `m ∈ {16, 32}` × `r ∈ {2, 4, 8}` (six policies), plus the truncation baseline. Each is paired with `refresh_256` run without masked reuse.

**Measured:**

- divergence events per task (the proxy);
- counted position flips, reported **separately**, since they are the new risk;
- live coverage: how often the dense model committed outside the protected set.

**Also planned before the data:**

- compute against the floor (0.325) and against truncation;
- a later batched wall-clock measurement at batch 8–16, which would need a batched backend.

## Caveats

- Coverage selection and the Step 2 policies use the same 64 tasks. The parity split gives a held-out coverage number, not a held-out fidelity number.
- The top-4 padding set and the compute targets noted in planning (0.22 of dense, 0.85× truncation, a projected 1.18× at batch 8–16) were proposals made before the data, not derived from it.
- The row-count FLOP model ignores norms and softmax, and counts a row's attention at full key length even when its keys are mostly stale.

## Step 0 and Step 1 results (2026-11-03)

64 tasks, all reproducing the teacher forcing's dense outputs exactly: 3,477 answer and 12,843 padding commits. The 2-task smoke run had written the first report; the 64-task report was run on 2026-11-03.

- **Commits are local.** Answer commits: distance to the nearest committed token median 1, p95 4, p99 10. Padding commits: median 1, p95 1.
- **Answer-position coverage** (m=16 / m=32):

  | Rule | lag 2 | lag 4 | lag 8 |
  | --- | --- | --- | --- |
  | `top` | 0.979 / 0.985 | 0.947 / 0.965 | 0.911 / 0.949 |
  | `split` | 0.998 / 1.000 | 0.992 / 0.999 | 0.980 / 0.999 |
  | `split+recent` | 0.999 / 1.000 | 0.996 / 0.999 | 0.990 / 0.999 |

- **Selected by the even-task rule:**

  | Window | Rule | Even-task coverage | Odd-task coverage |
  | --- | --- | --- | --- |
  | r=2 | `top+recent` m=16 (15.6 rows) | 0.992 | 0.981 |
  | r=4 | `split` m=16 | 0.997 | 0.985 |
  | r=8 | `split` m=16 | 0.990 | 0.967 |

  All three fall below 99% on the held-out odd tasks.
- **Step 1 compute fraction** (FLOPs / dense; projected speedup at batch 1 / 8 / 16):

  | Policy | Compute | Speedup b1 / b8 / b16 |
  | --- | --- | --- |
  | Floor | 0.317 | 1.53 / 2.72 / 2.74 |
  | Truncation | 0.315 | 1.53 / 2.73 / 2.76 |
  | Reuse r=2 m=16 | 0.181 | |
  | Reuse r=4 m=16 | 0.118 | 1.57 / 5.75 / 6.30 |
  | Reuse r=8 m=16 | 0.085 | 1.57 / 6.87 / 8.13 |

  Truncation saves almost nothing: once the end-of-text token is committed, the rows after it are mostly committed padding, which refresh_256 already reuses.

## Amendment 1 (2026-11-03, written after the Step 0 data)

Everything in this amendment was decided after seeing the Step 0 results above. The coverage numbers it relies on were selected on the same 64 tasks Step 2 teacher-forces.

1. **Selection rule.** The smallest set with ≥ 99% answer coverage on *both* halves. Applied, it picks r=2 `split` m=16, r=4 `split+recent` m=16 and r=8 `split` m=32.
2. **Fixed instead: `split+recent` m=32 at every r** (primary), plus `split` m=32 at every r (secondary, the rule named in the plan). Both reach 99% on both halves at every r. Reasons, all from Step 0 data:
   - The selection rule looks only at answer commits, but padding commits outside the set also count as divergence events. `split` leaves 15–57 padding commits per task outside the set; `split+recent` leaves about 5, at 0.1 more rows per step.
   - Estimated from the dense margins, the padding commits that `split` m=32 misses would add about 1.6 counted position changes per task at r=2 (0.75 at r=8). For `split+recent` m=32 the estimate is 0.00 padding and at most 0.03 answer position changes per task. The baseline refresh_256 has 11.7 divergence events per task.
   - One rule at every r makes r the only variable.
3. **Truncation is dropped from Step 2.** It saves 0.6% of compute, so it is not a useful comparison.

## Step 2 design (2026-11-03, before any Step 2 data)

Code: `masked_row_reuse_teacher_forcing.py`.

**Policies.** refresh_256 is the baseline and is unchanged. Each reuse policy is refresh_256 plus masked-row reuse:

- Every r steps, and at refresh_256's full steps 0 and 1, all masked rows are recomputed. The refreshes are synchronized, so in a batch every request does the same work.
- In between, the protected positions P are computed from the policy's *own* last confidences and top-1 tokens, which are stale for reused rows. The policy recomputes the masked rows in P ∪ (P − 1), because DiffuCoder predicts position p from row p − 1.
- Other masked rows keep their K/V, hidden states and logits.
- Only positions in P can be committed. When the dense model commits outside P, the teacher forcing records a position flip and logs the position.

**Runs.** 64 tasks in 4 parts, on any allowed GPU. Each task's comparison with refresh_256 is paired within the same run.

**Pipeline checks** (correctness, not fidelity):

- `reuse_check_r1`, reuse with r = 1, makes refresh_256's decisions, so it equals refresh_256 exactly on every task, or the run stops.
- On A100 80GB, the dense output and refresh_256 reproduce the earlier teacher forcing exactly.

**Measured:**

- divergence events per task, and their change against refresh_256 with a 95% paired-bootstrap interval;
- counted position changes;
- top-1 flips, large-margin decision changes and KL;
- live coverage of answer and padding commits, over all commits;
- compute fraction, from the backend's FLOP counter;
- rows per step, with projected speedups on Step 1b's curves.

Six policies are measured, each with a 95% interval and no correction for multiplicity.

## What the Step 1b projection does and does not include

The projection is an upper bound on batched speedup.

**Included:**

- each recomputed row's attention over all L keys, through every layer;
- the LM head on those rows;
- writing their K/V into a cache.

**Not included:**

- **Per-request rows.** The benchmark uses the same positions for every request. There is no gather of per-request rows, and no scatter of K/V and logits back.
- **Uneven requests.** Prompt lengths and row counts differ between requests, so a batch pads to its largest request. The projection averages each task's speedup as if every request did the same work.
- **Choosing the protected set.** Ranking the masked rows by confidence for each request at each step is not timed.
- **The sampler.** Its per-step overhead is not timed.
- **Refresh steps as they would fall in a batch.** The projection reads Step 1's per-step averages, in which the other masked rows are recomputed 1/r of the time at every step. Step 2's synchronized refreshes instead alternate cheap steps with full masked steps. The report recomputes the projection from Step 2's actual row counts.

**Comparison point.** The relevant comparison is not dense DiffuCoder. At batch 8, it is Qwen2.5-Coder and Fast-dLLM DualCache on the same GPU.

## Step 2 result (2026-11-03)

64 tasks: 32 on A100 80GB, 16 on A100 40GB and 16 on A40. The pipeline checks held: `reuse_check_r1` equals refresh_256 on every task, and on A100 80GB the dense output and refresh_256 reproduce the earlier teacher forcing exactly.

Divergence change is reuse minus refresh_256, as the mean per task with its 95% paired-bootstrap interval. Compute is FLOPs as a fraction of dense. Speedup is projected on Step 1b's curves from the measured rows per step.

| Policy | Compute | Rows/step | Divergence/task | Change [95%] | Answer cov. | Padding cov. | Speedup b1 / b8 / b16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| refresh_256 | 0.311 | 131.8 | 11.25 | | | | 1.53 / 2.71 / 2.73 |
| `split+recent` r=2 | 0.210 | 85.6 | 11.17 | −0.08 [−0.31, +0.17] | 1.0000 | 0.990 | 1.55 / 3.82 / 4.02 |
| `split+recent` r=4 | 0.159 | 62.1 | 11.39 | +0.14 [−0.16, +0.50] | 1.0000 | 0.984 | 1.55 / 4.85 / 5.28 |
| `split+recent` r=8 | 0.133 | 50.3 | 11.67 | +0.42 [−0.08, +1.06] | 0.9992 | 0.978 | 1.56 / 5.59 / 6.27 |
| `split` r=2 | 0.210 | 85.6 | 11.33 | +0.08 [−0.12, +0.30] | 1.0000 | 0.961 | 1.55 / 3.83 / 4.02 |
| `split` r=4 | 0.159 | 62.0 | 11.33 | +0.08 [−0.20, +0.39] | 1.0000 | 0.946 | 1.55 / 4.85 / 5.29 |
| `split` r=8 | 0.133 | 50.2 | 11.81 | +0.56 [−0.03, +1.30] | 0.9989 | 0.940 | 1.56 / 5.60 / 6.29 |

**Reading:**

- Every reuse policy is below 0.22 of dense compute and keeps at least 99.9% answer coverage.
- Every divergence interval includes 0. Changes in top-1 flips (≤ +0.09 per task), large-margin decision changes and counted position changes are also inside intervals that include 0, at every r.
- r=2 has no detectable effect: the interval bounds any harm at +0.17 events per task, or 1.5% of the baseline.
- r=4 bounds it at +0.50 (4.4%).
- r=8 shows a trend of about +0.4 to +0.6 events per task.
- `split+recent` and `split` do about equally well on divergence. The extra padding commits that `split` leaves outside the set rarely count as events. That fits Step 0's estimate that most of them are near-ties below the position margin, but this run does not test it directly.
- Measured compute is higher than Step 1 projected: 0.159 against 0.118 at r=4. The rows that predict protected positions (P − 1) add rows, and so do the synchronized refreshes.

## Amendment 2 (2026-11-03, written after the Step 2 data): a noise scale for divergence changes

To read "no detectable effect" against something, this amendment sets the scale that GPU rounding alone gives. It was written after seeing the Step 2 data.

- The 2026-11-02 correction pilot ran refresh_256, uncorrected, on A40 for 48 tasks. Against the same tasks' A100 80GB teacher forcing, GPU rounding alone changes a task's divergence events by 1.58 on average (SD 2.44). 33 of the 48 tasks change, and the mean change is 0.00.
- The 95% half-width that this rounding noise alone gives a 64-task mean is **δ = 1.96 × 2.44 / √64 = 0.60 events per task**, 5.3% of refresh_256's 11.25. A mean change smaller than δ cannot be told apart from running the same policy on another GPU.
- The measurement predates Step 2; the formula that turns it into δ was chosen afterwards.

| Policy | Upper end of the 95% interval | Below δ = 0.60 |
| --- | --- | --- |
| `split+recent` r=2 | +0.17 | yes |
| `split+recent` r=4 | +0.50 | yes |
| `split+recent` r=8 | +1.06 | no |
| `split` r=2 | +0.30 | yes |
| `split` r=4 | +0.39 | yes |
| `split` r=8 | +1.30 | no |

At r=2 and r=4, under either rule, any effect on divergence is within GPU-rounding scale. r=4 is the cheapest such window: 0.159 of dense FLOPs, against refresh_256's 0.311. These two windows were carried into the end-to-end check and Step 3.

### End-to-end check (designed before it ran)

The proxy was validated on committed-row caching, not on masked-row reuse, so pass@1 is checked directly.

- **Method.** One job generates dense, refresh_256, and `split+recent` m=32 at r=2 and r=4, on one GPU type. Greedy decoding, the same 64 tasks, EvalPlus base tests. Code: `masked_row_reuse_teacher_forcing.py generate`.
- **Reported for each reuse policy, against refresh_256 and against dense:** pass@1, tasks with changed code, and pass→fail and fail→pass counts.
- **Scale for reading it.** E3: swapping the attention kernel on dense changed 27 tasks' code, with 2 pass→fail and 2 fail→pass, so pass@1 was unchanged.
- **Reproduction.** On A100 80GB, dense and refresh_256 reproduce the output variance study's outputs exactly.

### End-to-end check result (2026-11-03)

One job on an A40 (job 22436512), so the reproduction against the A100 80GB variants does not apply. All four variants share the GPU type. Greedy decoding, 64 tasks, EvalPlus base tests.

| Variant | pass@1 | Compute | vs dense: changed / P→F / F→P | vs refresh_256: changed / P→F / F→P |
| --- | --- | --- | --- | --- |
| dense | 0.609 | 1.000 | | 52 / 2 / 14 |
| refresh_256 | 0.422 | 0.311 | 52 / 14 / 2 | |
| `split+recent` r=2 | 0.422 | 0.211 | 53 / 13 / 1 | 20 / 1 / 1 |
| `split+recent` r=4 | 0.422 | 0.160 | 51 / 13 / 1 | 23 / 1 / 1 |

Against refresh_256, both reuse policies change the code of 20–23 tasks, with one pass→fail and one fail→pass each, so the net change is 0. That is the scale of E3's kernel swap (27 changed, 2 / 2), and it agrees with the proxy. Measured compute matches the teacher forcing (0.211 and 0.160).

On A40, dense scores 0.609 and refresh_256 0.422 (0.641 and 0.438 on A100 80GB). The gap to dense belongs to refresh_256, not to masked reuse.

## Step 3: masked reuse on depth_22 (designed 2026-11-04, before any Step 3 data)

**Why.** End to end, refresh_256 loses about 20 points of pass@1 against dense (0.438 against 0.641 on A100 80GB). Masked reuse matched refresh_256's fidelity, but on a lossy base. depth_22 is a near-dense base:

- pass@1 0.656 at 0.449 compute;
- 2.44 divergence events per task in the earlier teacher forcing (refresh_256 11.22, refresh_2 1.58).

It still recomputes every masked row at every step. Step 3 asks whether masked reuse on top of it keeps its fidelity and cuts its compute.

**Base: `depth_22` as configured.**

- Full refresh every 8 steps, plus steps 0 and 1.
- Delay 1.
- On other steps, decoded rows recompute layers 22–28 from their cached layer-21 state; masked and new rows recompute every layer.

**Reuse policies.** `split+recent` m=32 at r=2 and r=4, otherwise unchanged from Step 2:

- On reuse steps, only masked rows outside P ∪ (P − 1) are skipped.
- Decoded rows keep depth_22's layers 22–28 recomputation.
- The base's full refreshes stay full.
- Both windows divide 8, so reuse refreshes line up with the base's.

**Runs.**

- **Teacher forcing:** `masked_row_reuse_teacher_forcing.py run --base depth_22`, 64 tasks in 4 parts, on any GPU.
  - Check: `reuse_check_r1` equals depth_22 exactly.
  - On A100 80GB, depth_22 and dense also reproduce the earlier teacher-forced run.
- **End to end:** `generate --base depth_22`, in one job generating dense, depth_22, r=2 and r=4.
- Compute is measured by the backend's FLOP counter, so no separate Step 1 accounting is needed.
- The projected speedup now times layers 1–21 at the full-depth row count and layers 22–28 at the total row count.

**Measured (Amendment 3):**

1. Teacher forcing compute as a share of depth_22's.
2. Live answer coverage.
3. End to end, paired per problem against depth_22: lost and gained tasks, and the net change.
4. The teacher forcing divergence change against depth_22, with its 95% interval, next to δ = 0.60. δ was anchored on refresh_256's rounding noise and is 25% of depth_22's 2.44 events, so on this base it is a loose scale.

The planned use: if a window keeps depth_22's fidelity, the batched spec targets depth_22 plus reuse. Supporting depth_22 needs cached layer-21 hidden states for decoded rows, plus per-request depth bookkeeping, and the spec would scope that cost. Otherwise refresh_256 plus reuse would be built only to demonstrate the mechanism.

## Step 3 result (2026-11-04)

**Teacher forcing.** 64 tasks, 16 each on A100 80GB, A100 40GB, A40 and RTX A6000. The pipeline checks held: `reuse_check_r1` equals depth_22 on every task, and on A100 80GB depth_22 and dense reproduce the earlier teacher-forced run.

| Policy | Compute | Share of depth_22 | Divergence/task | Change [95%] | Answer cov. | Padding cov. | Speedup b1 / b8 / b16 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| depth_22 | 0.449 | | 2.66 | | | | 1.42 / 1.97 / 1.96 |
| + `split+recent` r=2 | 0.349 | 0.78 | 2.77 | +0.11 [−0.06, +0.31] | 1.0000 | 0.991 | 1.44 / 2.50 / 2.55 |
| + `split+recent` r=4 | 0.297 | 0.66 | 2.78 | +0.12 [−0.05, +0.31] | 1.0000 | 0.985 | 1.45 / 2.90 / 3.00 |

Top-1 flips are unchanged (+0.00 and +0.02 per task); the small increase is in position changes.

**End to end.** One job on an RTX A6000 (job 22444833), so the reproduction against the A100 80GB variants does not apply.

| Variant | pass@1 | Compute | vs depth_22: changed / P→F / F→P | Net |
| --- | --- | --- | --- | --- |
| dense | 0.609 | 1.000 | 32 / 6 / 2 | −4 |
| depth_22 | 0.672 | 0.449 | | |
| + r=2 | 0.656 | 0.349 | 12 / 1 / 0 | −1 |
| + r=4 | 0.609 | 0.297 | 14 / 5 / 1 | −4 |

**Reading.**

- **r=2:** 0.78× depth_22's compute, 100% answer coverage, and a net change of −1.
- **r=4:** 0.66× the compute and 100% coverage, but a net change of −4. That loss rests on 6 discordant tasks (5 to 1; two-sided sign test p ≈ 0.22). 4 of its 5 lost tasks also change code under E3's kernel swap, and 3 of the 5 fail under dense on this GPU. Dense itself is at a net −4 against depth_22 on this GPU. So the loss may be noise on fragile tasks, but it is the larger of the two.

**Outcome.** r=2 was carried forward on depth_22 and r=4 was not. The batched spec targets **depth_22 + `split+recent` m=32 r=2**. Its projected speedup at batch 8 is 2.50× over dense, against 1.97× for depth_22 alone, so reuse adds about 1.27×. Most of what depth_22 still spends is outside masked reuse's reach:

- decoded rows recomputed in layers 22–28 at every step (about 208–232 rows per step in total);
- a full forward every 8 steps.

## Amendment 4: batched evaluation primary (2026-11-04, after Step 3)

Approved by Michael after reading Step 3: the primary batched fidelity comparison is **depth_22 + split+recent m=32 r=2 against depth_22**. It replaces r=4 as the proposed primary for the batched study, because of r=4's larger end-to-end loss in Step 3. Backend support for r=4 and other positive integer windows is retained; support is not evidence of fidelity.

**Full-dataset comparison.** The paired per-problem change in base-test pass@1, pooled over HumanEval (164) and MBPP (378), with a paired percentile bootstrap over problems (100,000 resamples, seed 0) and its one-sided 97.5% upper confidence bound. Reported alongside:

- per-dataset results;
- other windows;
- comparisons to dense;
- EvalPlus plus-test results;
- dense batched against unbatched discordance, as the measured numerical noise reference. The E3 discordance is a planning estimate, not an assumed result.

**Batching.** Batch manifests sort by tokenized prompt length, with task ID breaking ties, then chunk by batch size. Every arm uses the same manifest. The last batch may be smaller; its actual size is reported, and it is not counted as a full-batch timing sample. Batch sizes are kept separate in inference and reporting.

**Spec corrections** (not changes to the policy):

- Newly decoded rows get one subsequent full-depth forward.
- The float32 transfer schedule is preserved, including zero- and multiple-commit steps.
- Stable sorting applies to protected-set ranking; the sampler's topk is unchanged.
- Both arms run the output head on all masked-position predecessor states.
- Persistent hidden caches keep only final states and the configured depth boundary; intermediate workspace is transient.

**Speed.** Step 3 projects 2.50× dense at batch 8 for r=2, against 1.97× for depth_22 alone. Projections are not measurements. No GPU timing or fidelity validation of the new backend was run as part of implementation, and batch-1 CPU tests do not replace the real-model A100 bf16 correctness checks.
