# Proxy validation: pre-registered criteria

Written 2026-10-30, before any validation results existed. Do not change the
proxy, the outcomes, the policy set, or the thresholds after results arrive.
If they need to change, write a new dated section and run a new validation.
`validate_proxy.py` applies these rules mechanically.

## Question

Does the forced-trajectory replay proxy predict the outcome we care about,
EvalPlus base-test pass/fail? The search may use the proxy only if it does.

## Data

- Config: `configs/diffucoder7b_output_variance_greedy_64tasks.yaml`, greedy
  decoding, 64 tasks, the 15 policies listed under `dkv_policies`.
- End-to-end: `output_variance_study.py` runs dense and each policy once, with
  base tests.
- Proxy: `forced_trajectory_replay.py` replays all 15 policies on the dense
  trajectory.
- Everything comes from a single GPU-type folder.

## Definitions

- **Proxy (search objective)**: `divergence_events_per_task`. Each step scores
  the total variation distance between dense and cached decoding distributions
  at the committed position, plus 1 if the committed position changes. Steps
  are summed per task and averaged over tasks. Under greedy decoding this is the
  number of steps where the token or position decision changed.
- **Per-task proxy**: large-margin decision changes in that task. A step counts
  if it has a top-1 flip where the dense top-1/top-2 margin is at least 0.1, or
  a change of committed position.
- **Policy outcome**: pass→fail count, the number of tasks dense passes and the
  policy fails.
- **Task outcome**: for each (policy, task) pair where dense passes, whether
  the policy fails.

## Preconditions

If either fails, the verdict is INCONCLUSIVE and the cause is fixed first.

1. `dense_repeat` is byte-identical to `dense`, i.e. greedy generation is
   deterministic.
2. The replay's dense trajectory equals the end-to-end `dense` output, so both
   measurements use the same dense run.

## Criteria

- **A. Policy ranking.** Spearman correlation between the proxy and the
  pass→fail count across the 15 policies is at least **0.80**. Tied values get
  average ranks.
- **B. Task prediction.** Over all (policy, task) pairs where dense passes,
  the per-task proxy predicts failure with AUROC of at least **0.75**. B needs
  at least 10 failures; with fewer it is INCONCLUSIVE.

**PASS** if A and B both pass. **FAIL** if either is below its threshold.
Otherwise **INCONCLUSIVE**.

## Reported but not part of the verdict

- Spearman against pass→fail for KL p99, top-1 flips, and large-margin changes
  per task.
- AUROC of the per-task summed divergence.
- AUROC after excluding pairs whose replay shows only near-tie flips and no
  large-margin change, since such failures may be noise even under greedy.
- Totals of pass→fail versus fail→pass across policies (E2). If they are
  roughly equal, many greedy failures are noise rather than cache damage.

## If the proxy fails

Do not launch the replay-based search. Look at which policies or tasks
disagree before changing the proxy, and pre-register any new proxy here first.

## Result as registered (2026-10-31)

`validate_proxy.py --count-tied-position-flips` on `NVIDIA_A100-SXM4-80GB`:
both preconditions hold. A: Spearman 0.925 (pass). B: AUROC 0.754 over 152
failures in 615 pairs (pass). Verdict **PASS**. Pass→fail 152 versus
fail→pass 32, so greedy failures are mostly cache damage, not noise.

## Change (2026-10-31): ignore position flips where dense was tied

Written after the result above, and adopted by Michael. This is now the
default in `validate_proxy.py` and the replay search objective.

**Change.** A position flip counts, in both the proxy and the per-task proxy,
only if dense's own position margin is at least **0.01**. The position margin
is dense's confidence at the committed position minus its best confidence at
any other masked position. Nothing else changes: token terms, outcomes,
policies and thresholds are as above.

**Why.** Half of the 40,088 position flips in the replay had a dense position
margin of exactly 0, and three quarters were below 0.0035. There dense was
choosing between tied positions, and any rounding difference breaks the tie
the other way. Swapping the order of two near-certain positions usually
leaves the final text unchanged. These flips made up most of the proxy (45
events per policy-task pair, versus 9 after the filter), and their count
tracks how many tied positions a task has rather than whether the cache broke
it. Top-1 flips already had the same kind of margin filter (0.1).

**Caveat.** The margin was chosen after seeing the outcomes above, so its
scores on the same data are not an independent test. On that data: A = 0.943,
B = 0.816. B is between 0.805 and 0.816 for every margin from 0.005 to 0.2
(`auroc_by_position_tie_margin` in the report), so the result does not hinge
on the exact value.

**Confirmation.** Repeat the procedure above on 64 held-out tasks that share
no task with the original set: same 15 policies, same greedy settings, same
GPU type. Apply criteria A and B unchanged with the default proxy.

## Registered expanded-space re-check (2026-11-04, before new data)

Michael approved running this re-check and the cost study, with policy-ranking
Spearman **>= 0.90**. Historical thresholds/results above are unchanged.
Implementation: `proxy_recheck.py`; config `configs/diffucoder7b_proxy_recheck_96.yaml`.

The exposure audit scanned 3,307 JSON/JSONL artifacts under the ECAD results root
and found 94 prior task IDs (32 HumanEval, 62 MBPP). It extracts IDs without using
outcomes. The new manifest `data/code/proxy_recheck_96_seed20261104.jsonl` has 96
previously unobserved tasks: 48 per dataset, 16 per dataset/prompt-length tertile,
with round-robin balancing over fixed lexical prompt categories. Seed 20261104.
The adjacent selection/exposure JSON files record the audit, task IDs, lengths,
categories, and manifest hash. There are 352 remaining unexposed benchmark IDs
in this audit. No claim is made about exposure outside the audited artifact root.
These 96 problems become development/validation data after this experiment.

The 12 policies, frozen in `proxy_recheck.PANEL`, are:
- refresh_256 and depth_22;
- each base with split+recent m=32 at r=2 and r=4;
- depth_22+r4 with m=16 and m=64;
- depth_22+r8 m=32;
- depth_22+r4 m=32 with split only;
- depth_18+r4 and depth_24+r4, m=32, full refresh interval 8.
All use delay 1, greedy entropy, 256 steps/tokens. Additional controls are native
dense, dense_repeat, and dense_math. There is no new entropy-trigger or weighted
ranking mechanism in this panel; they cannot inherit its validation automatically.

All replay and generation run on A100 SXM4 80GB, torch 2.4.1+cu124, with the pinned
checkpoint. Four task shards share this GPU type. Replay uses the established
batch-one backend. End-to-end cached generation uses the new batched backend at
B=1 and is checked token-identical at every step against the reference backend
on every task/policy. Batched dense B=1 is likewise checked against native dense.
Pipeline disagreement halts the shard and is a correctness failure, not evidence
about the proxy. r=1 wrappers must match both base policies in replay. Dense
repeat and replay/end-to-end dense trajectories must match exactly.

The proxy, position margin 0.01, and token margin 0.1 remain fixed. Two views must
pass, with no post-result exclusions:
1. **Overall (12 policies):** mean replay divergence versus dense-pass -> fail
   count, Spearman >=0.90. Per-task large-margin changes predict failure among
   dense-passed problems with AUROC >=0.75.
2. **Depth family (8 alternatives to depth_22):** mean incremental replay
   divergence versus depth_22-pass -> fail count, Spearman >=0.90. Incremental
   per-task large-margin changes versus depth_22 predict regressions on problems
   depth_22 passes, with AUROC >=0.75. The anchor itself is excluded from ranking.

Undefined ranks (all ties) are INCONCLUSIVE. Task AUROC requires >=10 regression
policy/problem pairs and both outcome classes, otherwise INCONCLUSIVE. Report the
number of distinct failed problems too. FAIL in either view blocks replay-only
search. PASS requires both views to pass; otherwise INCONCLUSIVE. Report 95%
percentile intervals using 2,000 problem-cluster bootstrap resamples, seed 20261104;
point thresholds above are the gates. Undefined bootstrap replicates are excluded
with counts reported, not replaced by zeros. Report gains separately from losses.
Base tests are primary; full/plus tests are secondary. Dense-math pass/fail flips
supply a numerical-noise reference, not an exemption from a failed gate.

No evolutionary run is authorized by the study launcher. A failed/inconclusive
proxy is reported before any new proxy or rule is proposed.
