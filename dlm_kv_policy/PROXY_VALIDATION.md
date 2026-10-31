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
