# Proxy validation

Written 2026-10-30, before any validation results existed. `validate_proxy.py` computes the measures below.

## Question

How well does the teacher forcing proxy track the outcome we care about, EvalPlus base-test pass/fail?

## Data

- Config: `configs/diffucoder7b_output_variance_greedy_64tasks.yaml`: greedy decoding, 64 tasks, and the 15 policies listed under `dkv_policies`.
- End-to-end: `output_variance_study.py` runs dense and each policy once, with base tests.
- Proxy: `teacher_forcing.py` teacher-forces all 15 policies on the dense trajectory.
- Everything comes from a single GPU-type folder.

## Definitions

- **Proxy (search objective):** `divergence_events_per_task`. Each step scores the total variation distance between dense and cached decoding distributions at the committed position, plus 1 if the committed position changes. Steps are summed per task and averaged over tasks. Under greedy decoding this is the number of steps where the token or position decision changed.
- **Per-task proxy:** large-margin decision changes in that task. A step counts if it has a top-1 flip where dense's top-1/top-2 margin is at least 0.1, or a change of committed position.
- **Policy outcome:** pass→fail count, the number of tasks dense passes and the policy fails.
- **Task outcome:** for each (policy, task) pair where dense passes, whether the policy fails.

## Consistency checks

1. `dense_repeat` is byte-identical to `dense`, so greedy generation is deterministic.
2. The teacher forcing's dense trajectory equals the end-to-end `dense` output, so both measurements use the same dense run.

## Measures

- **Policy ranking:** Spearman correlation between the proxy and the pass→fail count across the 15 policies. Tied values get average ranks.
- **Task prediction:** AUROC of the per-task proxy for failure, over all (policy, task) pairs where dense passes. With few failures it rests on few positives, so the failure count is reported with it.
- **Also reported:**
  - Spearman against pass→fail for KL p99, top-1 flips and large-margin changes per task.
  - The AUROC of the per-task summed divergence.
  - The AUROC after excluding pairs whose teacher forcing shows only near-tie flips.
  - Totals of pass→fail versus fail→pass across policies (E2). If they are roughly equal, many greedy failures are noise rather than cache damage.

## Result (2026-10-31)

`validate_proxy.py --count-tied-position-flips` on `NVIDIA_A100-SXM4-80GB`:

- Both consistency checks hold.
- Spearman 0.925.
- AUROC 0.754, over 152 failures in 615 pairs.
- Pass→fail 152 against fail→pass 32, so greedy failures here are mostly cache damage, not noise.

## Change (2026-10-31): ignore position flips where dense was tied

Written after the result above, and adopted by Michael. This is now the default in `validate_proxy.py` and the teacher forcing search objective.

**Change.** A position flip counts, in both the proxy and the per-task proxy, only if dense's own position margin is at least **0.01**. The position margin is dense's confidence at the committed position minus its best confidence at any other masked position. Token terms, outcomes and policies are unchanged.

**Why.**

- Half of the 40,088 position flips in the teacher forcing had a dense position margin of exactly 0, and three quarters were below 0.0035. There dense was choosing between tied positions, and any rounding difference breaks the tie the other way.
- Swapping the order of two near-certain positions usually leaves the final text unchanged.
- These flips made up most of the proxy: 45 events per policy-task pair, against 9 after the filter. Their count tracks how many tied positions a task has, not whether the cache broke it.
- Top-1 flips already had the same kind of margin filter (0.1).

**Caveat.** The margin was chosen after seeing the outcomes above, so its scores on the same data are not an independent test. On that data, Spearman is 0.943 and AUROC 0.816. The AUROC lies between 0.805 and 0.816 for every margin from 0.005 to 0.2 (`auroc_by_position_tie_margin` in the report), so the result does not hinge on the exact value.

**Held-out repeat.** The procedure above, repeated on tasks that share none with the original set, with the default proxy.

## Expanded-space re-check (2026-11-04, designed before new data)

Michael approved running this re-check and the cost study. Implementation: `proxy_recheck.py`; config: `configs/diffucoder7b_proxy_recheck_96.yaml`.

**Tasks.** The exposure audit scanned 3,307 JSON/JSONL artifacts under the ECAD results root and found 94 prior task IDs (32 HumanEval, 62 MBPP). It extracts IDs without using outcomes.

- The new manifest `data/code/proxy_recheck_96_seed20261104.jsonl` has 96 previously unobserved tasks: 48 per dataset, 16 per dataset and prompt-length tertile, round-robin balanced over fixed lexical prompt categories. Seed 20261104.
- The adjacent selection and exposure JSON files record the audit, task IDs, lengths, categories and manifest hash.
- 352 benchmark IDs remain unexposed in this audit. No claim is made about exposure outside the audited artifact root.
- These 96 problems become development data after this experiment.

**Policies.** The 12 policies, fixed in `proxy_recheck.PANEL`, are:

- refresh_256 and depth_22;
- each base with split+recent m=32 at r=2 and r=4;
- depth_22+r4 with m=16 and m=64;
- depth_22+r8 m=32;
- depth_22+r4 m=32 with split only;
- depth_18+r4 and depth_24+r4, m=32, full refresh interval 8.

All use delay 1, greedy entropy, and 256 steps and tokens. The controls are native dense, dense_repeat and dense_math. No entropy-trigger or weighted ranking mechanism is in this panel, so these results say nothing about them.

**Runs.** All teacher forcing and generation run on A100 SXM4 80GB, with torch 2.4.1+cu124 and the pinned checkpoint, in four task shards.

- Teacher forcing uses the established batch-one backend.
- End-to-end cached generation uses the new batched backend at B=1. It is checked token-identical at every step against the reference backend, on every task and policy.
- Batched dense at B=1 is likewise checked against native dense.
- A pipeline disagreement halts the shard: it is a correctness bug, not evidence about the proxy.
- r=1 wrappers must match both base policies in teacher forcing.
- Dense repeat, and the teacher-forced and end-to-end dense trajectories, must match exactly.

**Measures.** The proxy, position margin 0.01 and token margin 0.1 are as above. There are two views, with no post-result exclusions:

1. **Overall (12 policies):** Spearman between mean teacher forcing divergence and the dense-pass→fail count. AUROC of per-task large-margin changes for failure, among problems dense passes.
2. **Depth family (8 alternatives to depth_22):** Spearman between mean incremental divergence over depth_22 and the depth_22-pass→fail count. AUROC of incremental per-task large-margin changes for regressions on problems depth_22 passes. The anchor itself is excluded from the ranking.

Both views report:

- 95% percentile intervals from 2,000 problem-cluster bootstrap resamples, seed 20261104. Undefined replicates (all ties, or one outcome class) are excluded, with their count reported, not replaced by zeros.
- Regression and distinct failed-problem counts.
- Gains separately from losses.

Base tests are primary and full (plus) tests secondary. Dense-math pass/fail flips give a numerical-noise reference. The study launcher starts no evolutionary run.

**Operational note, 2026-11-05 (before any recheck result existed).**

- The recheck runs on the CS cluster rather than Rivanna, with `diffucoder7b_proxy_recheck_96_cs.yaml` and `run_proxy_cost_prerequisites_cs.sh`.
- It uses 16 task shards rather than four, so GPUs that free up at different times can share the work.
- Every shard still runs on the A100 SXM4 80GB, which `proxy_recheck.py` enforces, and the manifest hash matches `SNAPSHOT.json`.
- Code moved to `ecad-dlm/` without a behavior change. The policies, proxy and margins are unchanged.

Results of the re-check are in `EAR_VALIDITY_STUDY.md`.
