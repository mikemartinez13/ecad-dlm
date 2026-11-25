# EAR validity study: is EAR a fair fidelity constraint for DiffuCoder caches?

Designed 2026-11-06, before any EAR value was computed. Code:
`ear_validity.py`. Outcome data for most policies already exist: the 96-task
proxy recheck, whose results were seen before this was written.

## Why

In its recheck (`PROXY_VALIDATION.md`), the divergence proxy ranked pass rates
across 12 policies at Spearman 0.67, and within the depth family at −0.18. It is
dropped as a search objective.

The new question is whether a continuous metric can predict which caches keep
passing the tests, well enough to act as a **constraint** rather than an
objective: any policy meeting a threshold counts as equally faithful, and the
search optimizes speed among those. Applying that in NSGA-II (Deb's constrained
domination) is later work; this study measures how well EAR could serve as the
constraint, and at what threshold.

## The metric

EAR is from Helcig, Kurtić and Alistarh, *Statistically-Lossless Quantization of
Large Language Models* (COLM 2026, arXiv 2605.02404), Eq. 3:

    EAR = (1/N) Σ_i Σ_{k=1..K} min(p_i(k), q_i(k)),  K = 10, top-K under p_i

It is the probability overlap between the two distributions, 1 − TV(p, q) up to
the excluded tail. Here p is dense, q is the cached policy.

Choices the paper does not fix, set here:

- **No renormalization.** p and q are full-vocabulary probabilities, and the sum
  stops at dense's top 10. The paper says "restricted to the top-K tokens" and
  calls this a tight approximation of the full overlap, which only holds without
  renormalizing.
- **Temperature 1.** The paper compares the models' output distributions.
  Decoding here is greedy; EAR measures soft agreement beneath the argmax.
- **Positions.** The paper uses calibration text. Here N is the forced-trajectory
  decision positions, the positions dense commits. Forced trajectory plays the
  role of the paper's teacher-forced comparison.

The paper's EAR ≥ 0.99 does not transfer: it was set for next-token prediction on
calibration text under sampled decoding. τ is derived here.

## Metrics

All in teacher forcing, over **answer commits only**: commits at a
position before dense's first stop token. Commit order is not left to right, so
this is defined by position, not by time. At the last step, which can commit
several positions, each position is judged separately.

1. **EAR.** Mean of EAR over a task's answer commits, then mean over tasks.
2. **Tail EAR.** 10th percentile of per-task EAR.
3. **Position flip rate.** Fraction of steps with an answer commit where the
   policy's confidences would commit a different position. Exact ties count as
   agreement. The rate above the existing 0.01 tie margin is also reported.

Also reported:

- **Masked EAR.** Per step, the mean EAR over every still-masked position inside
  dense's answer, then the mean over steps and tasks.

  Why it is added: decision positions are close to blind to masked reuse by
  construction, since protected rows get fresh logits. In the saved recheck, KL
  at commits (padding included) stays flat from refresh_256 to refresh_256 + r=4
  (0.0286, 0.0291), while KL over all masked rows moves for depth_22 at r=8
  (0.0097 to 0.0200). This metric was added after seeing that. It logs in the
  same pass.
- **KL at answer commits.** Already logged. It is the paper's calibration metric
  and the first cross-check if EAR looks odd.

## Outcomes

- **Paired pass@1 against dense:** lost, gained, net, on base tests.
- **Pass-fraction drop.** Per task, tests passed over tests run (base plus
  EvalPlus), averaged over tasks, as dense minus policy. Plus tests dominate the
  count. The recheck's saved scores have these counts, so existing policies need
  no test re-execution.

## Panel

**Main panel:** dense with the math attention kernel (noise floor),
depth_22, depth_22 + r=2, 4, 8, depth_24 + r=4, refresh_256, refresh_256 + r=2,
4, and two new policies to fill the gap between the clusters: refresh_32 and
refresh_32_depth_22. Reuse uses `split+recent`, m=32.

**Extra (second panel and continuous analysis):** depth_22_r4 with m=16,
m=64 and `split`, depth_18_r4, and two more new policies, dkv_reference and
refresh_16, for a spread of outcomes between the clusters.

Dense itself is the reference, so it is not classified.

Known in advance from the recheck: depth_22_r4_m16 has a net loss of 3, one task
outside the margin, while most depth policies are at 1–2. That label rests on a
single task, which is why the calls below are reported on both panels. Choosing the
main panel after seeing outcomes is a real weakness; it is the panel the study
plan named.

## Threshold fitting (fixed before computing EAR)

- **Within margin:** net loss of at most 2 of 96 tasks against dense, on base
  tests.
- **Leave-one-policy-out.** For each policy, the threshold is fitted on the
  others: the largest EAR that keeps every within-margin policy safe, which is
  their lowest EAR. The held-out policy is called safe if its EAR reaches it. The
  report lists false-safe calls (outside the margin, called safe) and false
  rejections (within the margin, called unsafe).
- **Position flips.** The same calls with the rule EAR ≥ τ and position flips ≤
  τ_pos, τ_pos being the highest flip rate among within-margin policies, also
  fitted leave-one-out.
- **Thresholds.** From the main panel with the same rule: τ_EAR, τ_tail and τ_pos.
  The search compares τ_EAR with a policy's one-sided 95% lower bound, from a
  bootstrap over tasks.

## The continuous question

The continuous question, which is the point of the study:

- Spearman of 1 − EAR, 1 − tail EAR and 1 − masked EAR against net loss and
  pass-fraction drop, over all policies and within the depth family.
- Per-task AUROC: does a task's own EAR predict that the policy loses it, over
  tasks dense passes, all policies pooled?
- Scatter data of 1 − EAR against net loss and against pass-fraction drop.

The paper's linear relation is between accuracy and KL, only in the near-lossless
regime; at aggressive compression it reports the relation turning non-linear and
task-dependent. Linearity is therefore judged on the near-dense side, not across
the lossy cluster.

## Caveats

- Thresholds fitted on these 96 tasks need confirming on tasks the fit never saw.
  352 tasks remain unexposed.
- Near the margin, labels hinge on one task: the attention-kernel swap alone
  flipped one task each way on these 96.
- With two clusters of policies, a threshold separating them is easy. The calls
  are only informative through the policies between the clusters.

## Result (2026-11-06)

96 tasks on A100 80GB. Dense and all 12 recheck policies reproduced the proxy
recheck exactly on every task. Sorted by EAR; `extra` rows are outside the
main panel.

| Policy | EAR | Tail EAR | Flip rate | Lost / gained | Net | Pass-fraction drop | Compute | Margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dense, math kernel | 0.9952 | 0.9918 | 0.110 | 1 / 1 | 0 | −0.005 | 1.000 | within |
| depth_18_r4 (extra) | 0.9934 | 0.9893 | 0.121 | 2 / 0 | 2 | −0.004 | 0.332 | within |
| depth_22 | 0.9895 | 0.9833 | 0.160 | 2 / 1 | 1 | −0.005 | 0.449 | within |
| depth_22_r2 | 0.9895 | 0.9833 | 0.164 | 3 / 1 | 2 | 0.006 | 0.349 | within |
| depth_22_r4 | 0.9895 | 0.9833 | 0.167 | 3 / 1 | 2 | −0.011 | 0.298 | within |
| depth_22_r4_split (extra) | 0.9895 | 0.9833 | 0.167 | 3 / 1 | 2 | −0.011 | 0.297 | within |
| depth_22_r4_m64 (extra) | 0.9895 | 0.9831 | 0.163 | 3 / 1 | 2 | 0.005 | 0.317 | within |
| depth_22_r8 | 0.9893 | 0.9833 | 0.166 | 2 / 1 | 1 | −0.011 | 0.272 | within |
| depth_22_r4_m16 (extra) | 0.9891 | 0.9798 | 0.171 | 3 / 0 | 3 | 0.004 | 0.277 | **outside** |
| depth_24_r4 | 0.9870 | 0.9758 | 0.188 | 2 / 1 | 1 | 0.006 | 0.280 | within |
| refresh_32_depth_22 | 0.9821 | 0.9624 | 0.265 | 5 / 0 | 5 | 0.050 | 0.395 | outside |
| dkv_reference (extra) | 0.9725 | 0.9488 | 0.221 | 9 / 1 | 8 | 0.063 | 0.389 | outside |
| refresh_16 (extra) | 0.9650 | 0.9326 | 0.292 | 16 / 2 | 14 | 0.132 | 0.349 | outside |
| refresh_32 | 0.9565 | 0.9199 | 0.339 | 23 / 0 | 23 | 0.219 | 0.329 | outside |
| refresh_256 | 0.9478 | 0.9077 | 0.398 | 26 / 0 | 26 | 0.272 | 0.311 | outside |
| refresh_256_r2 | 0.9478 | 0.9067 | 0.400 | 25 / 0 | 25 | 0.264 | 0.210 | outside |
| refresh_256_r4 | 0.9478 | 0.9066 | 0.400 | 30 / 0 | 30 | 0.295 | 0.159 | outside |

**Leave-one-out calls.**
- **Main panel:** no false-safe call and one false rejection (depth_24_r4).
- **All 17 policies:** depth_22_r4_m16 is false-safe, by EAR alone and with
  position flips. It is one task outside the margin, as flagged above.

**Thresholds (main panel).** τ_EAR = 0.9870, τ_tail = 0.9758. Position flips add
nothing on this panel, so the search does not use τ_pos. τ_EAR is depth_24_r4's own
EAR, so that policy's bootstrap lower bound (0.9848) falls below it by construction.
Nothing lies between 0.9821 and 0.9870, so the data do not place τ more precisely
than that gap.

**Continuous question (all 17 policies, same panel).**

| Metric | Spearman, net loss | Spearman, pass-fraction drop | Within depth family, net / drop |
| --- | --- | --- | --- |
| 1 − EAR | 0.84 | 0.87 | −0.20 / 0.41 |
| 1 − tail EAR | 0.87 | 0.84 | 0.02 / 0.21 |
| KL at answer commits | 0.83 | 0.85 | −0.28 / 0.29 |
| Position flip rate | 0.87 | 0.81 | 0.09 / 0.04 |
| Divergence events (all steps) | 0.76 | 0.72 | 0.02 / −0.34 |
| 1 − masked EAR | 0.55 | 0.56 | 0.15 / −0.24 |

Pearson with pass-fraction drop is above 0.97 for all but masked EAR (not
computed), but it is driven by the lossy end. Per-task AUROC: EAR 0.84, masked
EAR 0.74.

**Reading.**
- **The panel explains part of the gain over the recheck.** On this panel
  divergence rises from 0.67 to 0.76, so EAR's edge over it is moderate. The KL,
  flip-rate and divergence rows of that table were computed after the leave-one-out
  calls.
- **Divergence would also separate the main panel under this rule.** Its
  within-margin policies reach 2.91 and its outside-margin ones start at 4.52.
  What the recheck showed was poor ranking of near-dense policies, and no metric
  here ranks them well either (within-family Spearman −0.34 to 0.41). The move from
  objective to constraint does much of the work, not EAR alone.
- **EAR tracks committed-row caching well.** Between the clusters it falls in
  order with outcomes, and it sees `refresh_depth` (0.9934, 0.9895, 0.9870 at
  depths 18, 22, 24) and the refresh interval.
- **EAR does not see masked reuse.** depth_22 stays at 0.9895 from no reuse to
  r=4, and refresh_256 at 0.9478 while its losses move from 25 to 30. A search
  constrained only by EAR would treat `r` and `m` as free. That is also why
  depth_22_r4_m16 is false-safe.
- **Masked EAR did not fix that.** It is the weakest metric here, and the noise
  floor scores below every depth policy on it (0.849).
