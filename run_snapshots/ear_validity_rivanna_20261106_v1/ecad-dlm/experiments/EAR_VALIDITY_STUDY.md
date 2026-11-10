# EAR validity study: is EAR a fair fidelity constraint for DiffuCoder caches?

Registered 2026-11-06, before any EAR value was computed. Code:
`ear_validity.py`. Outcome data for most policies already exist: the 96-task
proxy recheck, whose results were seen before this was written.

## Why

The divergence proxy failed its recheck (`PROXY_VALIDATION.md`): across 12
policies it ranks pass rates at Spearman 0.67 against a 0.90 bar, and within the
depth family at −0.18. It is dropped as a search objective.

The new question is whether a continuous metric can predict which caches keep
passing the tests, well enough to act as a **constraint** rather than an
objective: any policy meeting a threshold counts as equally faithful, and the
search optimizes speed among those. Applying that in NSGA-II (Deb's constrained
domination) is later work; this study decides only whether EAR can be the
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

All in forced-trajectory replay, over **answer commits only**: commits at a
position before dense's first stop token. Commit order is not left to right, so
this is defined by position, not by time. At the last step, which can commit
several positions, each position is judged separately.

1. **EAR.** Mean of EAR over a task's answer commits, then mean over tasks.
2. **Tail EAR.** 10th percentile of per-task EAR.
3. **Position flip rate.** Fraction of steps with an answer commit where the
   policy's confidences would commit a different position. Exact ties count as
   agreement. The rate above the existing 0.01 tie margin is also reported.

Reported, outside the verdict:

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

**Registered (verdict):** dense with the math attention kernel (noise floor),
depth_22, depth_22 + r=2, 4, 8, depth_24 + r=4, refresh_256, refresh_256 + r=2,
4, and two new policies to fill the gap between the clusters: refresh_32 and
refresh_32_depth_22. Reuse uses `split+recent`, m=32.

**Extra (sensitivity check and continuous analysis):** depth_22_r4 with m=16,
m=64 and `split`, depth_18_r4, and two more new policies, dkv_reference and
refresh_16, for a spread of outcomes between the clusters.

Dense itself is the reference, so it is not classified.

Known in advance from the recheck: depth_22_r4_m16 has a net loss of 3, one task
outside the margin, while most depth policies are at 1–2. That label rests on a
single task and the zero-false-safe rule below is unforgiving of it, which is why
the verdict is reported on both panels. Choosing the registered panel after
seeing outcomes is a real weakness; it is the panel the study plan named.

## Decision (fixed before computing EAR)

- **Within margin:** net loss of at most 2 of 96 tasks against dense, on base
  tests.
- **Leave-one-policy-out.** For each policy, the threshold is fitted on the
  others: the largest EAR that keeps every within-margin policy safe, which is
  their lowest EAR. The held-out policy is safe if its EAR reaches it.
- **EAR is valid** if no outside-margin policy is classified safe, and at most one
  within-margin policy is wrongly rejected.
- **Position flips.** If EAR alone fails, the rule becomes EAR ≥ τ and position
  flips ≤ τ_pos, with τ_pos the highest flip rate among within-margin policies,
  also fitted leave-one-out. If that passes, the verdict is "valid with position
  flips". Otherwise EAR is **not valid** for this purpose, and that is reported.
- **Thresholds, if valid.** From the registered panel with the same rule: τ_EAR,
  τ_tail, and τ_pos if needed. To clear τ_EAR, a policy's one-sided 95% lower
  bound, from a bootstrap over tasks, must reach it.

## Reported, not gating

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
- With two clusters of policies, a threshold separating them is easy. The verdict
  is only informative through the policies between the clusters.
