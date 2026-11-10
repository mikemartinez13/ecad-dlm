# Constrained search over cache policies

Code: `evolution/constrained_kv_policy_search.py`. Config:
`configs/diffucoder7b_constrained_kv_search_96tasks.yaml`. Launcher:
`scripts/slurm/submit_constrained_search_rivanna.sh`.

## What it does

NSGA-II over 12 genes. Fidelity is a bound, not an objective: a policy is feasible
if it meets all three constraints, and feasible policies rank on one objective,
predicted batched time. Selection is Deb's constrained domination: feasible beats
infeasible, a smaller total shortfall beats a larger one, and feasible policies
compare on time.

| Constraint | Threshold | Source |
| --- | --- | --- |
| 95% lower bound of mean EAR | ≥ 0.9870 | `EAR_VALIDITY_STUDY.md` |
| Tail EAR | ≥ 0.9758 | `EAR_VALIDITY_STUDY.md` |
| Live answer coverage | ≥ 0.99 | `MASKED_ROW_REUSE_STUDY.md` |

Coverage is what guards `r` and `m`, because EAR does not see masked reuse.

**Objective.** `experiments/batched_cost_model.py`, at batch 8, from the replay's
per-step row counts. It failed its absolute calibration (it underpredicts by about
20% at batch 8) but ranked schedules at Spearman 0.99, and only the ranking is
relied on. Its time excludes request indexing and sampler overhead.

## Genes

| Gene | Values | Mechanism from |
| --- | --- | --- |
| `delay` | 1, 2 | dKV-Cache |
| `refresh_interval` | 4, 8, 16, 32 | dKV-Cache |
| `refresh_depth` | 1, 18, 22, 24 | Elastic-Cache (layer-aware refresh) |
| `terminal_refresh` | 0, 8, 16 | This project; no source found |
| `masked_window` (r) | 1, 2, 4, 8 | Fast-dLLM DualCache, d²Cache |
| `protected_size` (m) | 32, 48, 64 | d²Cache |
| `protected_rule` | split, split+recent | d²Cache; the padding split is ours |
| `prompt_refresh_interval` | 0, 8, 32 | dLLM-Cache |
| `local_refresh_radius` | 0, 2, 4 | dKV-Cache-Greedy's window, Polestar |
| `padding_window` | 0, 8, 16, 32 | DPad, Streaming-dLLM |
| `entropy_threshold` | 0, 1.5, 0.5 nats | EntropyCache |
| `entropy_scope` | local, full | EntropyCache (full); the plan (local) |

Depths are the four the cost table can price. "Never" for the prompt interval and
padding windows above 32 are left out until audited. The space is about 498,000
genomes before removing ones that have no effect (`canonical()`). Seeds are the
known controls, including depth_22 with r = 2, 4, 8 and depth_24 with r = 4.

Only EntropyCache was read in full for this. The other sources in the table come
from the study plan or from search summaries, and should be read before citing.

## How the new genes work

- **`prompt_refresh_interval`.** Prompt rows get their own schedule: a periodic full
  refresh skips the prompt unless the step is also a prompt step, and 0 means the
  prompt refreshes with every full refresh. Before this change the setting could
  only add refreshes. Existing named policies behave the same, because their
  intervals divide the refresh interval.
- **`padding_window`.** Masked rows whose stale top prediction is a stop token
  refresh on this window instead of `masked_window`, and they also skip the base's
  periodic full refreshes; otherwise an 8-step base refresh would cancel any window
  of 8 or more. Protected rows and the rows that predict them stay fresh. A row
  predicted to be padding that turns out to be code shows up as lost coverage.
- **`entropy_threshold`, `entropy_scope`.** With one commit per step, EntropyCache's
  E is the entropy of that commit (nats, temperature 1). Above the threshold, the
  next step refreshes rows. `full` recomputes the whole request, as the paper does;
  `local` refreshes committed rows within 4 positions of the commit, as the study
  plan described. Level 1.5 is the paper's τ and fires about 0.45 times per task
  here, because commits are the most confident position. Level 0.5 fires about 6
  times per task. The paper's other part, recomputing up to k = 64 recent tokens, is
  not a gene; `delay` is the nearest control.

## Running

Each generation is `advance` (CPU), then 16 replay shards (A100 80GB). A last
`advance` and an `audit` follow.

**Audit.** EAR was validated only on depth, refresh and masked-reuse policies. The
audit takes the fastest feasible policies that use any other gene, plus the fastest
overall and the fastest seed, on the audit set, and measures free-running pass@1
against dense and real batch-8 time.

## Known limits

- `refresh_32_depth_22`-style policies sit near the boundary, and no policy lies
  between EAR 0.9821 and 0.9870, so the threshold is only known to that gap.
- Batched rolling refresh is planned for a second run and is not a gene yet.

## Task sets and decision rules (fixed 2026-11-07, before any of these sets ran)

The 352 tasks no study had seen are split by `task_sets.py` into three disjoint sets
of 96, in `task_sets/`: 28 HumanEval and 68 MBPP each, balanced across length
tertiles. All 84 unseen HumanEval tasks are used, so the mix is 29% HumanEval. The 96
tasks that fit τ were 50% HumanEval, so EAR's level could shift, which is part of what
the confirmation tests. The 96 fit tasks are used for nothing below.

| Set | Used by | Purpose |
| --- | --- | --- |
| `tau_confirmation` | `tau_confirmation.py` | Confirm τ and the cost ranking, before the search |
| `search` | The search | Fitness only |
| `audit` | `audit` | Final pass@1 and measured speed; touched once, after the search |

### 1. τ confirmation

τ_EAR = 0.9870, τ_tail = 0.9758 and coverage ≥ 0.99 are applied unchanged, using the
search's own rule, including the bootstrap lower bound. Nothing is refit. On the
registered panel of `EAR_VALIDITY_STUDY.md` (the 11 policies), against dense at a net
loss of at most 2 of 96:

- **CONFIRMED:** no outside-margin policy is feasible, and at most one within-margin
  policy is infeasible.
- **UNSAFE:** an outside-margin policy is feasible. Do not launch.
- **TOO_STRICT:** two or more within-margin policies are infeasible. Do not launch.

The other old policies and five single-gene policies (`gene_*`, depth_22 plus one new
gene) are reported, not part of the verdict. If the verdict fails, τ may only be
changed in a new dated section, and it would then need a fresh set.

### 2. Cost model ranking

Predicted batch-8 speedup against real measured batched decoding, for all panel
policies including the `gene_*` ones, on the τ confirmation set, in one job on one
A100 80GB. **Passes if Spearman ≥ 0.90.** A fail means predicted time cannot be the
objective: do not launch. Absolute error is reported and not required, since the
calibration already failed on it. Measured time includes indexing and the sampler.

### 3. Launch

The search launches only if both 1 and 2 pass. The launcher checks this.

### 4. Success criterion for the search

Judged only on the audit set. The search succeeds if at least one searched policy,
not a seed, meets all three:

1. **Faithful:** free-running base-test net loss of at most 2 of 96 against dense.
2. **Fast:** measured batch-8 speedup over batched dense of at least 2×.
3. **Worth searching for:** measured speedup at least 1.10× the best seed that is
   itself faithful, the seeds being the hand-chosen policies such as depth_22 + r=8.

Pass/fail is measured in free-running batch-one generation, as in every earlier
study; batched outputs differ by rounding, and are timed here, not scored.

If no policy meets all three, the result is reported as it stands. A faithful seed
that reaches 2× with no searched policy 10% ahead means the hand-chosen policy was
already enough, and that is a finding about the search space, not a failed run.

## Result: τ confirmation and cost ranking (2026-11-07)

96 tasks of the `tau_confirmation` set on A100 80GB. Nothing was refit. The registered
verdict is **UNSAFE**, the cost ranking **passes**, and so **the search was not
launched**. Net loss is base tests against dense, whose pass@1 is 0.635 (61 of 96).
Speedup is measured at batch 8. Set `extra` is outside the registered panel; `gene` is
a single-gene policy.

| Policy | Set | EAR | Lower 95 | Tail | Net | Measured | Predicted | Feasible | Margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gene_entropy_full | gene | 0.9947 | 0.9939 | 0.9893 | 0 | 1.50× | 1.79× | yes | within |
| dense, math kernel | registered | 0.9942 | 0.9932 | 0.9881 | 3 | n/a | n/a | yes | **outside** |
| depth_18_r4 | extra | 0.9940 | 0.9930 | 0.9890 | −1 | 1.87× | 2.48× | yes | within |
| gene_local_2 | gene | 0.9939 | 0.9930 | 0.9878 | −2 | 1.61× | 1.92× | yes | within |
| gene_entropy_local | gene | 0.9925 | 0.9915 | 0.9858 | −1 | 1.58× | 1.94× | yes | within |
| depth_22 | registered | 0.9898 | 0.9884 | 0.9823 | −2 | 1.61× | 1.94× | yes | within |
| depth_22_r4_m64 | extra | 0.9897 | 0.9884 | 0.9823 | −1 | 1.89× | 2.49× | yes | within |
| depth_22_r2 | registered | 0.9897 | 0.9884 | 0.9823 | −2 | 1.83× | 2.38× | yes | within |
| depth_22_r4_split | extra | 0.9896 | 0.9883 | 0.9823 | −1 | 1.99× | 2.71× | yes | within |
| depth_22_r4 | registered | 0.9896 | 0.9883 | 0.9823 | −1 | 1.98× | 2.70× | yes | within |
| gene_padding_16 | gene | 0.9896 | 0.9883 | 0.9823 | −1 | 2.09× | 2.90× | yes | within |
| depth_22_r8 | registered | 0.9895 | 0.9881 | 0.9814 | −1 | 2.07× | 2.90× | yes | within |
| depth_22_r4_m16 | extra | 0.9892 | 0.9878 | 0.9805 | −3 | 2.06× | 2.85× | yes | within |
| depth_24_r4 | registered | 0.9873 | 0.9857 | 0.9770 | 0 | 2.04× | 2.83× | no | within |
| gene_prompt_32 | gene | 0.9872 | 0.9856 | 0.9742 | −6 | 1.71× | 2.07× | no | within |
| refresh_32_depth_22 | registered | 0.9824 | 0.9801 | 0.9650 | −2 | 1.76× | 2.15× | no | within |
| dkv_reference | extra | 0.9734 | 0.9708 | 0.9531 | 0 | 1.78× | 2.20× | no | within |
| refresh_16 | extra | 0.9670 | 0.9635 | 0.9463 | −2 | 1.92× | 2.40× | no | within |
| refresh_32 | registered | 0.9564 | 0.9524 | 0.9265 | 12 | 1.99× | 2.52× | no | outside |
| refresh_256 | registered | 0.9501 | 0.9454 | 0.9138 | 21 | 2.05× | 2.63× | no | outside |
| refresh_256_r2 | registered | 0.9500 | 0.9453 | 0.9142 | 23 | 2.39× | 3.52× | no | outside |
| refresh_256_r4 | registered | 0.9497 | 0.9449 | 0.9135 | 24 | 2.64× | 4.26× | no | outside |

**Verdicts.**
- **τ, registered panel: UNSAFE.** dense_math is feasible and outside the margin
  (4 lost, 1 gained). depth_24_r4 and refresh_32_depth_22 are also wrongly rejected,
  so the panel would be TOO_STRICT with dense_math set aside. It fails both ways.
- **Other old policies:** dkv_reference and refresh_16 are also wrongly rejected.
- **New genes (reported only):** none is lossy; gene_prompt_32 is rejected.
- **Cost ranking: passes,** Spearman 0.98 over 21 policies. Predicted speedups run
  high, by 19% to 61%.
- **Continuous relation:** Spearman of 1 − EAR is 0.38 against net loss and 0.44
  against pass-fraction drop, down from 0.84 and 0.87 on the fit set. The panels differ.

**Reading (not part of the verdict).**
- **The run looks sound.** No two identical programs got different test statuses, and
  EAR levels reproduce closely across the sets (refresh_16 0.9650 and 0.9670).
- **The margin is below the noise floor.** The kernel swap alone changes tokens on 45
  of 96 tasks and lands at net 3, which a paired test does not separate from zero
  (p = 0.38 here; 5 lost, 2 gained, p = 0.45 over both sets). Net loss of at most 2 of
  96 cannot label policies this close to dense. dense_math is also not a policy the
  search can reach, so this failure says more about the label than about EAR.
- **EAR stayed stable and outcomes moved.** refresh_32_depth_22 went from net 5 to −2,
  dkv_reference from 8 to 0, refresh_16 from 14 to −2. For refresh_16, 16 of 67 dense
  passes were lost on the fit set and 3 of 61 here (Fisher p = 0.003, uncorrected), too
  large for sampling alone. The sets differ in mix (50% and 29% HumanEval) and in
  HumanEval dense pass rate (69% and 46%). The cause has not been established.
- **Clearly lossy policies are still rejected.** Everything at net 12 or worse is
  infeasible on both sets, and nothing clearly lossy passes.
- **One allowed rejection was spent in advance.** τ_EAR is depth_24_r4's own EAR and
  feasibility uses the lower bound, so that policy could only pass with a higher EAR
  here. It reached 0.9873 and still failed, at a lower bound of 0.9857.

Pooled over both sets, 192 tasks:

| Policy | Fit set | τ set | Pooled net | Lost / gained | Paired p |
| --- | --- | --- | --- | --- | --- |
| dense, math kernel | 0 | 3 | 3 | 5 / 2 | 0.45 |
| depth_22 | 1 | −2 | −1 | 5 / 6 | 1.00 |
| depth_22_r8 | 1 | −1 | 0 | 4 / 4 | 1.00 |
| depth_24_r4 | 1 | 0 | 1 | 5 / 4 | 1.00 |
| refresh_32_depth_22 | 5 | −2 | 3 | 9 / 6 | 0.61 |
| dkv_reference | 8 | 0 | 8 | 13 / 5 | 0.10 |
| refresh_16 | 14 | −2 | 12 | 19 / 7 | 0.03 |
| refresh_32 | 23 | 12 | 35 | 39 / 4 | < 0.001 |

The pooled table was computed after the verdict. Thresholds and the launch rule are
unchanged; any change goes in a new dated section and needs a fresh set, as above.
64 unseen tasks remain outside the three sets, all MBPP.

## Launch decision (2026-11-08)

Michael chose to launch the search with τ unchanged, after it failed confirmation. τ
is therefore **not confirmed**, and the search's feasibility rule is only a working
filter. Any claim rests on the audit set. The launcher takes this as
`OVERRIDE_TAU_GATE=1`; the τ report still says `launch: false`. The audit job is
submitted on hold, so the audit set stays untouched until its net-loss margin and the
10% bar are settled.
