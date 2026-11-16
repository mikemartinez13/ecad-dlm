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

**Objective.** From run 3, FLOPs over dense FLOPs (`evolution.cost_objective: flops`),
counted during teacher forcing from each forward's rows by `utils/compute_cost.py`. On the
batched backend without per-layer waits, `experiments/cost_model_panel.py` timed the
21 tau-panel policies at batch 8 (2026-11-11, A100 80GB): FLOPs ranked them at
Spearman 0.995, FLOPs padded to the batch maximum at 0.994, and the timed table at
0.992. FLOP speedups read about 16% above measured ones; only the ranking is relied on.

Runs 1 and 2 minimised `experiments/batched_cost_model.py`'s predicted batch-8 time
(`cost_objective: predicted_seconds`). On the old backend it underpredicted batch-8
time by about 20%, but ranked schedules at Spearman 0.99. A run that continues from summaries scored under the other objective re-costs
them, since each summary keeps both FLOPs and predicted time.

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

Each generation is `advance` (CPU), then 16 teacher forcing shards (A100 80GB). A last
`advance` and an `audit` follow.

**Audit.** EAR was fitted only on depth, refresh and masked-reuse policies. The
audit takes the fastest feasible policies that use any other gene, plus the fastest
overall and the fastest seed, on the audit set, and measures free-running pass@1
against dense and real batch-8 time.

## Known limits

- `refresh_32_depth_22`-style policies sit near the boundary, and no policy lies
  between EAR 0.9821 and 0.9870, so the threshold is only known to that gap.
- Batched rolling refresh is planned for a second run and is not a gene yet.

## Task sets and measurements (fixed 2026-11-07, before any of these sets ran)

The 352 tasks no study had seen are split by `task_sets.py` into three disjoint sets
of 96, in `task_sets/`: 28 HumanEval and 68 MBPP each, balanced across length
tertiles. All 84 unseen HumanEval tasks are used, so the mix is 29% HumanEval. The 96
tasks that fit τ were 50% HumanEval, so EAR's level could shift, which is part of what
the τ set measures. The 96 fit tasks are used for nothing below.

| Set | Used by | Purpose |
| --- | --- | --- |
| `tau_confirmation` | `tau_confirmation.py` | τ and the cost ranking again, on unseen tasks, before the search |
| `search` | The search | Fitness only |
| `audit` | `audit` | Final pass@1 and measured speed; touched once, after the search |

### 1. τ on unseen tasks

τ_EAR = 0.9870, τ_tail = 0.9758 and coverage ≥ 0.99 are applied unchanged, with the
search's own rule, including the bootstrap lower bound. Nothing is refit. The report
lists, for the main panel of `EAR_VALIDITY_STUDY.md` (the 11 policies), against dense
at a net loss of at most 2 of 96:

- feasible policies outside the margin (false safe);
- infeasible policies within it (false reject).

The other old policies and five single-gene policies (`gene_*`, depth_22 plus one new
gene) are listed the same way.

### 2. Cost model ranking

Predicted batch-8 speedup against real measured batched decoding, for all panel
policies including the `gene_*` ones, on the τ set, in one job on one A100 80GB.
The report gives the Spearman correlation and the absolute errors. Measured time
includes indexing and the sampler.

### 3. Audit

Measured only on the audit set, per audited policy:

- free-running base-test net loss against dense;
- measured batch-8 speedup over batched dense;
- that speedup over the fastest audited seed, the seeds being the hand-chosen
  policies such as depth_22 + r=8.

Test results come from free-running batch-one generation, as in every earlier study.
Batched outputs differ by rounding, and are timed here, not scored.

## Result: τ on unseen tasks and cost ranking (2026-11-07)

96 tasks of the `tau_confirmation` set on A100 80GB. Nothing was refit. Net loss is base
tests against dense, whose pass@1 is 0.635 (61 of 96). Speedup is measured at batch 8.
Set `main` is the EAR study's main panel, `extra` the rest, and `gene` a single-gene
policy.

| Policy | Set | EAR | Lower 95 | Tail | Net | Measured | Predicted | Feasible | Margin |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gene_entropy_full | gene | 0.9947 | 0.9939 | 0.9893 | 0 | 1.50× | 1.79× | yes | within |
| dense, math kernel | main | 0.9942 | 0.9932 | 0.9881 | 3 | n/a | n/a | yes | **outside** |
| depth_18_r4 | extra | 0.9940 | 0.9930 | 0.9890 | −1 | 1.87× | 2.48× | yes | within |
| gene_local_2 | gene | 0.9939 | 0.9930 | 0.9878 | −2 | 1.61× | 1.92× | yes | within |
| gene_entropy_local | gene | 0.9925 | 0.9915 | 0.9858 | −1 | 1.58× | 1.94× | yes | within |
| depth_22 | main | 0.9898 | 0.9884 | 0.9823 | −2 | 1.61× | 1.94× | yes | within |
| depth_22_r4_m64 | extra | 0.9897 | 0.9884 | 0.9823 | −1 | 1.89× | 2.49× | yes | within |
| depth_22_r2 | main | 0.9897 | 0.9884 | 0.9823 | −2 | 1.83× | 2.38× | yes | within |
| depth_22_r4_split | extra | 0.9896 | 0.9883 | 0.9823 | −1 | 1.99× | 2.71× | yes | within |
| depth_22_r4 | main | 0.9896 | 0.9883 | 0.9823 | −1 | 1.98× | 2.70× | yes | within |
| gene_padding_16 | gene | 0.9896 | 0.9883 | 0.9823 | −1 | 2.09× | 2.90× | yes | within |
| depth_22_r8 | main | 0.9895 | 0.9881 | 0.9814 | −1 | 2.07× | 2.90× | yes | within |
| depth_22_r4_m16 | extra | 0.9892 | 0.9878 | 0.9805 | −3 | 2.06× | 2.85× | yes | within |
| depth_24_r4 | main | 0.9873 | 0.9857 | 0.9770 | 0 | 2.04× | 2.83× | no | within |
| gene_prompt_32 | gene | 0.9872 | 0.9856 | 0.9742 | −6 | 1.71× | 2.07× | no | within |
| refresh_32_depth_22 | main | 0.9824 | 0.9801 | 0.9650 | −2 | 1.76× | 2.15× | no | within |
| dkv_reference | extra | 0.9734 | 0.9708 | 0.9531 | 0 | 1.78× | 2.20× | no | within |
| refresh_16 | extra | 0.9670 | 0.9635 | 0.9463 | −2 | 1.92× | 2.40× | no | within |
| refresh_32 | main | 0.9564 | 0.9524 | 0.9265 | 12 | 1.99× | 2.52× | no | outside |
| refresh_256 | main | 0.9501 | 0.9454 | 0.9138 | 21 | 2.05× | 2.63× | no | outside |
| refresh_256_r2 | main | 0.9500 | 0.9453 | 0.9142 | 23 | 2.39× | 3.52× | no | outside |
| refresh_256_r4 | main | 0.9497 | 0.9449 | 0.9135 | 24 | 2.64× | 4.26× | no | outside |

**Summary.**
- **τ, main panel.** dense_math is feasible though outside the margin (4 lost,
  1 gained). depth_24_r4 and refresh_32_depth_22 are within the margin but
  infeasible.
- **Other old policies:** dkv_reference and refresh_16 are within the margin but
  infeasible.
- **New genes:** none is outside the margin; gene_prompt_32 is infeasible.
- **Cost ranking:** Spearman 0.98 over 21 policies. Predicted speedups run high, by
  19% to 61%.
- **Continuous relation:** Spearman of 1 − EAR is 0.38 against net loss and 0.44
  against pass-fraction drop, down from 0.84 and 0.87 on the fit set. The panels differ.

**Reading.**
- **The run looks sound.** No two identical programs got different test statuses, and
  EAR levels reproduce closely across the sets (refresh_16 0.9650 and 0.9670).
- **The margin is below the noise floor.** The kernel swap alone changes tokens on 45
  of 96 tasks and lands at net 3, which a paired test does not separate from zero
  (p = 0.38 here; 5 lost, 2 gained, p = 0.45 over both sets). Net loss of at most 2 of
  96 cannot label policies this close to dense. dense_math is also not a policy the
  search can reach, so its false-safe call says more about the label than about EAR.
- **EAR stayed stable and outcomes moved.** refresh_32_depth_22 went from net 5 to −2,
  dkv_reference from 8 to 0, refresh_16 from 14 to −2. For refresh_16, 16 of 67 dense
  passes were lost on the fit set and 3 of 61 here (Fisher p = 0.003, uncorrected), too
  large for sampling alone. The sets differ in mix (50% and 29% HumanEval) and in
  HumanEval dense pass rate (69% and 46%). The cause has not been established.
- **Clearly lossy policies are still rejected.** Everything at net 12 or worse is
  infeasible on both sets, and nothing clearly lossy is feasible.
- **depth_24_r4 sits at τ by construction.** τ_EAR is depth_24_r4's own EAR and
  feasibility uses the lower bound, so that policy is feasible only with a higher EAR
  here. It reached 0.9873 and was still infeasible, at a lower bound of 0.9857.

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

The pooled table was computed after the summary above. The thresholds are unchanged.
64 unseen tasks remain outside the three sets, all MBPP.

## Launch decision (2026-11-08)

Michael chose to launch the search with τ unchanged, given the results above. On the
τ set, τ did not separate near-dense policies as it did on the fit set, so the
feasibility rule is a working filter and any claim rests on the audit set. The audit
job was submitted on hold, so the audit set stayed untouched until its reporting was
settled. (Until 2026-11-13 the launchers required `OVERRIDE_TAU_GATE=1` for this; that
gate has been removed.)

**Audit reporting (2026-11-08).** Michael dropped the yes/no success label. The audit
is reported as raw numbers per policy: lost and gained tasks against dense, pass@1 and
measured batch-8 speedup. The audit job was released from hold. (From 2026-11-13 the
audit script reports only these numbers.)

## Result: search and audit (2026-11-09)

8 generations of 16, 128 distinct policies, 104 feasible. Seed 0, one run.

**Search.** Best feasible predicted batch-8 speedup per generation: 2.88× (the
depth_22 + r=8 seed), 3.05, 3.34, 3.23, 3.08, 3.22, 3.20, 3.57. Among the 20 fastest
feasible policies: all have r = 8, the top of its range; 19 have m = 32, the bottom of
its range; all have a padding window (16 or 32); 13 have prompt interval 32; 16 have
entropy off. Two families lead: full-depth refresh every 8 steps with delay 2,
terminal refresh and prompt interval 32, and depth 22 or 24 with padding. The fastest
infeasible policy, 3.98×, is the leading depth-1 policy without delay 2, terminal
refresh or a wider local radius.

**Audit** on the 96 audit tasks, free-running batch one, base tests. Dense pass@1 0.635.

| Policy | Predicted | Measured | pass@1 | Lost / gained | Net | Tasks with changed code |
| --- | --- | --- | --- | --- | --- | --- |
| d2, interval 8, depth 1, terminal 8, prompt 32, radius 4, r=8, m=32, split+recent, padding 16 | 3.57× | 2.33× | 0.635 | 4 / 4 | 0 | 51 |
| d2, interval 8, depth 1, terminal 16, prompt 32, radius 2, r=8, m=32, split, padding 16 | 3.34× | 2.24× | 0.646 | 3 / 4 | −1 | 34 |
| depth 22, prompt 32, radius 2, r=8, m=32, split, padding 16 | 3.23× | 2.19× | 0.646 | 2 / 3 | −1 | 58 |
| depth 22, prompt 32, radius 2, r=8, m=32, split+recent, padding 16 | 3.23× | 2.17× | 0.646 | 2 / 3 | −1 | 57 |
| depth_22 + r=8 (seed) | 2.88× | 2.02× | 0.635 | 5 / 5 | 0 | 56 |

The best searched policy is 15% faster in measured time than the best seed, with no
net loss on base tests. Measured order matches predicted order; the cost model
overpredicts by 43% to 53%. EvalPlus extra tests were not run in the audit.

**EvalPlus extra tests on the audit (2026-11-09).** The same saved programs, rerun
under the full scope. Base results reproduce the audit exactly. Plus pass@1 needs the
base and extra tests to pass; pass fraction is tests passed over tests run, per task.

| Policy | Measured | Base pass@1 | Plus pass@1 | Plus lost / gained | Plus net | Pass fraction |
| --- | --- | --- | --- | --- | --- | --- |
| dense | 1.00× | 0.635 | 0.573 | | | 0.721 |
| d2, depth 1, terminal 8, radius 4 (best) | 2.33× | 0.635 | 0.552 | 5 / 3 | 2 | 0.721 |
| d2, depth 1, terminal 16, radius 2, split | 2.24× | 0.646 | 0.562 | 4 / 3 | 1 | 0.741 |
| depth 22, prompt 32, radius 2, split | 2.19× | 0.646 | 0.562 | 3 / 2 | 1 | 0.725 |
| depth 22, prompt 32, radius 2, split+recent | 2.17× | 0.646 | 0.562 | 3 / 2 | 1 | 0.725 |
| depth_22 + r=8 (seed) | 2.02× | 0.635 | 0.562 | 4 / 3 | 1 | 0.720 |

## Run 2 (submitted 2026-11-09)

Config `configs/diffucoder7b_constrained_kv_search_v2_96tasks.yaml`, output
`constrained_kv_search_v2`. Ranges are widened where run 1's 20 fastest feasible
policies sat at a bound (r = 8, m = 32, padding 16 or 32, prompt interval 32):

| Gene | Run 1 | Run 2 |
| --- | --- | --- |
| delay | 1, 2 | 1, 2, 3 |
| masked window r | 1, 2, 4, 8 | 1, 2, 4, 8, 16 |
| protected size m | 32, 48, 64 | 16, 24, 32, 48 |
| prompt interval | 0, 8, 32 | 0, 8, 32, 64, 128 |
| local radius | 0, 2, 4 | 0, 2, 4, 8 |
| padding window | 0, 8, 16, 32 | 0, 16, 32, 64, 128 |

Padding 8 is dropped because every padding window must be a multiple of every masked
window. Depths stay at the four the cost table prices. 24 generations of 16, about
3.5 million genomes. Search set, thresholds and cost table are run 1's.

- **Start.** Generation 0 is run 1's final population (`initial_population`), all 16
  inside the new ranges. It is re-scored on the same tasks, which also checks that
  scoring reproduces.
- **Risk.** The widened genes are the ones EAR does not see; only answer coverage
  guards them, and m = 16 was the one false-safe policy in the EAR study.
- **Audit.** Same audit set, 8 policies using unvalidated genes plus the fastest
  overall, the fastest seed and the fastest of generation 0 (run 1's best, as a
  control). EvalPlus extra tests follow automatically. A batched-backend check of
  generation 0 runs alongside. Reported as raw numbers, no success label.
- **Account** `raiselab`.

## Run 3 (2026-11-13 to 2026-11-14)

Config `configs/diffucoder7b_constrained_kv_search_v3_96tasks.yaml`, output
`constrained_kv_search_v3`. Run 2's genes, search set and thresholds, with FLOPs as the
objective and batched teacher forcing (16 tasks per forward,
`BATCHED_TEACHER_FORCING_CHECK.md`). Generation 0 is run 2's survivors. 24 generations
of 16 on one A100 80GB, 29.7 h (job 22962057, raiselab-paid).

**Search.** 384 distinct policies, 270 feasible. Best feasible FLOPs speedup so far,
by the generation it appeared: 4.84× (generation 0, run 2's best), 5.04× (4), 5.53×
(6), 5.82× (9), 5.91× (13), 6.10× (17), 6.47× (20), 7.01× (22). The best,
`ga_d2_r16_l1_t0_p128_n4_w8_m16_split_pad64_h0.5full`, has mean EAR 0.9905, lower
bound 0.9890, tail 0.9806 and answer coverage 0.998. All 20 fastest feasible policies
have depth 1, 17 have padding 64, 18 have entropy threshold 0.5 and 19 full scope.

**Audit.** The first audit job (22996905) stopped before generating: it read the cost
objective from the audit config, which has no evolution section. It now reads it from
the search's summaries. Resubmitted as 23014143 (3 h 26 min), with the extra tests as
23014144. 96 audit tasks, free-running batch one, measured speed at batch 8 against
dense. Dense base pass@1 0.635, plus pass@1 0.573. "Padded" is FLOPs with every request
charged its batch-8 group's largest row count (`cost_model_panel.padded_costs`), from
the search-set records.

| Policy | FLOPs speedup | Padded | Measured | Base pass@1 | Base lost / gained | Plus pass@1 | Plus lost / gained |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ga_d2_r16_l1_t0_p128_n4_w8_m16_split_pad64_h0.5full` | 7.01× | 4.55× | 4.39× | 0.625 | 5 / 4 | 0.542 | 6 / 3 |
| `ga_d2_r16_l1_t8_p64_n2_w8_m16_split_pad128_h0.5full` | 6.29× | 4.41× | 4.30× | 0.656 | 2 / 4 | 0.552 | 4 / 2 |
| `ga_d3_r32_l1_t0_p64_n4_w8_m24_split+recent_pad64_h0.5full` | 6.69× | 4.50× | 3.98× | 0.635 | 4 / 4 | 0.562 | 5 / 4 |
| `ga_d2_r32_l1_t0_p64_n4_w8_m24_split_pad64_h0.5full` | 6.70× | 4.43× | 3.95× | 0.615 | 5 / 3 | 0.542 | 5 / 2 |
| `ga_d2_r8_l1_t8_p32_n4_w8_m32_split+recent_pad16` (generation 0's best) | 4.84× | 4.43× | 3.88× | 0.635 | 4 / 4 | 0.552 | 5 / 3 |
| `ga_d2_r32_l1_t0_p128_n4_w4_m24_split_pad64_h0.5full` | 6.57× | 4.25× | 3.86× | 0.635 | 3 / 3 | 0.562 | 3 / 2 |
| `ga_d2_r32_l1_t0_p64_n4_w4_m24_split_pad64_h0.5full` | 6.47× | 4.22× | 3.85× | 0.646 | 2 / 3 | 0.562 | 3 / 2 |
| `ga_d3_r16_l1_t0_p128_n4_w8_m24_split_pad64_h0.5full` | 6.40× | 4.30× | 3.84× | 0.635 | 3 / 3 | 0.552 | 4 / 2 |
| `ga_d2_r16_l1_t0_p64_n2_w4_m24_split_pad64_h0.5full` | 6.25× | 4.13× | 3.51× | 0.625 | 5 / 4 | 0.552 | 5 / 3 |

- The best searched policy measures 13% faster than generation 0's best; FLOPs predicted
  45%. The other searched policies measure within −10% to +3% of it.
- Against measured speed, FLOPs rank the 9 at Spearman 0.52 and overpredict by 1.25–1.78×;
  padded FLOPs rank them at 0.83 and overpredict by 1.02–1.18×. Generation 0's best,
  which has no entropy trigger, has the smallest gap between FLOPs and padded FLOPs.
- Generation 0's best measured 2.33× in run 1's audit and 3.88× here: the backend got
  faster, so measured speedups compare only within one audit.
- Lost and gained tasks are of similar size for every policy, including generation 0's
  best; base net loss ranges from −2 to 2 tasks and plus net loss from 1 to 3.

## Run 4 (prepared 2026-11-15, not submitted)

Config `configs/diffucoder7b_constrained_kv_search_v4_96tasks.yaml`, output
`constrained_kv_search_v4`. Run 3's search set, thresholds and batched teacher forcing,
64 generations of 16. Generation 0 is run 3's survivors, re-scored under run 4's objective.

**Objective** (2026-11-16): padded FLOPs at batch 8 (`cost_objective: padded_flops`,
`utils/compute_cost.padded_batch_flops`). The 96 search tasks are sorted by prompt length
and grouped in eights, as the audit's timing groups its tasks. At each step every request
is charged its group's largest row count per layer and its group's longest sequence;
dense is charged its full forward at that length. At batch 1 this equals plain FLOPs
exactly (checked on 4,608 run 3 task records, depths 1 to 24). On run 3's 9 audited
policies it ranks measured batch-8 speed at Spearman 0.83 and reads 1.09–1.24× above
it; plain FLOPs gave 0.52 and 1.25–1.78×. (The audit table above used
`cost_model_panel.padded_costs`, which charges dense's head on masked rows only, so its
padded speedups read 5–9% lower; the ranking is the same.) Plain FLOPs are kept in each
summary as `flops_speedup`.

**Ranges.** Widened from run 3's comparisons of policies that differ in one gene
(median change in FLOPs speedup and in slack, the smaller margin above the two EAR
bounds, in units of 1e-3):

| Gene | Run 3 | Run 4 | Run 3 evidence |
| --- | --- | --- | --- |
| refresh interval | 4, 8, 16, 32 | adds 64 | 16→32: +0.35×, −3.0 slack (6 pairs); 7 of the 20 fastest at 32 |
| protected size m | 16, 24, 32, 48 | adds 8, 12, 20 | 16→24: −0.45×, +2.6 slack (3 pairs); the best at 16 |
| prompt interval | 0, 8, 32, 64, 128 | adds 256 (prompt computed once) | 64→128: +0.08×, −0.6 slack (4 pairs); 8 of the 20 fastest at 128 |
| local radius | 0, 2, 4, 8 | adds 6 | 2→4: −0.09×, +3.3 slack (7 pairs), the cheapest slack in run 3 |
| padding window | 0, 16, 32, 64, 128 | adds 256 | 64→128: +0.05×, no slack change (3 pairs) |
| entropy threshold | 0, 1.5, 0.5 | adds 0.25, 1.0 | 0.5→1.5: +0.31×, −5.1 slack (11 pairs); 18 of the 20 fastest at 0.5 |

Delay, depth, terminal refresh, masked window, rule and scope are unchanged. Depth 1
adds no work; deeper values recompute decoded rows from that layer up, and in run 3
cost 1.0–1.75× for 4–6 slack, the most expensive slack of any gene. About 22.7 million
genomes. Most comparisons rest on a few pairs, and the speedups are FLOPs, not time.

- **Risk.** m below 16 goes further than run 2's m = 16, the one false-safe policy in
  the EAR study, and EAR was not fitted on any of the widened genes. The audit is the
  check.
- **Objective risk.** Padded FLOPs fit the batched backend as it is: padded, batch 8.
  A ragged backend, or another batch size, would change which policies are fastest.
- **Smoke test.** `run_search_smoke_v4_rivanna.sh`: run 3's best and 15 variants using
  the added values, batched on the first 16 tasks and at batch one on tasks 0 and 1.
  The CPU equality test (`tests/test_batched_teacher_forcing.py`) includes a genome at
  these edges, scaled to its 12 steps.
- **Account** `raiselab`. About 80 GPU-hours, 3.5 days on one GPU; the job resubmits
  itself at the 3-day limit.
