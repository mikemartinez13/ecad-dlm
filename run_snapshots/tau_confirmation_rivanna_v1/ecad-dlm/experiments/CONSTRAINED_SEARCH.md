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
