# Batched teacher forcing check

**Status:** written 2026-11-12, not yet run.

## Question

Can the search teacher-force several tasks per forward without changing its decisions?

At batch one, a generation costs about 4.6 GPU-hours, the best of 1–3 processes sharing the GPU (`search_concurrency_bench/report.json`). Two processes add only 8% throughput while the GPU is already 95% busy, so the time goes to compute or weight reads, not idle gaps. Batching tasks reads the weights once for many tasks.

## Method

`dlm_kv_policy/batched_teacher_forcing.py` gives each policy one `BatchedDiffuCoderKVBackend` holding B tasks' caches.

Dense's trajectory is forced from a batch-one recording: the step at which each position was committed. It is not regenerated, because dense at batch B rounds differently and would break near-ties differently from the dense reference. Every policy therefore sees the token sequences it sees at batch one. Only the logits it is scored on, and dense's logits, can round differently. FLOPs are counted per task from that task's own rows and unpadded length.

**CPU tests** (`tests/test_batched_teacher_forcing.py`):

- One unpadded task equals `teacher_force_task` exactly. That covers metrics, FLOPs and rows, for every gene, including zero-commit steps and DiffuCoder's own modeling code.
- Padded tasks match each task forced alone:
  - Exact: commits, top-1 flips, counted position changes, FLOPs and rows.
  - Within 1e-5: EAR.
  - KL within 1e-3 relative.

**GPU check** (`experiments/batched_teacher_forcing_check.py`, one A100 80GB job on raiselab-paid):

| Stage | What it does |
|---|---|
| trajectories | Dense at batch one on the 96 search tasks. Its tokens must equal the search's dense reference (run 1's `dense/`). |
| control | The search's own batch-one `evaluate` on tasks 0 and 48 of each panel generation. Do the saved records still reproduce exactly under the current code? |
| force | Batched teacher forcing at B=16 of the panel: run 1 generation 7 and run 2 generation 1 (v3's starting population). That is 32 policies, 26 feasible under v3's thresholds, compared with their saved batch-one records. Then two batches each at B=8 and B=32, for throughput only. |
| report | Both record sets scored under v3's settings (thresholds, FLOPs objective) with the search's own `score` and `environmental_selection`. |

Tasks are batched by prompt length to limit padding. Compiled layers stay off.

## Pass bar (fixed before the run)

1. Per policy, |ΔEAR|, |Δ lower 95% bound| and |Δ tail EAR| are all ≤ 0.001.
2. No feasibility flips, except for policies whose batch-one values lie within 0.001 of a threshold.
3. The same survivors in both generations.
4. FLOPs are identical.
5. The control reproduces the saved records exactly. If it does not, the comparison is against stale records and the first four criteria cannot be read until that is explained.

If all five hold, `evaluate` switches to batched teacher forcing at the fastest batch size that fits, with the trajectories recorded once by the `dense` stage. That change is coordinated with the agent that maintains the search.

## Results

Not run yet.
