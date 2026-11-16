# Batched teacher forcing check

**Status:** run 2026-11-12 (job 22927521, A100 80GB, 4 h 42 min). Results are in `results/dlm_kv_policy/batched_teacher_forcing_check_v1/report.json`.

## Question

How far does teacher-forcing several tasks per forward move the numbers and decisions the search reads, and how much faster is it?

At batch one, a generation costs about 4.45 GPU-hours, the best of 1–3 processes sharing the GPU (`search_concurrency_bench/report.json`). Two processes add only 8% throughput while the GPU is already 95% busy, so batching tasks, which reads the weights once for many tasks, was the next thing to try.

## Method

`dlm_kv_policy/batched_teacher_forcing.py` gives each policy one `BatchedDiffuCoderKVBackend` holding B tasks' caches.

Dense's trajectory is forced from a batch-one recording: the step at which each position was committed. It is not regenerated, because dense at batch B rounds differently and would break near-ties differently from the dense reference. Every policy therefore sees the token sequences it sees at batch one. Only the logits it is scored on, and dense's logits, can round differently. FLOPs are counted per task from that task's own rows and unpadded length.

**CPU tests** (`tests/test_batched_teacher_forcing.py`):

- One unpadded task gives the same results as `teacher_force_task`, bit for bit, for every gene. That includes zero-commit steps and DiffuCoder's own modeling code.
- Padded tasks agree with each task forced alone up to rounding.

**GPU check** (`experiments/batched_teacher_forcing_check.py`), on run 1 generation 7 and run 2 generation 1 (v3's starting population): 32 policies, compared with their saved batch-one records.

| Stage | What it does |
|---|---|
| trajectories | Dense at batch one on the 96 search tasks, checked against the search's dense reference. |
| control | The search's own batch-one `evaluate` on tasks 0 and 48 of each generation. Do the saved records reproduce under the current code? |
| force | Batched teacher forcing at B=16, tasks grouped by prompt length, compiled layers off. Two batches each at B=8 and B=32, for throughput. |
| report | Both record sets scored under v3's settings with the search's own `score` and `environmental_selection`. |

## Results

**Reproducibility:** all 96 dense trajectories matched the dense reference token for token. The control reproduced the saved batch-one records exactly on all four tasks.

**EAR** (per policy, batched minus batch one; 16 policies per generation):

| Generation | Mean EAR, max \|Δ\| | Lower 95% bound, max \|Δ\| | Tail EAR, max \|Δ\| (median) | Answer coverage, max \|Δ\| | Largest per-task \|Δ\| |
|---|---|---|---|---|---|
| run 1, gen 7 | 1.7e-4 | 2.2e-4 | 1.5e-3 (4.3e-4) | 2.1e-4 | 7.5e-3 |
| run 2, gen 1 | 2.9e-4 | 3.1e-4 | 1.1e-3 (4.4e-4) | 2.1e-4 | 8.2e-3 |

The mean differences are slightly positive (+6e-5 and +1e-4). Tail EAR, the 10th percentile over 96 tasks, moves most because it rests on few tasks.

**Decisions:**

- No policy changed feasibility.
- The policy closest to a threshold (run 2: `ga_d2_r16_l22_t0_p8_w8_m32_split_pad16`, infeasible) moved from 0.0007 to 0.0005 below it.
- Survivors were the same in both generations.
- Feasible policies came out in the same cost order (Spearman 1.0).

**FLOPs:** all 32 panel policies use masked-row reuse, which chooses rows from the policy's own confidences. At batch 16 some near-tied rows are chosen differently. For example, task 0 at step 2 computes 42 rows instead of 44. The differences start at step 2 or 3, the first steps where reuse chooses rows, and they appear in unpadded tasks too, so padding is not the cause.

- Per-policy FLOPs move by at most 2.8e-4 relative, in both directions.
- Policies whose row choice ignores logits were not in this panel, so their FLOPs were not compared on the GPU. On the CPU they match exactly.

**Speed** (seconds per task, all 16 policies):

| Batch | s/task | GPU-h per generation | Peak memory |
|---|---|---|---|
| 1 (best of 1–3 processes) | 173 | 4.45 | 22 GB per process |
| 8 | 54.7 | 1.46 | 21 GB |
| 16 | 51–67 (longer prompts slower) | 1.44–1.55 | 32 GB |
| 32 | 49–51 | 1.34 | 41 GB |

Batching is about 3× faster. Going from B=8 to B=32 barely helps, so most of the remaining ~50 s per task does not shrink with batch size. The likely cause is the per-policy, per-task, per-step scoring (`step_metrics` over full-vocabulary rows), which batching does not touch. A profile would confirm it.

## Vectorized scoring (2026-11-13, job 22943787, 4 h 07 min)

Michael approved batched teacher forcing for v3 and asked whether scoring could be vectorized without changing the results. `teacher_force_batch(..., scoring="vectorized")`, now the default, computes dense's side of every metric once per step for all tasks, shared by all policies. It scores each policy on the whole batch in one pass, and reads the per-row values back once per step. On CPU it gives the same results as per-task scoring, bit for bit (`tests/test_batched_teacher_forcing.py`). Results: `scoring_report.json`.

**Same numbers.** Against the per-task-scored records of the same runs, in all 3,072 task–policy records of both generations:

- EAR, position-flip and position-change rates, FLOPs, rows and answer coverage are identical.
- KL and masked EAR differ by at most 5e-8, because their per-task means are now taken on the CPU.
- Against batch one, nothing moves: the same EAR differences as above, no feasibility flips, the same survivors.

**Where the time goes** (one batch of 16 short-prompt tasks, synchronized between phases, seconds per task):

| Phase | Per-task scoring | Vectorized |
|---|---|---|
| Policy forwards (16 policies) | 32.6 | 32.8 |
| Dense forward | 6.4 | 6.4 |
| Scoring | 12.4 | 3.9 |
| Read-back | | 0.9 |
| Total | 51.6 | 44.1 |

**Speed, unsynchronized:**

| Run | Per-task scoring | Vectorized |
|---|---|---|
| Run 1 gen 7, all 96 tasks, B=16 | 1.55 GPU-h | 1.37 GPU-h |
| Run 2 gen 1, all 96 tasks, B=16 | 1.44 GPU-h | 1.26 GPU-h |
| First two batches, B=8 | 54.7 s/task | 47.8 s/task (24 GB) |
| First two batches, B=32 | 50.2 s/task | 43.5 s/task (53 GB) |

Vectorizing saves about 12%. Scoring was a quarter of the time, not most of it. Two-thirds is the policies' own forwards, whose cost per task barely changes from B=8 to B=32. One possible explanation is that the many small per-layer operations of a cached forward leave the GPU waiting on launches; that is not yet profiled inside the forward.

## Reading

Michael to decide whether these differences are small enough for the search to use batched teacher forcing. Using it would take a generation from about 4.5 to about 1.4 GPU-h, and vectorizing the scoring would likely gain more. Wiring it into `evaluate` would be coordinated with the agent that maintains the search.
