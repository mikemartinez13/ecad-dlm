# Dream-Instruct reasoning pilot

Michael, the active scope is now GSM8K only. MATH search and MATH audit work are
abandoned as of 2026-11-17; the historical results and frozen snapshots remain for
provenance. MATH configurations are disabled, and submission/resume launchers
reject MATH splits, including those in old snapshots.

**Current execution mode (2026-11-17 UTC):** the old GSM8K and MATH jobs
`23061509` and `23061517` were canceled before starting. The separate DiffuCoder
v4 job is unchanged. No full Dream search is being submitted with this update.
The next allocation is one bounded GSM8K teacher-sharing comparison on `raiselab`:
one A100 80GB, four CPUs, 48 GiB RAM, four-hour limit, no automatic successor.
Expected runtime is approximately 2–3 GPU-hours, pending measurement.
Initially submitted as job **23063802**, snapshot `run_snapshots/dream_gsm8k_sharing_check_v1`.
At Michael's request on 2026-11-17, the pending regular-account job was canceled
and the unchanged snapshot resubmitted as **23077727** on **raiselab-paid**, QoS
`paid`. Slurm denied the in-place account update. Resources remain one A100 80GB,
four CPUs, 48 GiB RAM and a four-hour limit; no automatic successors. The live paid
balance before submission was 1,976 SUs; estimated maximum charge is about 218 SUs.
`paid_submission.json` in the snapshot records the replacement command and job ID.
The 11 focused CPU tests passed before submission, including exact shared/serial
metrics on a small native Dream model, partial checkpoint recovery, and batch-16
teacher execution with batch-8 FLOPs accounting. GPU results remain pending.

### Single-GPU teacher sharing

`Experiment.proxy_many` passes multiple independent policy backends to the existing
vectorized evaluator. Each step shares the dense forward and teacher-side scoring.
Both search and calibration use this path. The model weights remain shared, and
only policy caches multiply. `teacher_policy_group_size` defaults to 16; setting
it to a smaller number limits concurrent cache memory without changing batch size.

`batch_size: 8` continues to define reference trajectories, inference, checkpoint
groups, and padded-FLOPs accounting. `teacher_forcing_batch_size` defaults to eight
and may combine contiguous canonical batches (e.g. 16). Scoring uses the original
recorded tokens/commit positions; FLOPs are regrouped into the original eights.
Larger teacher batches and grouped attention can change rounding and policy row
decisions, so their metric changes are measured rather than assumed absent.
Completed batch/policy checkpoints survive interruption. Progress reports include
completed/total proxy batches and the currently shared policies.

`experiments.dream_teacher_sharing_check` compares 16 policies on 16 previously
exposed GSM8K search prompts (one short and one long original batch), 256 steps:
serial B8; shared B8; shared/grouped B8; shared B16; shared/grouped B16.
All variants replay identical recorded trajectories. Full-vocabulary EAR,
temperature 0.1, top-p 0.9, and batch-8 FLOPs remain fixed. It includes an all-refresh
control, a small warmup per variant, one timed pass, and per-policy EAR/CI/tail,
coverage, row-count, FLOPs, and descriptive feasibility comparisons. This is not
an accuracy evaluation or a recalibration. OOM is recorded without retrying or
allocating more resources. It does not start a search or MATH evaluation.

Launch once with `bash scripts/slurm/submit_dream_sharing_check.sh`. Inspect
`/scratch/gqs7zr/dream_reasoning/results/dream_gsm8k_sharing_check_v1/report.json`,
the Slurm log, and each variant's `progress.md`/`activity.json` during execution.
Timing includes proxy preparation, scoring and checkpoint I/O, but excludes model
loading and warmup. One pass does not establish timing confidence intervals.

Runtime correction: 48 consecutive batch-checkpoint intervals from the original
GSM8K search have median 92.62 s per batch of eight (observed min/max 79.91/116.64).
At 1,024 candidates × 12 batches, this extrapolates to **316 GPU-hours** for search
alone, with 273–398 hours from the observed interval extremes. Those extremes
are not confidence bounds. The new question set and evolving policies may change
throughput. Calibration, audits, queue time, and startup are additional. A 72-hour
allocation is therefore a bounded installment, not a credible full-search ETA.
MATH proxy throughput is not yet measured; its observed dense batch was 200 s
versus GSM8K's 56–64 s, which is insufficient to claim a precise MATH search ETA.

## Native protocol and literature comparability

The upstream checkout lives at `Dream/`, pinned in `assets/docs/dream_sources.json`.
If absent, clone https://github.com/DreamLM/Dream directly at the repository root
and check out commit `31f94a60d187e3fd481fee3bbc2c732eb94a879c`.

The recipe follows `Dream/eval_instruct/eval.sh` and its task implementations:

| Setting | GSM8K | MATH / MATH-500 |
| --- | --- | --- |
| Prompt | `Q: {question}\nA:` | `Problem:\n{problem}\n\nSolution:` |
| Few-shot examples | 0 | 0 |
| Chat formatting | Checkpoint template, one user message | Same |
| Steps / canvas tokens | 256 / 256 | 512 / 512 |
| Token sampling | temperature 0.1, top-p 0.9 | Same |
| Position selection | entropy, algorithm temperature 0 | Same |
| Timestep endpoint | 0.001 | 0.001 |
| Score | strict-match and flexible-extract separately | corrected math_verify primary; boxed equivalence and original upstream scores retained |

The native sampler finishes the fixed canvas; EOS and task `until` strings trim
responses afterwards. Search trajectories include the first EOS commit in EAR
to retain termination fidelity and make empty responses well-defined.
Prompts are left-truncated to 1,024 tokens, matching the official instruct harness.
Native padded generation receives a numerical mask fix: fully masked padding
queries may attend valid keys, while every real query retains its original mask.
This prevents NaNs in older SDPA kernels. It changes no real-token attention edges.
The unmodified native loop can otherwise catch a sampling exception and silently
emit invalid argmax outputs. The checkpoint files themselves remain unchanged.

Fast-dLLM v1 uses Dream **Base**, GSM8K five-shot and MATH four-shot. Its headline
speedups also include parallel decoding and batch-one timings. The present results
must be labeled **Dream Instruct, zero-shot, batch 8, caching-only**. They do not
numerically reproduce those tables. d2Cache supports Dream Instruct and batching;
batch support by itself does not establish ragged computation. The checkpoint's
native sampler and our backend use padded tensors and fixed canvases, not true
ragged/continuous batching. Length bucketing reduces padding; padded FLOPs account
for the remaining batch maximum. No ragged kernel was introduced.

## Data, scoring, and isolation

`scripts/prepare_dream_reasoning.py` downloads GSM8K, MATH-lighteval (MATH train/test),
and MATH-500. It records pinned dataset revisions, question IDs, splits, hashes,
and the upstream Dream commit under `/scratch/gqs7zr/dream_reasoning/data`.
Data cache: `/scratch/gqs7zr/cache/huggingface/datasets`.

Calibration/search/held-out GSM8K contain 16/32/32 disjoint training examples.
MATH-500 OOD pilot contains 32 randomly selected test examples; all 500 are also
materialized. Full GSM8K test and MATH train/test manifests remain available.
MATH-500 overlaps MATH test and must not be counted as independent confirmation.
The small pilot sets cannot establish statistical losslessness at a one-point margin.

MATH scoring imports the pinned upstream file directly, without installing its
entire lm-eval fork. The dependencies are `math-verify==0.8.0` and
`antlr4-python3-runtime==4.11.1`. GSM8K mirrors upstream regex filtering, including
the distinction between strict and flexible extraction. The output preserves raw
responses, extracted answers where available, and both scores. No API judge is used.

Scoring version `reasoning_v2_latex_gold` fixes two pilot evaluation problems.
Upstream Minerva `exact_match` requires a literal final-answer phrase absent from
these zero-shot boxed responses; it remains a diagnostic, not the primary score.
Upstream `math_verify` parses bare normalized LaTeX gold as prose and can fail on
correct answers such as `\\sqrt{53}`. The primary metric now parses unnormalized
gold inside math delimiters with `LatexExtractionConfig`, and uses the verifier's
standard response extraction. Original upstream metrics and boxed equivalence are
recorded separately. This is an explicitly documented scoring correction, not an
exact reproduction of the upstream harness metric. CPU tests cover radicals,
equivalent fractions, wrong answers, missing answers, and an earlier correct value
followed by an incorrect boxed final answer.

## EAR and FLOPs

Dream explicitly uses full-vocabulary temperature-1 probability overlap:
`sum(min(p_dense, p_cached)) = 1 - TV(p_dense,p_cached)`. It is a smooth fidelity
proxy, not the acceptance probability of the temperature-0.1/top-p sampler.
The proxy also records decoding-distribution TV and commit-position disagreement.
Historical DiffuCoder callers retain their top-10 default and calibrated thresholds.

Dense trajectories are generated at batch 8 with the actual sampling settings and
replayed by batched teacher forcing. They are never replaced by batch-one references.
The calibration panel spans all-refresh, dKV-style interval 8, depth refresh,
combined caching, and aggressive caching. Its report pairs EAR with free-running
answer accuracy, paired bootstrap intervals, gains/losses, and rank correlation.
It anchors exploratory mean-lower-bound and tail EAR constraints to the conservative
dKV-style control. This is a starting search region, not an accuracy guarantee;
there is no downstream accuracy pass/fail gate. Thresholds and raw data are retained
for later sensitivity analysis. Coverage is reported, not imposed as a new constraint.

All 12 genes remain open in `dream7b_reasoning_pilot.yaml`. The optimizer reuses
the existing constrained-domination NSGA-II selection, crossover, and mutation.
The 8-candidate, 2-generation pilot minimizes **padded FLOPs** subject to the two
EAR constraints. Four diverse fixed controls seed it; the remainder are random.
Each forward, proxy, and free-running audit uses batch 8. A short final batch is
filled with repeated requests; filler outputs are excluded from scores and their
compute cost is included. This enables full 500/1,319-example evaluation later.

FLOPs include active rows, refresh depth, full key length, and the vocabulary head.
The dense reference uses the same masked-only vocabulary projection as our dense
adapter, avoiding an artificial caching gain from head pruning alone. Teacher
forcing projects all masked logits, so this cost is conservative for the greedy
packed-head path. At official stochastic settings, every masked row must be
resampled every step, so the full masked-head path is necessary. FLOPs exclude
sorting, softmax, row selection, and cache copies: measured latency can differ
substantially, especially with top-p sorting. Final audit timings include these
operations but exclude loading, tokenization, and answer scoring. Initial timings
are a single pass, not a stable serving-throughput measurement.

## Running and outputs

```bash
export PYTHONPATH="$PWD/ecad-dlm:$PWD"
export HF_HOME=/scratch/gqs7zr/cache/huggingface
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python scripts/prepare_dream_reasoning.py
RUN=dream_reasoning_v1 bash scripts/slurm/submit_dream_reasoning.sh
```

The launcher freezes runtime sources and the upstream scorer, then requests one
A100 80GB for at most 24 hours under `raiselab` by default. Historical pilot
launches used `raiselab-paid`; new work must use Michael's specified `raiselab`.
Override `ACCOUNT`, `GPU_TYPE`, `TIME_LIMIT`, `RUN`, or `OUTPUT` as needed.
The runner also accepts individual `parity`, `calibration`, `search`, and `audit`
stages. Batch artifacts are atomic and resumable; provenance mismatches are rejected.

Main outputs: `parity.json`, `calibration_panel.json`, `ear_calibration.json`,
`search/generation_*/summary.json`, `finalists.json`, `audit.json`, and per-batch
responses/proxy records. `provenance.json` records source hashes, model path,
dataset provenance, GPU, and PyTorch. GPU parity reports both trajectory agreement
and identical-state logit errors because bf16 rounding can change stochastic paths.
CPU tests compare native and all-refresh batch-8 trajectories on a tiny Dream model.

After the GPU run, `python -m experiments.dream_reasoning_report OUTPUT` writes
`report.json`. It uses GSM8K flexible numerical extraction as the primary accuracy,
retains strict phrase match, and reports EAR correlations with both. MATH uses
corrected math_verify as primary. Do not apply that interpretation to unrescored
v1 batch artifacts. This CPU report
does not change search constraints or select policies using OOD outcomes.

The initial pilot was submitted as Slurm job **23046867**, snapshot
`run_snapshots/dream_reasoning_v1`, output
`/scratch/gqs7zr/dream_reasoning/results/dream_reasoning_v1`.
Before launch, 164 CPU tests passed. The 7B A100 parity check matched native and
all-refresh generated tokens exactly for batch 8 over 32 steps; all 32 native and
dense-adapter history steps matched. Identical-state logits differed by at most
0.0625 (bf16) at the final inspected step. All-refresh teacher-forced EAR was
approximately 1 (float32 summation error about 1e-7), and padded FLOP ratio was 1.

Native and cached generation share logits alignment, prompt masks, timestep
schedule, sampling, and postprocessing. Stochastic reuse does not freeze previously
sampled tokens: all still-masked rows are sampled again each step.

## Follow-up runs (v2)

Michael, the follow-up launches begin domain searches without waiting for the
larger accuracy audit. Each uses batch 8, all 12 genes, population 12, and six
generations (up to 72 candidate evaluations), with the pilot's better-preserving
GSM8K finalist included as a warm-start candidate. The metric remains full EAR
and the speed objective remains padded FLOPs. There is no downstream accuracy
gate. Calibration runs first inside each search allocation and then proceeds
automatically; GSM8K uses numerical extraction, MATH uses corrected math_verify.

| Configuration | Calibration | Search | Frozen-policy audit |
| --- | --- | --- | --- |
| `dream7b_gsm8k_search.yaml` | 32 GSM8K train | 64 GSM8K train | 32 GSM8K train holdout + 64 previously unseen MATH-500 |
| `dream7b_math_search.yaml` | 16 MATH train | 32 MATH train | 32 MATH train holdout + 64 GSM8K test |
| `dream7b_gsm8k_confirm.yaml` | none | none | pilot finalist vs dense on all 1,319 GSM8K test questions |

`scripts/prepare_dream_followup.py` materializes `data_v2` from the pinned cached
datasets, checks development question separation, and records content hashes.
MATH training selection excludes question text appearing in either MATH test
manifest. The 32 already exposed MATH-500 items are excluded from the 64-question
follow-up transfer audit. MATH-domain search is domain-specific optimization;
only its frozen GSM8K audit is a cross-domain transfer test. Audits across runs
can overlap and must not be presented as independent evidence.

The confirmation run also benchmarks the same 64 GSM8K test requests three times,
warms each inference path, rotates dense/cached order, and fixes sampling seeds.
It checks output hashes across repeats. This measures timing variability, not
quality variability across seeds. The full accuracy audit records no dense
trajectories, eliminating the extra history run. Timing excludes tokenization,
loading, and scoring and uses CUDA synchronization. It compares the batched
cached path to the optimized batched dense adapter, not native batch-one timing.

Pilot rescoring is saved separately in
`/scratch/gqs7zr/dream_reasoning/results/dream_reasoning_v1_rescored` with hashes
of original artifacts. Corrected MATH-500 scores are dense 12/32, dKV8 14/32,
both finalists 13/32. The better-preserving GSM8K finalist loses four dense-correct
MATH answers and gains five; its net +1 is not evidence of losslessness.
Boxed-equivalence scores differ (dense 11/32 and that finalist 10/32), emphasizing
why both scorers remain visible. All 32 dense gold answers parsed successfully.

The pilot GSM8K EAR calibration already covered five policies on 16 prompts.
Mean EAR ranged from 0.9353 to 1.0000; its rank correlation with numerical-answer
accuracy was 0.4104. The stored v1 threshold file's 0.4743 correlation used strict
phrase matching; v2 corrects that diagnostic. These five points are exploratory,
and the v2 GSM8K calibration expands the independent calibration set to 32.

To launch a frozen configuration:

```bash
RUN=dream_gsm8k_search_v2 CONFIG="$PWD/ecad-dlm/experiments/configs/dream7b_gsm8k_search.yaml" \
  bash scripts/slurm/submit_dream_reasoning.sh
```

The source snapshots preserve each run's configuration, all runtime sources, and
upstream scorer. Resume checks compare source hashes rather than unrelated
working-tree changes. The confirmation configuration freezes the selected pilot
genome in `dream7b_pilot_selected.json`.

Submitted jobs (2026-11-16): `23060573` (`dream_gsm8k_confirm_v2`),
`23060578` (`dream_gsm8k_search_v2`), and `23060595` (`dream_math_search_v2`).
Each has its frozen snapshot under `run_snapshots/<run>` and outputs under
`/scratch/gqs7zr/dream_reasoning/results/<run>`. They were queued for A100-80 GPUs
at submission; inspect the scheduler for current status. The seven focused CPU
tests passed, including native stochastic batch-8 parity and the new scoring
regressions. Manual inspection of dense/winner scorer disagreements confirmed
correct unboxed final answers explain the extra math_verify successes; one cached
answer repeats its correct conclusion, which answer accuracy alone does not penalize.

## Full 96-question searches (v3)

Michael, the short v2 jobs `23060573`, `23060578`, and `23060595` were canceled
at your request. Their existing files are retained. Neither short search had
reached generation zero, so the full searches retain the completed pilot's
selected genome as a warm start and recompute fitness on their new search sets.

Both full searches use **population 16, 64 generations, 96 training questions,
batch 8**, all 12 caching genes, domain-specific EAR constraints, and padded FLOPs.
The frozen configurations are `dream7b_gsm8k_full_96.yaml` and
`dream7b_math_full_96.yaml`. There are 1,024 candidate evaluation slots per domain.
`scripts/prepare_dream_full_search.py` creates immutable `data_v3_96` manifests:
GSM8K samples across question-length strata; MATH balances subject/level cells and
interleaves question lengths within each cell. Calibration and audit questions
remain excluded from search. Calibration remains 32 GSM8K / 16 MATH prompts.

Runs and output roots:

- `dream_gsm8k_full_96_v3`: `/scratch/gqs7zr/dream_reasoning/results/dream_gsm8k_full_96_v3`
- `dream_math_full_96_v3`: `/scratch/gqs7zr/dream_reasoning/results/dream_math_full_96_v3`

The superseded launch submitted each run with `scripts/slurm/submit_dream_full_search.sh` as **66
dependent jobs**: calibration, generations 0–63, and final audit/timing. Each
generation requires its predecessors' checkpoints. Completed generations are
not reevaluated on resume. Each job loads the same frozen sources and configuration.
`jobs.tsv` records the stage, Slurm ID, and predecessor for every submission.
GSM8K calibration is job `23061118` and its final audit is `23061235`;
MATH calibration is `23061241` and its final audit is `23061308`.
Downstream jobs require a successful predecessor; a failed job leaves the chain
blocked for investigation rather than skipping missing results. A partially
submitted chain can be completed with `RESUME_SUBMISSION=1` and the same launcher
environment. Individual interrupted generation jobs can resume from atomic batch
artifacts using the frozen runner and their original `--generation` argument.

Inspect progress without a GPU or model load:

```bash
PYTHONPATH=ecad-dlm /home/gqs7zr/.conda/envs/ecad-dlm/bin/python \
  -m experiments.dream_search_progress \
  /scratch/gqs7zr/dream_reasoning/results/dream_gsm8k_full_96_v3
```

Use the corresponding MATH output path for its report; add `--json` for structured
output. This command reads the latest batch activity and all completed summaries.
Progress artifacts are:

- `activity.json`: current stage, generation, candidate, batch, UTC update, job ID,
  and failure message if the Python runner raises an exception; updated per batch.
- `progress.md` / `progress.json`: summary refreshed after candidates, calibration
  policies, and completed stages/generations; use the inspector for freshest activity.
- `generation_history.csv`: per-generation feasible count, best padded FLOP ratio,
  mean EAR, lower EAR confidence bound, tail EAR, policy name, and elapsed job time.
- `search/generation_NNN/partial.json`: scores for completed candidates in a running
  generation, alongside the complete planned `genomes.json`.
- `search/generation_NNN/summary.json`: all evaluated candidates, survivors, and
  search history; `feasible_survivors.json` contains the feasible surviving policies.
- `best_so_far.json`: leading policies after the last completed generation.
- `finalists.json`: written when generation 63 finishes; final audits depend on it.
- `calibration-<job>.out`, `gNNN-<job>.out`, `audit-<job>.out`: separate Slurm logs.

Search progress reports EAR fidelity and analytical FLOPs, not measured answer
accuracy or inference speedup. GSM8K final audits cover all 1,319 test questions
and 64 unseen MATH-500 items. MATH final audits cover the 468 previously unseen
MATH-500 items and 64 GSM8K test questions. Both include the dKV8 control and two
finalists, plus three warmed timing repeats on 64 domain-specific questions.
The 32 already exposed MATH-500 pilot items remain separately reported.

Ten focused tests passed before submission, including stochastic native batching
parity, math scoring, generation continuation without reevaluation, rejection of
missing predecessors, partial progress, and reporting when no policy is feasible.

## Literature sources

- [Official Dream-Instruct recipe](https://github.com/DreamLM/Dream/blob/31f94a60d187e3fd481fee3bbc2c732eb94a879c/eval_instruct/eval.sh)
- [Fast-dLLM v1](https://arxiv.org/html/2505.22618v1)
- [Fast-dLLM Dream evaluation](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/dream/eval.md)
- [d2Cache](https://arxiv.org/abs/2509.23094)
- [dKV-Cache](https://arxiv.org/abs/2505.15781)
