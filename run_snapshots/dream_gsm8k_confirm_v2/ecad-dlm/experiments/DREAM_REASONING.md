# Dream-Instruct reasoning pilot

Michael, this experiment ports the token-level K/V schedule search to Dream only.
It searches on GSM8K training questions and tests frozen schedules on disjoint
GSM8K training questions and MATH-500. MATH is a domain transfer test and uses
512 generation steps/tokens, versus GSM8K's 256. It is not evidence for arbitrary OOD
generalization. The 32 exposed MATH-500 pilot examples must not become tuning data.

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
| Score | strict-match and flexible-extract separately | upstream Minerva exact_match and math_verify separately |

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
A100 80GB for at most 24 hours under the existing `raiselab-paid` allocation.
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
upstream Minerva exact_match as primary and retains math_verify. This CPU report
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

## Sources

- [Official Dream-Instruct recipe](https://github.com/DreamLM/Dream/blob/31f94a60d187e3fd481fee3bbc2c732eb94a879c/eval_instruct/eval.sh)
- [Fast-dLLM v1](https://arxiv.org/html/2505.22618v1)
- [Fast-dLLM Dream evaluation](https://github.com/NVlabs/Fast-dLLM/blob/main/v1/dream/eval.md)
- [d2Cache](https://arxiv.org/abs/2509.23094)
- [dKV-Cache](https://arxiv.org/abs/2505.15781)
