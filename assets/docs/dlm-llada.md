# LLaDA evolutionary caching

> The shared LLaDA/DiffuCoder entry points are documented in the
> [DLM ECAD guide](dlm.md). This page retains the LLaDA-specific sampling and
> benchmark provenance.

This first-pass integration applies ECAD to the attention and gated-MLP
residual branches in every LLaDA transformer layer. It loads one model replica,
patches the already-loaded transformer blocks, and keeps that replica resident
while NSGA-II evaluates all schedules.

## What is evolved

As in the original ECAD implementation, `true` means recompute a component and
`false` means reuse its most recent cached output. A DLM decision controls one
of two residual branches:

- `attention`: normalized Q/K/V projection, bidirectional attention, and output
  projection, cached after dropout and before its residual addition.
- `mlp`: normalized gated MLP and output projection, cached after dropout and
  before its residual addition.

LLaDA-8B-Instruct has 32 layers. With 32 local denoising steps, independent
raw decisions would produce `32 * 32 * 2 = 2,048` bits. The text-prompt search groups four
adjacent cacheable steps and four adjacent layers, producing a 128-bit genome.
Candidate JSON files still contain the fully expanded, human-readable schedule.

The first local step is not part of the genome and always recomputes. The same
local-step schedule repeats in every semi-autoregressive generation block, and
all residual caches are cleared at each block boundary. A missing or
shape/device/dtype-incompatible cache also forces safe recomputation.

## Objectives

The NSGA-II objectives are both minimized:

1. `quality_loss = 1 - token_agreement` against deterministic all-recompute
   answers on the evaluation prompts.
2. `compute_fraction`, an analytical dense-MAC ratio derived from LLaDA's
   attention and gated-MLP dimensions, including the always-computed vocabulary
   projection as fixed work.

Measured wall-clock time and speedup are recorded as diagnostics but are not an
evolutionary objective because single-sample GPU timings are noisy. Token
agreement is deliberately a cheap search surrogate, not a final language-model
quality benchmark. Pareto schedules should subsequently be evaluated on real
tasks before making quality claims.

The optimization uses the same NSGA-II design as ECAD: pymoo ask/tell,
multi-point crossover, bit-flip mutation, binary decision vectors, and one
minimum-difference constraint. The LLaDA evaluator is in-process to avoid
reloading 16 GB of weights for every schedule.

## Running LLaDA experiments

Use the shared residual-cache schedule-search launcher with the LLaDA configuration:

```bash
CONFIG_PATH="$PWD/configs/dlm/llada8b_residual_cache_text_agreement_search_8prompts_64tokens.yaml" \
sbatch scripts/slurm/run_dlm_ecad.sh
```

The launcher requests one GPU and supports `a100`, `a40`, and `a6000`. Supply
an allocation account or a specific GPU constraint at submission time as needed.
It reads `.env` and uses `python` from the active environment. Submit jobs from
the repository root. Command-line options after the script name are forwarded
to the Python runner.

Override the configured checkpoint explicitly when needed:

```bash
CONFIG_PATH="$PWD/configs/dlm/llada8b_residual_cache_text_agreement_search_8prompts_64tokens.yaml" \
PYTHON_BIN=/path/to/python \
MODEL_PATH=/path/to/model/snapshot \
sbatch scripts/slurm/run_dlm_ecad.sh
```

See the shared guide for artifact paths and the retained regression suites.

## Coding task datasets

The generic eight-prompt text dataset remains available for the mechanics
schedule search. For coding experiments, first materialize a deterministic EvalPlus
task dataset. The task dataset keeps the original task records for the execution-based
evaluator that will be added next, while the model receives a strict Python-only
prompt:

```bash
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python \
  /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/prepare_dlm_code_tasks.py \
  --output /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/data/code/humaneval_mbpp_64_strict_python_seed0.jsonl \
  --datasets humaneval mbpp \
  --max-tasks 64 \
  --seed 0 \
  --prompt-style strict_python
```

The selection is balanced round-robin across HumanEval+ and MBPP+ and is
reproducible for the same EvalPlus data and seed. This earlier experiment uses
the repository's strict Python-only prompt and a 1024-token generation budget.
The official-style controls below use the OpenCompass prompt format. After the
task dataset is created, submit the code token-agreement search:

```bash
CONFIG_PATH="$PWD/configs/dlm/llada8b_residual_cache_code_agreement_search_64tasks_1024tokens.yaml" \
sbatch scripts/slurm/run_dlm_ecad.sh
```

This legacy coding run uses token agreement as its evolutionary objective.
Generated programs can be scored with the post-hoc evaluator below.
Baseline and candidate artifacts retain task IDs, dataset metadata,
raw responses, normalized responses, and reconstructed executable code so that
the execution evaluator can consume them directly.

## Official-style all-recompute coding controls

Before evolving cache schedules, run separate all-recompute controls using the
official LLaDA coding settings. The HumanEval and MBPP controls use pure
diffusion and dataset-specific EOS handling. Their prompt datasets use the
same OpenCompass-style HumanEval suffix prompt and MBPP three-shot prompt as
the official harness. Materialize their fixed 32-task datasets:

```bash
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python \
  /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/prepare_dlm_code_tasks.py \
  --output /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/data/code/humaneval_32_opencompass_seed0.jsonl \
  --datasets humaneval \
  --max-tasks 32 \
  --seed 0 \
  --prompt-style opencompass

/home/gqs7zr/.conda/envs/ecad-dlm/bin/python \
  /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/prepare_dlm_code_tasks.py \
  --output /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/data/code/mbpp_32_opencompass_seed0.jsonl \
  --datasets mbpp \
  --max-tasks 32 \
  --seed 0 \
  --prompt-style opencompass
```

Submit the baseline-only controls through the shared launcher.
`MAX_GENERATIONS=0` generates the baseline without evolving candidates:

```bash
CONFIG_PATH="$PWD/configs/dlm/llada8b_humaneval_dense_baseline_32tasks_512tokens.yaml" \
MAX_GENERATIONS=0 \
sbatch --time=24:00:00 scripts/slurm/run_dlm_ecad.sh

CONFIG_PATH="$PWD/configs/dlm/llada8b_mbpp_dense_baseline_32tasks_256tokens.yaml" \
MAX_GENERATIONS=0 \
sbatch --time=24:00:00 scripts/slurm/run_dlm_ecad.sh
```

These controls are faithful to the official sampler and prompt construction,
but they are not a numerical reproduction of the published OpenCompass score:
the task datasets intentionally contain 32 evaluation tasks, and the post-hoc
evaluator uses EvalPlus base/plus tests rather than the official benchmark
evaluator.

After each job completes, evaluate its saved baseline with the existing
post-hoc Apptainer runner by setting `RUN_DIR` and `ARTIFACT_PATH` to the
corresponding absolute paths:

```bash
RUN_DIR=/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_humaneval_official \
ARTIFACT_PATH=/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_humaneval_official/baseline.json \
sbatch /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/slurm/run_saved_dlm_code_evalplus_scoring.sh

RUN_DIR=/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_mbpp_official \
ARTIFACT_PATH=/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_mbpp_official/baseline.json \
sbatch /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/slurm/run_saved_dlm_code_evalplus_scoring.sh
```

## Post-hoc EvalPlus execution

The saved coding artifacts can be evaluated without rerunning LLaDA. By
default, the evaluator treats each saved raw `response` as immutable source
data and rebuilds `executable_code` with the current dataset-specific
postprocessor before invoking EvalPlus. HumanEval uses OpenCompass's
`humaneval_postprocess_v2` first-fenced-block rule and then supports either a
canonical function-body suffix or a complete replacement definition. MBPP
uses the ordered extraction patterns from OpenCompass's `MBPPEvaluator`.
Original imports are retained when a complete HumanEval definition replaces
the prompt stub.

The evaluator reports the postprocessor version and extraction strategy, runs
a non-executing Python syntax preflight for every task, splits HumanEval and
MBPP samples, and invokes the Apptainer image once per dataset:

```bash
/home/gqs7zr/.conda/envs/ecad-dlm/bin/python \
  /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/scripts/evaluate_saved_dlm_code_generations.py \
  --run-dir /scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_code_evalplus_64_len1024 \
  --artifact /scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_code_evalplus_64_len1024/baseline.json \
  --image /scratch/gqs7zr/apptainer/evalplus.sif \
  --humaneval-data /path/to/HumanEvalPlus.jsonl \
  --mbpp-data /path/to/MbppPlus.jsonl \
  --parallel 8 \
  --output /scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_code_evalplus_64_len1024/baseline.code_eval.json
```

The same baseline evaluation can be submitted through Slurm:

```bash
RUN_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_code_evalplus_64_len1024" \
sbatch scripts/slurm/run_saved_dlm_code_evalplus_scoring.sh
```

The Slurm runner defaults to the persistent
`/scratch/gqs7zr/apptainer/evalplus.sif` image; override it with
`APPTAINER_EVALPLUS_IMAGE`. If `apptainer` is not already on `PATH`, the runner
attempts `module load apptainer`; an absolute executable can also be supplied
through `APPTAINER_BIN`.

Historical artifacts may contain `task_outputs.executable_code` produced by
an older generic Markdown-stripper. Those derived fields are intentionally not
trusted by default. Set `USE_SAVED_TASK_OUTPUTS=1` for the Slurm runner (or pass
`--use-saved-task-outputs` to the Python command) only when reproducing an old
evaluation exactly. The output JSON's `postprocessing` section records whether
code was rebuilt from raw responses, syntax-valid counts and task IDs, and the
per-task extraction/reconstruction strategy. `base_pass_at_1` and
`plus_pass_at_1` are the benchmark-level scores; the latter requires both the
original and extra EvalPlus tests to pass. Assertion-weighted fractions are
retained only as diagnostics. Each dataset result also records the EvalPlus
package version, benchmark-data hash, and deterministic subset key.

To evaluate a candidate, replace `baseline.json` and the output filename with
for example:

```text
/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_code_evalplus_64_len1024/generations/gen_000/candidates/cand_000.metrics.json
```

If the image was pulled to a persistent SIF, pass its absolute path to
`--image` instead of the `docker://` URI. The post-hoc results contain both
per-task base/plus test counts and aggregate pass fractions. The shared DLM
runner now also supports `evaluation.quality_objective:
base_regression_rate`: it runs base tests in the hardened Apptainer sandbox
during candidate scoring while reserving Plus tests for post-hoc evaluation.

## Text-prompt schedule search and resuming

The text-prompt schedule search is configured in
`/sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/configs/dlm/llada8b_residual_cache_text_agreement_search_8prompts_64tokens.yaml`: 16 candidates, 10 generations, 8
prompts, and 64 generated/denoising tokens. Edit `run.name` before beginning an
independent experiment.

The optimizer checkpoints after every completed generation. Candidate metrics
are written immediately, so a job interrupted in the middle of a generation
will reuse completed candidates when resubmitted with the same config. The
config hash must match when resuming; this prevents accidentally combining two
different experiments.

The important outputs are:

```text
/scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_ecad_pilot/
├── baseline.json
├── baseline_schedule.json
├── algorithm_checkpoint.pkl
├── run_state.json
├── pareto_front.json
├── pareto_schedules/pareto_HASH.json
└── generations/gen_XXX/candidates/
    ├── cand_YYY.json
    └── cand_YYY.metrics.json
```

To run only one additional generation, useful for budgeting:

```bash
python -m ecad.genetic.search_dlm_residual_cache_schedules \
  --config /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/configs/dlm/llada8b_residual_cache_text_agreement_search_8prompts_64tokens.yaml \
  --max-generations 1
```

## Replaying a schedule

Use any evolved candidate directly:

```bash
python -m ecad.inference.generate_with_dlm_residual_cache_schedule \
  --config /sfs/gpfs/tardis/home/gqs7zr/ai4mat/ecad/configs/dlm/llada8b_residual_cache_text_agreement_search_8prompts_64tokens.yaml \
  --schedule /scratch/gqs7zr/ai4mat/ecad/results/dlm_ecad/llada8b_ecad_pilot/generations/gen_000/candidates/cand_000.json \
  --prompt "Explain diffusion language models in plain language." \
  --output /scratch/gqs7zr/ai4mat/ecad/results/dlm_inference/example.json
```

Omitting `--schedule` runs the patched all-recompute baseline.

## Regression tests

The retained tests cover shared code task preparation and sandboxed execution,
alongside the token-level KV policy implementation. Commands are in the
[DLM ECAD guide](dlm.md).

## Current scope

This pass supports the `llama` and `sequential` blocks in the official LLaDA
remote code and requires `use_cache=False`. It does not implement
autoregressive KV caching, semantic-embedding quality scoring, lm-eval tasks,
multi-GPU candidate workers, or B200-specific optimization. Those are natural
follow-ups to the current implementation.
