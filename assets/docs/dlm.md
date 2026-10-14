# Evolutionary caching for diffusion language models

The [main README](../../README.md#evolutionary-caching-for-dlms) covers DiffuCoder setup, inference, evolution, and scoring. Run commands from the repository root with your DLM Python environment active. Paths below are placeholders to replace with your own files.

## Caching implementation

The shared runner adapts ECAD's image-model search to diffusion language models. NSGA-II evolves binary decisions to recompute or reuse attention and MLP outputs for groups of layers and denoising steps. Model weights stay fixed, and one model remains loaded throughout candidate evaluation.

| File | Purpose |
| --- | --- |
| `ecad/language_models/residual_cache_model_backend.py` | Load the selected model, format prompts, and call its sampler |
| `ecad/language_models/diffucoder_residual_output_cache.py` | Install attention and MLP output caching in DiffuCoder's Dream decoder |
| `ecad/language_models/residual_output_cache_runtime.py` | Store and reuse outputs during denoising |
| `ecad/schedulers/cache_scheduler/dlm_cache_schedule.py` | Define schedules and expand grouped decisions |
| `ecad/genetic/search_dlm_residual_cache_schedules.py` | Run NSGA-II, evaluate candidates, and save checkpoints |
| `ecad/language_models/residual_cache_schedule_evaluator.py` | Generate responses and compute search objectives |
| `scripts/evaluate_saved_dlm_code_generations.py` | Score saved responses through sandboxed EvalPlus |

DiffuCoder uses the checkpoint's native `diffusion_generate` sampler and schedules caching over global denoising steps. LLaDA uses the repository's implementation of its native sampler, with a local-step schedule repeated for each output block. YAML `model.family` selects `diffucoder` or `llada`; schedules from these two domains cannot be exchanged.

Token-level K/V caching is implemented separately in `dlm_kv_policy`. Its [experiment guide](../../dlm_kv_policy/README.md) describes the dKV-Decode adaptation, refresh policies, and KV evolution commands.

## EvalPlus container and offline data

The host environment runs the model and prepares task data. An Apptainer SIF image with Python and `evalplus==0.3.1` executes generated programs using its own installed dependencies. Installing EvalPlus on the host does not install it inside the image.

You can use a shared `evalplus.sif` without rebuilding it. Set `evaluation.evalplus_image: /path/to/evalplus.sif` in the experiment YAML. If you need to build an image, create this definition on a machine that supports Apptainer building with `--fakeroot`:

```bash
cat > evalplus.def <<'EOF'
Bootstrap: docker
From: python:3.11-slim

%post
    python -m pip install --no-cache-dir evalplus==0.3.1
EOF
apptainer build --fakeroot /path/to/evalplus.sif evalplus.def
```

Download the full benchmark files from the [HumanEval+ v0.1.10 release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10) and [MBPP+ v0.2.0 release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0). These are the benchmark versions used by EvalPlus 0.3.1. The full files contain both the original and additional Plus tests; choose the full release assets rather than the Mini or NoExtreme variants. Decompress the downloads before scoring:

```bash
gzip -d /path/to/HumanEvalPlus.jsonl.gz
gzip -d /path/to/MbppPlus.jsonl.gz
```

Set `evaluation.evalplus_humaneval_path: /path/to/HumanEvalPlus.jsonl` and `evaluation.evalplus_mbpp_path: /path/to/MbppPlus.jsonl` in the YAML. The files are mounted read-only in the container, so offline jobs need no benchmark downloads. Keep the image and data files unchanged for a run. The evaluator records their hashes and checks that baseline and candidate results use matching data. Use regular JSONL files rather than symlinks.

During execution, `ecad/language_models/code_execution.py` starts Apptainer with a clean environment, filesystem and process isolation, no host home or working-directory mounts, disabled networking, and no GPU bindings. It mounts the run's execution directory for writing and the evaluator code and benchmark data read-only. EvalPlus applies per-program time limits; the batch allocation provides the job's CPU, memory, and wall-time limits. If the required isolation options fail, execution stops.

## Task datasets and experiment variants

`scripts/prepare_dlm_code_tasks.py` selects tasks, renders prompts, and writes a reusable task JSONL. Its default output is `data/code/humaneval_mbpp_64_strict_python_seed0.jsonl`, containing 32 HumanEval and 32 MBPP tasks. Relative `--output` paths are accepted. The dataset fixes task selection and prompt formatting; generation length is configured separately in YAML.

`ecad/language_models/code_tasks_and_response_processing.py` reconstructs executable Python from responses. New runs save selected task records as `code_tasks.json`, so later scoring uses the exact tasks from that run. Historical `calibration_manifest.json` files remain readable.

Copy an experiment YAML and update model, task, SIF, and benchmark paths as described in the README. Set `evaluation.baseline_artifact: null` for a fresh baseline. Use an existing baseline path only when deliberately reusing compatible generation results.

| DiffuCoder config in `configs/dlm/` | Purpose |
| --- | --- |
| `diffucoder7b_residual_cache_base_regression_search_64tasks_256tokens.yaml` | Correctness-aware coding search: 64 tasks, 256 tokens, eight candidates, 20 generations |
| `diffucoder7b_residual_cache_code_agreement_search_64tasks_256tokens.yaml` | Coding search using token agreement with the all-recompute baseline |
| `diffucoder7b_residual_cache_text_agreement_search_8prompts_32tokens.yaml` | Short text experiment: eight prompts, 32 tokens, 16 candidates, 10 generations |

Correctness-aware search minimizes base-test regression rate and estimated compute fraction. Regression rate is the fraction of tasks passed by the baseline that fail under caching. Syntax-invalid programs make a candidate infeasible. Plus tests are used only after evolution.

Token-agreement configs compare generated tokens with the baseline without executing programs, so their search objective does not require an EvalPlus image. Token agreement measures output changes, rather than program correctness. Use a new run name when switching objectives. LLaDA configurations and controls are described in the [LLaDA guide](dlm-llada.md).

## Batch jobs and result directories

`run_dlm_ecad.sh` runs the search; `run_saved_dlm_code_evalplus_scoring.sh` scores saved generations. Both scripts are in `scripts/slurm/`. They use `python` from the active environment and run in the repository where you submit the job. Pass Python command-line options after the script name:

```bash
sbatch scripts/slurm/run_dlm_ecad.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --max-generations 1
```

Omit `--max-generations` to run the configured total. Use `--max-generations 0` to prepare only the baseline and its quality objective. Existing submissions that use environment variables remain supported, but the commands above require none.

The shared default is `/scratch/<username>/ai4mat/ecad/results`, selected from the current username in `ecad/language_models/experiment_artifact_paths.py`. Residual-cache searches use `dlm_ecad/<run.name>/`; KV experiments use `dlm_kv_policy/<run.name>/`. The programs create their result directories automatically. Add `run.output_dir: /path/to/results` to override the search's output parent, or pass `--output-dir /path/to/results`. Keep the same location when resuming and scoring. The single-sample inference program accepts `--output /path/to/sample.json`.

Slurm logs default to `/scratch/<username>/slurm-<job-name>-<job-id>.out`. Use `sbatch --output=/path/to/job.log` before the script name to change that path. The batch launchers also create Apptainer cache and temporary directories under the user's scratch area. No repository-local `results` or `scratch` directories or links are needed.

The search saves proposed schedules before evaluation, generated responses before base tests, and metrics after scoring succeeds. With `run.resume: true`, resubmission reuses completed work and continues an interrupted generation. Keep the configuration and compatible GPU type unchanged when resuming.

## Saved schedules and scoring

A correctness-aware run saves the following files:

```text
<run.output_dir>/<run.name>/
├── resolved_config.yaml
├── code_tasks.json
├── baseline.json
├── baseline.base_eval.json
├── algorithm_checkpoint.pkl
├── run_state.json
├── pareto_front.json
├── pareto_schedules/pareto_HASH.json
└── generations/gen_000/candidates/
    ├── cand_000.json
    ├── cand_000.generation.json
    └── cand_000.metrics.json
```

`pareto_front.json` lists schedules and their quality and compute objectives. Compute fraction is an analytical estimate; elapsed generation time is also recorded and should be checked when comparing practical speed.

The scorer can read run and evaluation paths from the experiment YAML. It interprets a relative artifact path inside the selected run. Within an existing allocation, the candidate-scoring command is:

```bash
python scripts/evaluate_saved_dlm_code_generations.py \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --artifact generations/gen_000/candidates/cand_000.metrics.json
```

Omit `--artifact` to score `baseline.json`. To evaluate without an experiment config, supply explicit paths:

```bash
python scripts/evaluate_saved_dlm_code_generations.py \
  --run-dir /path/to/diffucoder_code_search \
  --artifact /path/to/diffucoder_code_search/baseline.json \
  --image /path/to/evalplus.sif \
  --humaneval-data /path/to/HumanEvalPlus.jsonl \
  --mbpp-data /path/to/MbppPlus.jsonl
```

The scorer rebuilds Python from saved raw responses using the current postprocessor, then runs the original and Plus tests without loading the model. The `.code_eval.json` output contains per-task outcomes, subset-level pass rates, postprocessing details, and sandbox metadata. The scoring program uses CPUs; the supplied batch launcher requests a GPU allocation, although generated programs cannot access it.

## Local checks

These regression suites run without loading model weights:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m unittest discover -s dlm_kv_policy/tests -p 'test_*.py'
```
