# DiffuCoder token-level KV policy experiments

This is a small research package inside ECAD for learning *when, where, and
for which tokens* DiffuCoder must recompute during diffusion denoising. It does
not use ECAD's existing whole-attention/MLP residual cache.

The active backend stores K/V tensors after RoPE, layer-boundary hidden states,
final normalized hidden states, and token/layer refresh ages. A policy chooses active tokens and a
one-based refresh boundary \(\ell^\star\). Current masks and newly decoded
tokens always start at layer one; stable decoded tokens may resume deeper.

The two evolutionary objectives are paired EvalPlus base-test
regression rate and `compute_fraction`: the policy's FLOPs divided by native
dense FLOPs, counted from the rows each forward computes (`compute_cost.py`).
It is deterministic and identical on every GPU type, so the search never
depends on which GPU it ran on. Wall-clock time is still recorded, and the
reported speedup should be measured on finalists on one GPU type. On an A40, wall
time fell only about 0.38x as fast as recomputed rows; about half of dense
time is fixed overhead that caching does not remove. Token agreement, first
divergence, K/V drift, syntax validity, and active-row fractions are
diagnostics only.

## Status of results before 2026-10-30: read this first

Do not cite the policy comparisons, ablation rankings, or search results
produced before 2026-10-30 (everything under `dlm_kv_policy/` on scratch
before that date). They are **unmeasured**, not tied. Three problems:

1. **Regression rate measured sampling luck.** Generation used temperature
   0.4. Reruns of the same policy are byte-identical, and EvalPlus pass/fail
   was identical across 2,560 repeated task comparisons. But any cache policy
   changes the generated code for 34–50 of 64 tasks, because one changed
   sampled token alters every later step. Across 49 policies, fail→pass flips
   (mean 3.0) were as common as pass→fail flips (mean 2.7). Only `delay_0`
   and `age_immediate` were clearly worse than dense. Cached policies
   "beating" dense (76.6% vs 68.75% pass) is a symptom of this noise.
2. **`delay` did nothing when `refresh_depth > 1`.** Delayed rows got the deep
   boundary instead of a full recompute, so `delay=1` and `delay=2` produced
   identical outputs. Delayed rows are now always fully recomputed. Only
   policies with `delay >= 2` and `refresh_depth > 1` change.
3. **The `depth_15`/`depth_22` `_v1` runs were not depth policies.** They ran
   before depth was implemented, so they are byte-identical to the dKV
   reference. They are not reruns of the `_v2` runs.

Hence "evolved vs hand-tuned", "the evolved policy has 0% regression", and the
ablation ordering are not established. The diagnostics below measure cache
quality directly.

## Diagnostics: noise floor and forced-trajectory replay

Two programs, both submitted with
`scripts/slurm/run_diffucoder_kv_diagnostics.sh` from the repository root:

| Program | Experiments | What it answers |
| --- | --- | --- |
| `output_variance_study.py` | E1, E2, E3 | How many tasks change, and in which direction, under a new seed, greedy decoding, a different attention kernel, or a cache policy |
| `forced_trajectory_replay.py` | E4 | At every step on the dense trajectory: KL(dense‖cached), top-1 token flips, committed-position flips, and the dense top-1/top-2 margin at each flip |

The configs `diffucoder7b_output_variance_sampling_t04_64tasks.yaml`
(temperature 0.4) and `diffucoder7b_output_variance_greedy_64tasks.yaml`
(greedy) drive both programs:

```bash
CONFIGS=dlm_kv_policy/configs
LAUNCH=dlm_kv_policy/scripts/slurm/run_diffucoder_kv_diagnostics.sh
POLICIES=full_refresh_every_step,dkv_reference,depth_22,delay_0

sbatch $LAUNCH output_variance_study --config $CONFIGS/diffucoder7b_output_variance_greedy_64tasks.yaml
sbatch $LAUNCH output_variance_study --config $CONFIGS/diffucoder7b_output_variance_sampling_t04_64tasks.yaml
sbatch $LAUNCH forced_trajectory_replay --config $CONFIGS/diffucoder7b_output_variance_greedy_64tasks.yaml --policies $POLICIES
sbatch $LAUNCH forced_trajectory_replay --config $CONFIGS/diffucoder7b_output_variance_sampling_t04_64tasks.yaml --policies $POLICIES
```

Both programs store results under `<run.name>/<GPU type>/`, e.g.
`diffucoder7b_output_variance_greedy_v1/NVIDIA_A40/`. Different GPU types
round bf16 differently, which at temperature 0.4 changes most outputs, so only
compare results from the same GPU-type folder. The launcher accepts A100,
A6000, or A40; pin one with `sbatch --constraint=a40 ...` when two jobs must
be compared. B200 and RTX PRO 6000 need a CUDA 12.8 PyTorch build.

Each variance study writes `variants/<name>.json` and `variance_report.json`,
and prints one row per variant with its changed tasks, P→F and F→P flips, and
pass@1. Finished variants are skipped on resubmission.

Each replay writes `forced_trajectory_replay/tasks/task_NNN.json` (per-step
rows) and `forced_trajectory_replay/summary.json`. Finished tasks are skipped
on resubmission. Add `--max-tasks 2` for a quick check first.

Reading the replay:

- `divergence_events_per_task` is the continuous proxy and the replay search
  objective. Each step scores the total variation distance between the dense
  and cached decoding distributions at the committed position, plus 1 if the
  committed position changes. Under greedy decoding it counts changed
  decisions. TV is a lower bound on how often a shared-random-number sampler
  diverges, which is fine for ranking.
- A position change only counts if dense's own position margin is at least
  `POSITION_TIE_MARGIN` (0.01). Below that, dense was breaking a tie, and the
  flip is rounding noise (`PROXY_VALIDATION.md`).
  `position_flips` counts all flips; `position_flips_counted` counts the rest.
  The summary recomputes the proxy from the saved per-step rows.
- Each policy keeps its own cache along the forced trajectory, so stale
  entries accumulate on its own refresh schedule
  (`tests/test_replay_compounding.py`). Only token-sequence divergence is
  excluded.
- `full_refresh_every_step` reuses nothing. Its KL is the cached backend's own
  rounding difference from native dense; real policies must clearly exceed it.
- `top1_flips_at_near_ties` counts flips where dense's top-1/top-2 probability
  margin was below 0.1. Flips only at near-ties are benign; flips at large
  margins mean the cache is wrong.
- `dense_output_hash` should equal the `output_hash` of the dense variant with
  the same settings, which confirms the replay followed the real trajectory.
- To check the proxy before using it as a search objective, compare each
  policy's KL p99 and flip count with its greedy pass→fail count from E2.

## Proxy validation and the replay search

The search may optimize the replay proxy only after it predicts pass/fail.
`PROXY_VALIDATION.md` fixes the criteria before results exist: Spearman ≥ 0.80
across 15 greedy policies, and per-task AUROC ≥ 0.75. The greedy config holds
the 15 policies. Once its variance study and replay are complete, run (no GPU
needed):

```bash
python -m dlm_kv_policy.validate_proxy \
  --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml \
  --gpu-folder NVIDIA_A100-SXM4-80GB [--count-tied-position-flips]
```

On A100, the proxy as first registered (every position flip counts) passed
with Spearman 0.925 and AUROC 0.754. The default, which is also the search
objective, ignores position flips where dense was tied. It scores 0.943 and
0.816 on the same data, which is not an independent test;
`PROXY_VALIDATION.md` describes the held-out confirmation. The reports are
`proxy_validation_report.json` and
`proxy_validation_report_counting_ties.json`.

Setting `evolution.quality_objective: replay_divergence` makes NSGA-II score
each generation with one shared replay instead of EvalPlus.
`diffucoder7b_dkv_replay_nsga2_search_greedy_64tasks.yaml` is ready, but do
not launch it until the validation verdict is PASS. Check finalists end to end
with pass@1 afterwards.

Evaluations now record `output_hash`. A policy whose output is byte-identical
to one already scored in the run reuses that EvalPlus result and records
`same_output_as`.

## Speed ceilings and the autoregressive reference

Decoding time is the sum over steps of one step's cost. Caching lowers the
cost of a step; parallel decoding lowers the number of steps.
`decoding_oracles.py` bounds both on one GPU type, and
`autoregressive_reference.py` measures the model DiffuCoder was adapted from.
Both use the greedy config and must run on the same GPU type:

```bash
for program in decoding_oracles autoregressive_reference; do
  sbatch --constraint=a100_80gb dlm_kv_policy/scripts/slurm/run_diffucoder_kv_diagnostics.sh \
    $program --config dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml
done
```

- **Timing curve t(n)**: wall time of one forward that recomputes n rows and
  attends to all L positions. `minimal` is only the math a step needs;
  `backend` is our cached backend with its bookkeeping. Each ceiling is a
  speedup over DiffuCoder's native dense forward.
- **Caching, speed only**: after the first step, recompute only masked rows
  and reuse everything else forever, ignoring quality. No policy in this
  family is faster. The script also runs it for real on the first 8 tasks to
  check the timing model.
- **Parallel decoding**: decode toward the greedy dense output, committing at
  each step every masked position whose argmax already equals it. The output
  is unchanged; the step count is how few steps the model's own predictions
  allow under this rule.
- **Combined, approximate**: the parallel oracle's steps at caching-oracle
  cost. Optimistic, because fewer steps make cached entries go stale faster.
- **Autoregressive estimate**: t(prompt) + N·t(1) with N the dense answer's
  non-padding tokens. DiffuCoder-7B has Qwen2.5-Coder-7B's shape, so its t(1)
  is the AR per-token cost.
- **Autoregressive reference**: Qwen2.5-Coder-7B-Instruct, greedy with a KV
  cache and no repetition penalty, on the same prompts: seconds per task and
  base pass@1.

Results go to `<run>/<GPU type>/decoding_oracles/` and
`<run>/<GPU type>/autoregressive_reference/`.

## Stale K/V error study

`stale_kv_error_study.py` asks whether the error of a reused K/V row,
dense minus cached, can be predicted from the row's layer, K or V, head, age
and row type (prompt, answer or padding) and from the cached vector itself.
This is the question behind training-free corrections for image diffusion
caches. It fits corrections from one scale up to a full affine map on three
folds of tasks and scores them on the fourth. Keys are compared before RoPE.
Nothing is corrected during decoding. The design and decision rule are
pre-registered in `STALE_KV_ERROR_STUDY.md`.

```bash
script=dlm_kv_policy/scripts/slurm/run_diffucoder_kv_diagnostics.sh
config=dlm_kv_policy/configs/diffucoder7b_output_variance_greedy_64tasks.yaml
for fold in 0 1 2 3; do
  sbatch --constraint=a100_80gb $script stale_kv_error_study collect --config $config --fold $fold
done
# after all four: fit, then evaluate per fold, then report
sbatch --constraint=a100_80gb $script stale_kv_error_study fit --config $config --gpu-folder NVIDIA_A100-SXM4-80GB
```

It needs the forced-trajectory replay of the same config on the same GPU
type, which supplies each task's final dense output. Results go to
`<run>/<GPU type>/stale_kv_error_study/`; the running sums alone take about
16 GB.

Two follow-ups, both pre-specified in `STALE_KV_ERROR_STUDY.md`:

- **Rounding check** (`stale_kv_error_study rounding --fold K --part P`, then
  `rounding-check`). It measures the rounding-only error of rows recomputed
  in partial forwards and asks whether the registered classification
  survives its worst-case contribution.
- **Correction pilot** (`stale_kv_correction_pilot.py run --fold K --part P`,
  then `report`). It adds the fitted M4 correction to stale rows during the
  replay, at strengths 0.5 and 1.0. It measures paired changes in
  divergence events.
- **Correction diagnostic** (`stale_kv_correction_diagnostic.py run --fold K
  --part P`, then `report`). It asks why the correction did not help: whether
  corrected K/V and attention outputs end up closer to dense.

All three read the A100 80GB study's data from any GPU and write their results
back to it. The rounding check runs on either size of A100 and stops unless
the GPU reproduces the A100 80GB replay exactly. The pilot and diagnostic
run on any allowed GPU type, since each comparison stays on one GPU.

## Implementation and experiment selection

| File | Responsibility |
| --- | --- |
| `run_diffucoder_kv_experiments.py` | CLI dispatch for policy evaluation, ablations, NSGA-II search, and held-out scoring |
| `diffucoder_active_token_backend.py` | Selective token/layer computation using cached K/V |
| `active_token_kv_cache.py` | Per-layer K/V and hidden-state storage with sparse row updates |
| `denoising_token_state.py` | Prompt, mask, newly decoded, and stable decoded token categories |
| `dkv_decode_refresh_policy.py` | dKV-Decode reference, ablation, and searched refresh rules |
| `fixed_token_refresh_policy.py` | Legacy general fixed-policy screen |
| `diffucoder_denoising_sampler.py` | Denoising steps and token sampling |
| `code_correctness_and_latency_evaluator.py` | Paired code correctness and generation-time measurements |
| `nsga2_kv_policy_search.py` | Categorical NSGA-II search and checkpoint resume |
| `kv_policy_experiment_config.py` | YAML experiment configuration |
| `compute_cost.py` | Deterministic FLOP count used as the search's speed objective |
| `output_variance_study.py` | E1–E3: changed tasks and flips versus a reference variant |
| `forced_trajectory_replay.py` | E4: per-step cached-versus-dense drift on the dense trajectory; the proxy |
| `validate_proxy.py` | Applies the pre-registered criteria in `PROXY_VALIDATION.md` |
| `decoding_oracles.py` | Timing curve t(n), caching and parallel-decoding ceilings, autoregressive estimate |
| `autoregressive_reference.py` | Qwen2.5-Coder-7B-Instruct time per task and pass@1 on the same tasks |
| `stale_kv_error_study.py` | Whether stale K/V error is predictable; pre-registered in `STALE_KV_ERROR_STUDY.md` |
| `stale_kv_correction_pilot.py` | Exploratory pilot: applies the fitted stale K/V correction during the replay |
| `stale_kv_correction_diagnostic.py` | Why the correction did not help: corrected K/V and attention-output error against dense |

All modes use `scripts/slurm/run_diffucoder_kv_policy_experiments.sh`:

| Environment setting | Experiment |
| --- | --- |
| No mode setting | Fixed-policy screen selected by `fixed_policy_evaluation.policies` |
| `DKV_DECODE_REFERENCE_EVALUATION=1` | Full paired dKV-Decode reference evaluation |
| `DKV_DECODE_ABLATION_EVALUATION=1` | Named ablations selected with `ABLATION_POLICY_NAMES` |
| `KV_POLICY_NSGA2_SEARCH=1` | Policy search; `EVOLUTION_POPULATIONS_PER_JOB` sets the job's population budget |
| `HELDOUT_EVALPLUS_POLICY=<name>` | Score existing generated code with expanded EvalPlus tests |

Set `REUSE_DENSE_BASELINE_DIR` to reuse matching dense outcomes. YAML filenames
state the model, cache mechanism, objective, and task or timing budget.
Historical `run.name` values remain unchanged so the renamed configs resume
existing runs. This launcher takes exported environment settings, including
`PYTHON_BIN`; it does not source `.env`.

## Running experiments

Submit GPU generation through Slurm. The main launcher supports full fixed-policy
evaluation, ablations, resumable KV evolution, and held-out EvalPlus scoring.
Saved artifacts default to `/scratch/<username>/ai4mat/ecad/results/dlm_kv_policy`.
The runner creates each run directory. Override the parent with YAML
`run.output_dir` or the `--output-dir /path/to/results` command-line option.
Slurm logs go to `/scratch/<username>/ai4mat/ecad/logs/slurm-<job-name>-<job-id>.out`;
the directory must exist before submitting. See the
[shared DLM guide](../assets/docs/dlm.md) for setup and the retained tests.

## Full paired EvalPlus-base dKV evaluation

Run the fixed dKV policy on all 64 prompts.
This command reuses the validated A40 dense generations and EvalPlus-base
result only after matching both signatures. It re-times dense execution in the
new allocation, then evaluates the dKV candidate with the same public base
tests; Plus remains held out.

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_reference_base_evaluation_64tasks_256tokens.yaml" \
RUN_NAME=diffucoder7b_dkv_sdpa_evalplus_base_a40_v1 \
DKV_DECODE_REFERENCE_EVALUATION=1 \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=06:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

It writes `dkv_decode_reference_evaluation_summary.json` and
`policies/dkv_decode_reference.json`. The summary reports the paired base-test
regression rate and measured median wall-time speedup.

## Full one-axis dKV ablations

The ablation config contains direct variants of the validated dKV rule:
`delay_0`, `delay_2`, `refresh_4`, `refresh_16`,
`refresh_16_terminal_16`, `refresh_16_depth_22`, `depth_15`, `depth_22`,
`local_4`, `prompt_4`, and `terminal_16`. A depth policy now recomputes stable
decoded rows only from its configured boundary upward on ordinary cached steps;
it is not a no-op.
Submit one policy per job so each result has its own fresh A40 dense timing
reference. For example:

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_ablation_base_evaluation_64tasks_256tokens.yaml" \
RUN_NAME=diffucoder7b_dkv_ablation_delay_0_a40_v1 \
DKV_DECODE_ABLATION_EVALUATION=1 \
ABLATION_POLICY_NAMES=delay_0 \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=06:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

Each run writes `dkv_decode_ablation_evaluation_summary.json` and `policies/<policy>.json`.
Do not use the older `dkv_style` fixed policy for these comparisons: it is not
the validated dKV-Decode schedule.

The next fixed-policy composition is available as follows:

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_ablation_base_evaluation_64tasks_256tokens.yaml" \
RUN_NAME=diffucoder7b_dkv_ablation_refresh_16_terminal_16_a40_v1 \
DKV_DECODE_ABLATION_EVALUATION=1 \
ABLATION_POLICY_NAMES=refresh_16_terminal_16 \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=06:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

Give each GPU job a distinct `RUN_NAME`. The configs use only the local
checkpoint, pinned EvalPlus datasets, and the existing no-network Apptainer
evaluator. Results live under `/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/<RUN_NAME>/`.

After selecting a policy using base tests, run its held-out EvalPlus Plus score
from that policy's run directory:

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_ablation_base_evaluation_64tasks_256tokens.yaml" \
RUN_NAME=<the_completed_run_name> \
HELDOUT_EVALPLUS_POLICY=<the_selected_policy> \
sbatch --constraint=a40 --time=02:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

## Eight-population categorical KV policy search

`diffucoder7b_dkv_decode_nsga2_search_64tasks_8populations.yaml` searches a four-coordinate dKV policy:
decoded-token delay, global refresh interval, stable-token refresh depth, and
terminal full-refresh length. The only NSGA-II objectives are paired
EvalPlus-base regression rate and the deterministic compute fraction. Token
agreement, syntax, recoveries, and active-row fractions remain diagnostics.

Each candidate still generates all 64 prompts and receives its own paired
EvalPlus Base result. To make evolution practical, the output-producing pass is
also the candidate timing measurement: it records no histories, K/V drift, or
active-row diagnostics. One dense warm-up and one timed dense pass are shared
by every population run in the same Slurm job. Final Pareto policies should be
re-evaluated separately with the three-repeat timing and diagnostics protocol
below; the search does not automatically run that comparison.

Generation 0 is the seeded population and seven evolved populations follow.
Submit all eight in one job with:

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_nsga2_search_64tasks_8populations.yaml" \
RUN_NAME=diffucoder7b_code_kv_nsga2_a40_fast_v2 \
KV_POLICY_NSGA2_SEARCH=1 \
EVOLUTION_POPULATIONS_PER_JOB=8 \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=30:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

The checkpoint is `/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/<RUN_NAME>/evolution/state.json`.
If Slurm ends early, resubmit the same command and it resumes from the first
unfinished population. Each population report is saved under
`evolution/generations/`, while the final frontier is in `evolution/summary.json`.
EvalPlus Plus is run only after Base has selected the final evolved policy and
never participates in evolution.

## Repeated-timing comparison of selected policies

The fast evolutionary measurement is a selection result, not the final timing
claim. `diffucoder7b_dkv_decode_selected_policy_comparison_64tasks_3timing_repeats.yaml` re-evaluates the evolved
`delay=2, refresh=12, depth=24` policy, its delay-one sibling, and the fixed
depth-22 control on all 64 tasks. They share one freshly measured dense
reference and each receives three candidate timing repeats, histories, K/V
diagnostics, and paired EvalPlus Base results.

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_selected_policy_comparison_64tasks_3timing_repeats.yaml" \
DKV_DECODE_ABLATION_EVALUATION=1 \
ABLATION_POLICY_NAMES=evolved_delay_2_refresh_12_depth_24,evolved_delay_1_refresh_12_depth_24,depth_22_control \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=10:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

After that job completes, run held-out EvalPlus Plus separately for the dense
reference, evolved winner, and depth-22 control. These jobs reuse generated
programs; they do not generate candidates again.

```bash
CONFIG_PATH="$PWD/dlm_kv_policy/configs/diffucoder7b_dkv_decode_selected_policy_comparison_64tasks_3timing_repeats.yaml" \
HELDOUT_EVALPLUS_POLICY=dense \
REUSE_DENSE_BASELINE_DIR="/scratch/gqs7zr/ai4mat/ecad/results/dlm_kv_policy/diffucoder7b_code_active_kv_a40_v1" \
sbatch --constraint=a40 --time=04:00:00 \
  dlm_kv_policy/scripts/slurm/run_diffucoder_kv_policy_experiments.sh
```

## Intentional v1 limits

The backend is batch-size one because each prompt has a different active-row
set. It is not a general DLM framework. The implementation supports fixed-policy studies and categorical evolutionary
search; specialized kernels remain a separate extension.

## Batched DiffuCoder backend

Michael, the batched implementation lives alongside the batch-one reference:

- `batched_diffucoder_backend.BatchedDiffuCoderKVBackend` subclasses the existing
  backend and calls its `_layer_forward`/`_attention`; it changes row orchestration,
  not transformer arithmetic. `_repeat_kv` supports torch 2.4.1+cu124.
- `batched_active_token_cache.py` stores per-request K/V and final hidden states,
  an optional starting-layer boundary, and one transient hidden workspace.
  With depth 22, only boundary 21 persists. Dummy rows never write to real slots.
- `masked_row_reuse_policy.py` is shared with replay (the old import remains valid).
  Any positive integer `window` is supported, including windows not dividing the
  base refresh interval. Base full refreshes always take precedence.
- `BatchedDenseDiffuCoderBaseline` calls the loaded native decoder core and projects
  only masked-position predecessors. The existing full-head baseline remains the
  numerical reference for rung 1.
- `DiffuCoderDenoisingSampler` supports both paths. Batched decoding currently
  requires greedy entropy, top_p=1 and top_k=null. It computes the float32 transfer
  schedule once, preserving truncation, and keeps required decode state on-device.
  Generation fails if the model emits MASK and violates the common schedule.

Use the existing loader; do not patch checkpoint files or Hugging Face caches:

```python
from dlm_kv_policy.batched_decoding_study import build_backend
from dlm_kv_policy.dkv_decode_refresh_policy import DKVDecodeRefreshPolicy
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler

base = DKVDecodeRefreshPolicy(cache_reloading_step=8, refresh_depth=22)
backend = build_backend(model, base=base, window=2, stop_ids={tokenizer.eos_token_id}, m=32)
sampler = DiffuCoderDenoisingSampler(backend)
# sampler.generate(prompt_ids, attention_mask=prompt_mask, ...) uses the existing API.
```

For research runs, `python -m dlm_kv_policy.batched_decoding_study --help` exposes
`--base`, `--window`, `--m`, `--batch-size`, `--verify-b1`, and `--evalplus`.
Omit `--window` for the base alone; use `--base dense` for the native dense arm.
Use a separate empty output directory for every arm/batch size. The runner stores
length-sorted batch manifests, full task records, commit and dirty-source hashes,
configuration, torch/CUDA/GPU identity, generated tokens, optional per-step
batch-one identity checks, row diagnostics, and peak allocated GPU memory for
every padded batch. `--evalplus` scores base and full/plus scopes separately.
Change the config's task manifest and max_prompts for full datasets; the provided
64-task config remains the subset. This is a fidelity runner, not a speed benchmark.

After Michael approves GPU validation, proceed through the correctness ladder:
1. `--base dense --batch-size 1 --verify-b1` against native full-head dense.
2. Dense at batch 8/16, compared by task ID to unbatched outcomes and E3 noise.
3. `--base refresh_256 --batch-size 1 --verify-b1`.
4. The base with `--window 2` (and other intended windows), batch 1, `--verify-b1`.
5. All intended policies at batch 8/16, paired by task ID; evaluate base and plus tests.

CPU tests cover per-step batch-one identity, multiple windows/depths, independent
padded forced trajectories, dummy-write exclusion, stable ties, and diagnostic
on/off equivalence. They do not certify bf16 GPU identity or speed.

For uninstrumented generation set `collect_diagnostics=False` and
`collect_history=False`. Policy count `.item()` calls and CPU history copies are
then disabled. Row packing still uses two count synchronizations per step (three
with depth), and boolean gather/scatter can synchronize internally. Those costs,
the full-key `_repeat_kv` traffic, and padded-key attention must remain in future
end-to-end timing. Retaining the original all-masked-predecessor head also leaves
that head's cost on cheap steps. Register the speed target before timing; no speed
claim follows from implementing the backend.

## Registered proxy re-check and batch-cost calibration (2026-11-04)

Michael authorized both precursor studies. `proxy_recheck.py` prepares the audited
96-task manifest, runs four resumable GPU task shards, and scores/reports results
on CPU. Full registration is in `PROXY_VALIDATION.md`; the new ranking gate is
Spearman >=0.90 overall and within the depth family, AUROC >=0.75.

`batched_cost_model.py` is a separate engineering calibration, using the batched
backend rather than assuming its gather/scatter overhead is negligible. It first
checks B=1 identity against legacy generation for all panel policies on the first
manifest task; padded B=8/16 native-versus-full-refresh logits must be finite with
relative L2 <=0.01. These checks do not establish batch-invariant pass@1.

Before any new timing data, the cost protocol is frozen as follows:
- A100 SXM4 80GB, one allocated GPU; B=1,8,16; context buckets 384,512,640,768
  (the manifest's total lengths are 347..727). Model dtype bf16.
- Independently measure layer blocks [0,17), [17,21), [21,23), [23,28) and LM head.
  Decoder row grid: 0,1,2,4,8,16,24,32,48,64,96,128,160,192,224,256,L.
  Head uses 0..256 on the same grid. Zero-row decoder/head cost is exactly zero.
  Both mask-free and padded-key-mask paths are measured. No extrapolation.
- CUDA events, synchronization, 2 warmups, median of 5 repeats. Retain individual
  samples and synchronized wall time. This is calibration, not the later three-
  repeat generation benchmark or its still-unregistered speed-success target.
- Twelve fixed synthetic schedules in `batched_cost_model.CASES`, eight steps each,
  test depths 1,18,22,24; unseen row counts; equal/unequal requests; refresh every
  4 or 8 steps. Calibrate at context 640 and 768 for each batch size. Schedules
  use per-request different positions and valid-only cache writes, with decreasing
  masked-head rows. The initial cache population is excluded in both predictions
  and measurements; scheduled refresh steps are included.
- Predict by summing block costs at the *per-block batch maximum* row count plus
  the head cost at the remaining masked count. Replay projections length-sort the
  fixed manifest, conservatively round context up to a calibrated bucket, and
  report that rounding. They never average individual-task speedups.
- **Calibration gate, separately at each B:** median absolute relative error <=10%,
  p90 <=20%, and Spearman >=0.95. All three batch sizes must pass. This ranking
  threshold is separate from the proxy's >=0.90. Calibration records are held
  out from table fitting. Do not refit on them and call that a validation pass.
- Predictions exclude policy/sampler overhead; calibration includes the real
  backend's selection, packing, gathers/scatters and head but no sampler. Record
  the discrepancy and peak allocated memory. These are predicted model-compute
  times, not a measured decoding speedup claim. Failure is a useful result and
  blocks using the model unqualified as an evolutionary speed objective.

Runs use immutable source/config/task snapshots with hashes and separate output
directories; historical studies are not overwritten. No production backend code
or decision rule is changed based on new results without recording an amendment.
