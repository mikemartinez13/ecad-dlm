# DiffuCoder token-level KV policy experiments

This is a small research package inside ECAD for learning *when, where, and
for which tokens* DiffuCoder must recompute during diffusion denoising. It does
not use ECAD's existing whole-attention/MLP residual cache.

The active backend stores K/V tensors after RoPE, layer-boundary hidden states,
final normalized hidden states, and token/layer refresh ages. A policy chooses active tokens and a
one-based refresh boundary \(\ell^\star\). Current masks and newly decoded
tokens always start at layer one; stable decoded tokens may resume deeper.

The two evolutionary objectives are paired EvalPlus base-test
regression rate and measured median generation wall time. Token agreement,
first divergence, K/V drift, syntax validity, and active-row fractions are
diagnostics only.

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
Slurm logs default to `/scratch/<username>/slurm-<job-name>-<job-id>.out`. See the
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
EvalPlus-base regression rate and measured candidate wall time. Token
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
