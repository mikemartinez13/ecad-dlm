# Evolutionary Caching for Diffusion Language Models

This repository extends ECAD (Evolutionary Caching for Accelerated Diffusion) from image diffusion models to diffusion language models. The code in [`ecad-dlm/`](ecad-dlm/README.md) uses NSGA-II to search for K/V caching policies that speed up [DiffuCoder-7B-cpGRPO](https://huggingface.co/apple/DiffuCoder-7B-cpGRPO) on code generation while keeping its answers correct. A policy decides, at every denoising step, which tokens' keys and values to recompute and which to reuse. The search mixes caching mechanisms from recent work, needs no retraining, and is checked at the end against real HumanEval+ and MBPP+ tests.

<div align="center">
<a href="https://arxiv.org/abs/2506.15682"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:ECAD&color=red&logo=arxiv"></a> &ensp;
<a href="https://huggingface.co/apple/DiffuCoder-7B-cpGRPO"><img src="https://img.shields.io/static/v1?label=Model&message=DiffuCoder-7B-cpGRPO&color=yellow&logo=huggingface"></a> &ensp;
</div>

The image-model code in `ecad/` is described in the [original ECAD repository](https://github.com/AniAggarwal/ecad).

## Requirements

Set up the environment and install all necessary dependencies. The DLM code uses Python 3.10, as specified in `ecad-dlm.yml`.

1. **Create a virtual environment**

   ```bash
   conda env create -f ecad-dlm.yml
   conda activate ecad-dlm
   export PYTHONPATH="$PWD/ecad-dlm:$PWD"
   ```

   Run the last line from the repository root. `ecad-dlm/` is not an installable package, so it must be on the path.

2. **Download the model and benchmarks**

   ```bash
   huggingface-cli download apple/DiffuCoder-7B-cpGRPO \
       --revision 98bc42bfd871eeec92fd0a8439ab34f75b405204 \
       --local-dir /path/to/DiffuCoder-7B-cpGRPO
   ```

   Generated programs are tested by [EvalPlus](https://github.com/evalplus/evalplus) inside an Apptainer image, with no network access. Building the image and downloading `HumanEvalPlus.jsonl` and `MbppPlus.jsonl` are covered in the [DLM guide](assets/docs/dlm.md#evalplus-container-and-offline-data).

### Preventing OOM Issues
When running any of the scripts, make sure to have set the below variable; alternatively add this to your shell rc file.

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## How It Works

DiffuCoder writes a 256-token answer in 256 denoising steps, one token per step. Without caching, every step recomputes every position through all 28 layers. A caching policy skips most of that work. The search treats a policy as a list of genes, scores each candidate for **quality** and **cost**, and keeps the fastest candidates that are still faithful to the uncached model (*dense*).

### Features we search over

Each policy has 12 genes. Most come from a published caching method, and the search is free to combine them.

| Gene | Values | What it does | From |
| --- | --- | --- | --- |
| `delay` | 1, 2 | Wait this many steps after a token is decoded before caching it, since its features are still settling. | dKV-Cache [1] |
| `refresh_interval` | 4, 8, 16, 32 | Recompute every position from scratch every N steps. | dKV-Cache [1] |
| `refresh_depth` | 1, 18, 22, 24 | Between full refreshes, decoded tokens recompute only the layers from this depth up; shallow layers change little. 1 means all layers. | Elastic-Cache [2] |
| `terminal_refresh` | 0, 8, 16 | Recompute everything for the last N steps. | Ours |
| `masked_window` (r) | 1, 2, 4, 8 | Still-masked positions reuse their features for r steps instead of being recomputed every step. | Fast-dLLM DualCache [3], d²Cache [4] |
| `protected_size` (m) | 32, 48, 64 | The m masked positions most likely to be decoded next are always kept fresh. | d²Cache [4] |
| `protected_rule` | split, split+recent | How those positions are chosen: the most confident code and padding positions, optionally plus neighbors of recently decoded tokens. | d²Cache [4]; the padding split is ours |
| `prompt_refresh_interval` | 0, 8, 32 | The prompt gets its own, slower refresh schedule. 0 means it refreshes with everything else. | dLLM-Cache [5] |
| `local_refresh_radius` | 0, 2, 4 | Positions within this distance of the newest decoded token are fully recomputed on that step. | dKV-Cache-Greedy [1], Polestar [6] |
| `padding_window` | 0, 8, 16, 32 | Masked positions that look like trailing padding refresh on this longer window. 0 means the same as `masked_window`. | DPad [7], Streaming-dLLM [8] |
| `entropy_threshold` | 0, 1.5, 0.5 | If a newly decoded token is uncertain (entropy above this, in nats), refresh on the next step. 0 is off. | EntropyCache [9] |
| `entropy_scope` | local, full | That refresh covers either the decoded tokens near the uncertain one, or the whole sequence as in the paper. | EntropyCache [9] |

These are the ranges of the first search, about 500,000 policies. The second search widens the genes the first one pushed against; see `diffucoder7b_constrained_kv_search_v2_96tasks.yaml`.

### Scoring quality: EAR

Quality is measured by **forced-trajectory replay** (`dlm_kv_policy/forced_trajectory_replay.py`). Dense decodes each task as usual, and every candidate policy is shown the exact same sequence at each step while keeping its own, possibly stale, cache. Comparing the two at the position dense commits gives a clean per-step measure, without one early mistake cascading into a different answer.

The comparison is the **Expected Acceptance Rate** (EAR) from Helcig, Kurtić and Alistarh [10]:

    EAR = Σ_{k in dense's top 10} min(p_dense(k), p_policy(k))

It is the probability that the two models would pick the same token, close to 1 − TV(p, q). We average it over the answer's tokens and use it as a **constraint, not an objective**: a policy is *feasible* if

- the 95% lower bound of its mean EAR is at least **0.9870**,
- its 10th-percentile task EAR is at least **0.9758**, and
- at least **99%** of answer tokens are decoded at positions whose predictions were actually recomputed (*answer coverage*).

Feasible policies count as equally faithful and compete only on speed. Infeasible ones rank by how far they miss. This is the constrained-domination rule from NSGA-II [11]. The thresholds were fitted on 11 policies with known pass rates, as the lowest EAR that keeps every policy within 2 tasks of dense (see `ecad-dlm/experiments/EAR_VALIDITY_STUDY.md`).

### Scoring cost

Cost is the predicted wall time of batched decoding at batch size 8 (`experiments/batched_cost_model.py`). We time the model's layer blocks once on an A100 80GB for many combinations of sequence length and recomputed rows. Replay records how many rows each policy recomputes, and at what depth, at every step, and the table turns those counts into seconds. Predicted speedups rank real batched speedups well (Spearman 0.98 over 21 policies), but they overestimate them, because they leave out indexing and sampler overhead. Only the ranking is relied on.

### Why EAR is a decent measure, but not a perfect one

EAR is cheap: one replay pass, no program execution. It is continuous, and it reliably separates broken policies from good ones: every policy with a net loss of 12 or more tasks was rejected on two separate task sets. But:

- **It is blind to masked reuse.** The positions that get decoded are the protected ones, which are always fresh, so EAR barely moves as `masked_window` and `protected_size` change. Answer coverage is the guard there.
- **It does not rank near-dense policies.** Among policies within a few tasks of dense, EAR's order barely tracks their pass rates.
- **It measures one step at a time.** Replay removes compounding errors by design, but free-running generation does compound.
- **Its thresholds did not confirm.** On 96 new tasks, the thresholds rejected several good policies, and the 2-task margin turned out to be within noise: swapping only the attention kernel moves dense by 3 tasks.

EAR comes from LLM quantization, where it measures how closely a compressed model matches the original. Whether it also predicts test results for caching had to be checked here, and it only partly does. It works as a filter that keeps the search away from clearly lossy policies, and every result is confirmed by running the tests.

## Running the Evolution

Each search runs on three disjoint sets of 96 tasks (28 HumanEval, 68 MBPP) in `ecad-dlm/experiments/task_sets/`. The **search** set scores candidates, the **audit** set is touched once at the end, and the **tau confirmation** set checked the thresholds. They were built by `experiments/task_sets.py`.

Start from a supplied config and edit its paths:

```bash
cp ecad-dlm/experiments/configs/diffucoder7b_constrained_kv_search_v2_96tasks.yaml \
    ecad-dlm/experiments/configs/my_search.yaml
```

| YAML field | Example value |
| --- | --- |
| `model.path` | `/path/to/DiffuCoder-7B-cpGRPO` |
| `evaluation.code_tasks_file` | `/path/to/ecad/ecad-dlm/experiments/task_sets/search_96_seed20261107.jsonl` |
| `evaluation.evalplus_image` | `/path/to/evalplus.sif` |
| `evolution.cost_table` | `/path/to/cost/table.json` |
| `evolution.dense_reference` | `/path/to/results/my_search/dense` |
| `evolution.initial_population` | a previous run's `generation_XXX/summary.json`, or remove it to start fresh |
| `evolution.generations` | `24` |

The cost table is built once, on an A100 80GB; this also runs a calibration check and takes a few hours:

```bash
python -m experiments.batched_cost_model \
    --config ecad-dlm/experiments/configs/my_search.yaml \
    --output /path/to/cost
```

On a Slurm cluster, the launcher submits the whole chain: dense reference, then for each generation a CPU `advance` step and 16 GPU replay shards, then the audit and the EvalPlus extra tests. The scripts hold Rivanna paths; edit `ARTIFACTS` in the launcher and `PYTHON_BIN` in `run_proxy_cost_prerequisites_rivanna.sh` for your cluster.

```bash
RUN=my_search CONFIG=my_search.yaml ACCOUNT=my_account AUDIT_TOP=8 OVERRIDE_TAU_GATE=1 \
    bash ecad-dlm/experiments/scripts/slurm/submit_constrained_search_rivanna.sh
```

**Note**: the launcher refuses to start unless the τ confirmation passed. It did not (see `ecad-dlm/experiments/CONSTRAINED_SEARCH.md`), so `OVERRIDE_TAU_GATE=1` is required.

To run on a single GPU instead, step through the generations by hand:

```bash
CONFIG=ecad-dlm/experiments/configs/my_search.yaml
OUT=/path/to/results/my_search
python -m evolution.constrained_kv_policy_search dense --config $CONFIG --output $OUT
for generation in $(seq 0 23); do
    python -m evolution.constrained_kv_policy_search advance --config $CONFIG --output $OUT
    python -m evolution.constrained_kv_policy_search evaluate --config $CONFIG --output $OUT \
        --generation $generation --parts 1
done
python -m evolution.constrained_kv_policy_search advance --config $CONFIG --output $OUT
```

Each generation is saved under `$OUT/generation_XXX/`: `genomes.json` holds its candidates, and `summary.json` holds their scores, the surviving population and every policy tried so far. Finished tasks are kept, so to **resume** an interrupted run, rerun `evaluate` for the unfinished generation, then continue the loop from the next `advance`. To **continue** a finished run with more generations or wider ranges, point `evolution.initial_population` at its last `summary.json` and start a new output folder.

Finally, audit the fastest feasible policies on the audit set, then add the extra tests:

```bash
python -m evolution.constrained_kv_policy_search audit \
    --config ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml --output $OUT --top 8
python -m evolution.constrained_kv_policy_search plus \
    --config ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml --output $OUT
```

The search, replay and audit check that they are on an A100 80GB, since the thresholds and cost table were measured there.

## Testing a Policy

A policy is a JSON file of its 12 genes. Put the policies you want to test in a `policies/` folder, one file each; the file name becomes the policy's name. An entry copied from a search's `summary.json` works too. Three examples from the first search are in [`ecad-dlm/policies/`](ecad-dlm/policies):

```json
{
  "delay": 2, "refresh_interval": 8, "refresh_depth": 1, "terminal_refresh": 8,
  "masked_window": 8, "protected_size": 32, "protected_rule": "split+recent",
  "prompt_refresh_interval": 32, "local_refresh_radius": 4, "padding_window": 16,
  "entropy_threshold": 0.0, "entropy_scope": "local"
}
```

Test them against dense on a GPU. Each policy decodes every task freely, is scored by EvalPlus on the base and extra tests, and is timed in batches of 8:

```bash
python -m experiments.evaluate_policies \
    --config ecad-dlm/experiments/configs/diffucoder7b_audit_96tasks.yaml \
    --policies policies/ \
    --output /path/to/results/my_policies
```

Results go to `/path/to/results/my_policies/policies.json`: pass@1 on base and extra tests, tasks lost and gained against dense, and measured speedup. Add `--tasks 4` for a quick check, or `--no-timing` to skip timing. Decoding is greedy, so pass@1 is simply the fraction of tasks whose one program passes.

## Results So Far

The first search ran 8 generations of 16 candidates. On the 96 untouched audit tasks:

| Policy | Measured speedup (batch 8) | Base pass@1 | Plus pass@1 |
| --- | --- | --- | --- |
| Dense | 1.00× | 0.635 | 0.573 |
| Hand-chosen: depth 22, r = 8 | 2.02× | 0.635 | 0.562 |
| Searched: depth 22, prompt 32, radius 2, padding 16 | 2.19× | 0.646 | 0.562 |
| Searched: delay 2, terminal 8, prompt 32, radius 4, r = 8, padding 16 | **2.33×** | 0.635 | 0.552 |

The best searched policy is 15% faster than the best hand-chosen one, at a difference of at most two tasks on either test suite, which is within noise. Details are in [`CONSTRAINED_SEARCH.md`](ecad-dlm/experiments/CONSTRAINED_SEARCH.md).

## Code Layout

| Folder | Contents |
| --- | --- |
| `ecad-dlm/dlm_kv_policy/` | Caching policies, the batch-one and batched backends, the sampler, and forced-trajectory replay |
| `ecad-dlm/evolution/` | The constrained NSGA-II search (`constrained_kv_policy_search.py`) and the genome (`nsga2_kv_policy_search.py`) |
| `ecad-dlm/experiments/` | The cost model, the EAR study, policy testing, configs, task sets, Slurm scripts and study write-ups |
| `ecad-dlm/language_models/` | Model loading, code tasks and EvalPlus execution |
| `ecad-dlm/policies/` | Example policies |
| `ecad-dlm/tests/` | CPU tests |

Run the tests from the repository root:

```bash
PYTHONPATH=ecad-dlm:ecad-dlm/tests:. python -m unittest discover -s ecad-dlm/tests -p 'test_*.py'
```

## Copied Repositories

Copied or cloned upstream repositories live directly at the repository root, under their own names, such as [`dkv_cache_official/`](dkv_cache_official/) and [`Fast-dLLM/`](Fast-dLLM/).

## References

1. Ma, Yu, Fang and Wang. [dKV-Cache: The Cache for Diffusion Language Models](https://arxiv.org/abs/2505.15781). NeurIPS 2025.
2. Nguyen-Tri, Ranjan and Shen. [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/abs/2510.14973) (Elastic-Cache). ICLR 2026.
3. Wu et al. [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618). ICLR 2026.
4. Jiang et al. [d²Cache: Accelerating Diffusion-Based LLMs via Dual Adaptive Caching](https://arxiv.org/abs/2509.23094). ICLR 2026.
5. Liu et al. [dLLM-Cache: Accelerating Diffusion Large Language Models with Adaptive Caching](https://arxiv.org/abs/2506.06295). arXiv 2025.
6. Lee, Ramachandran, Kundu and Krishna. [Polestar: Drift-Aware Cache Calibration and Token Commitment for Efficient Inference of Diffusion LLMs](https://arxiv.org/abs/2607.14107). arXiv 2026.
7. Chen et al. [DPad: Efficient Diffusion Language Models with Suffix Dropout](https://arxiv.org/abs/2508.14148). ICLR 2026.
8. Xiao et al. [Streaming-dLLM: Accelerating Diffusion LLMs via Suffix Pruning and Dynamic Decoding](https://arxiv.org/abs/2601.17917). arXiv 2026.
9. Cheong, Son, Lim and Yoo. [EntropyCache: Decoded Token Entropy Guided KV Caching for Diffusion Language Models](https://arxiv.org/abs/2603.18489). ICML 2026.
10. Helcig, Kurtić and Alistarh. [Statistically-Lossless Quantization of Large Language Models](https://arxiv.org/abs/2605.02404). arXiv 2026.
11. Deb, Pratap, Agarwal and Meyarivan. A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II. IEEE Transactions on Evolutionary Computation, 2002.
12. Gong et al. [DiffuCoder: Understanding and Improving Masked Diffusion Models for Code Generation](https://arxiv.org/abs/2506.20639). ICLR 2026.
13. Liu, Xia, Wang and Zhang. [Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation](https://arxiv.org/abs/2305.01210) (EvalPlus). NeurIPS 2023.

## Contact

For issues, questions, or contributions:
- **GitHub Issues**: Please open an issue on this [GitHub repository](https://github.com/AniAggarwal/ecad/issues) for bug reports or feature requests
- **Email**: For other inquiries, contact Anirud Aggarwal at anirud@umd.edu

## BibTeX

    @inproceedings{aggarwal2026evolutionary,
      title={{Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model}},
      author={Aggarwal, Anirud and Shrivastava, Abhinav and Gwilliam, Matthew},
      booktitle={ICLR},
      year={2026}
      url={https://arxiv.org/abs/2506.15682}
    }
