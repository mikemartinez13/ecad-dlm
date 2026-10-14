<p align="center">
  <img src="assets/images/teaser.png" width=90%>
</p>

# [ICLR 2026] Evolutionary Caching to Accelerate Your Off-the-Shelf Diffusion Model

This repository contains the official implementation of ECAD (Evolutionary Caching for Accelerated Diffusion), a method that uses evolutionary algorithms to optimize caching schedules for diffusion models. ECAD automatically discovers optimal layer-wise caching patterns that accelerate inference while maintaining image quality. Our approach works with off-the-shelf diffusion models including PixArt-α, PixArt-Σ, and FLUX, achieving significant speedups without requiring model retraining or architectural modifications.


<div align="center">
<a href="https://research.aniaggarwal.com/ecad"><img src="https://img.shields.io/static/v1?label=Project%20Page&message=Website&color=blue&logo=github-pages"></a> &ensp;
<a href="https://arxiv.org/abs/2506.15682"><img src="https://img.shields.io/static/v1?label=Paper&message=Arxiv:ECAD&color=red&logo=arxiv"></a> &ensp;
</div>

## Requirements

Set up the environment and install all necessary dependencies. Ensure that you are using Python 3.10 or later as specified in `ecad.yml`.

1. **Create a virtual environment**

   ```bash
   conda env create -f ecad.yml
   conda activate ecad
   pip install -r requirements.txt
   pip install -e .
   ```

   For the LLaDA/DiffuCoder diffusion-language-model harness, use the dedicated
   `ecad-dlm` environment instead. This installs the pinned Python 3.10
   dependencies, the CUDA 12.4 PyTorch wheels, and this package in editable
   mode:

   ```bash
   conda env create -f ecad-dlm.yml
   conda activate ecad-dlm
   ```

   Both setups install this repository as an editable package. If you run into issues with imports, run `python -m pip install -e .` from the repository root with the intended environment active.

2. **Configure environment variables**:

   Copy the `.env_template` file as follows:

   ```bash
   cp .env_template .env
   ```

   Then modify the `.env` file to match your local setup. Update any paths accordingly. For example:

   ```bash
   # In .env
   PROJECT_ROOT="~/ecad" # SET THIS TO YOUR PROJECT ROOT, i.e. where this file is located, as returned by `git rev-parse --show-toplevel`
   ```

   DLM launchers use `python` from the active environment and assume jobs are
   submitted from the repository root. Their examples below do not require `.env`.

   Replace the example path with the actual location of your project directory on your system.

### Preventing OOM Issues
When running any of the scripts, make sure to have set the below variable; alternatively add this to your shell rc file.
This will prevent OOM issues during image generation.

```bash
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

## Evolutionary caching for DLMs

Once your environment is set up and `ecad-dlm` is active, you can proceed with initial experiments. We use HumanEval+ and MBPP+ to evaluate DiffuCoder. Install the evaluation package if it is not already available:

```bash
python -m pip install evalplus==0.3.1
```

HumanEval asks a model to complete Python functions from their signatures and docstrings. MBPP (Mostly Basic Python Problems) provides short programming problems with example tests. [HumanEval+ and MBPP+](https://github.com/evalplus/evalplus/tree/v0.3.1) add tests to catch incorrect solutions that pass the original tests. Our initial experiments use 32 tasks from each benchmark.

Most experiments run as Slurm batch jobs. The `.sh` launchers call Python programs that generate responses, score schedules, and run NSGA-II optimization. The shared ECAD runner supports DiffuCoder and LLaDA through YAML `model.family` and keeps one model on the GPU throughout the search. Run the commands below from the repository root with your Python environment active.

This section covers schedules that reuse attention and MLP outputs across denoising steps. The caching code is in `ecad/language_models/diffucoder_residual_output_cache.py` and `residual_output_cache_runtime.py`; schedules are defined in `ecad/schedulers/cache_scheduler/dlm_cache_schedule.py`. The separate [DiffuCoder KV policy guide](dlm_kv_policy/README.md) covers token-level K/V caching and its evolutionary experiments.

### Model and benchmark files

Download the DiffuCoder snapshot used by the supplied configs:

```bash
huggingface-cli download apple/DiffuCoder-7B-cpGRPO \
  --revision 98bc42bfd871eeec92fd0a8439ab34f75b405204 \
  --local-dir /path/to/DiffuCoder-7B-cpGRPO
```

For offline evaluation, download `HumanEvalPlus.jsonl.gz` from the [HumanEval+ v0.1.10 release](https://github.com/evalplus/humanevalplus_release/releases/tag/v0.1.10) and `MbppPlus.jsonl.gz` from the [MBPP+ v0.2.0 release](https://github.com/evalplus/mbppplus_release/releases/tag/v0.2.0). Decompress them with `gzip -d` and use the resulting JSONL paths in your config. These versions match EvalPlus 0.3.1 and include both the original and additional tests.

Use an EvalPlus container containing `evalplus==0.3.1`; set its path to `/path/to/evalplus.sif`. Apptainer containers use the `.sif` extension. See the [DLM guide](assets/docs/dlm.md#evalplus-container-and-offline-data) for building or using a shared image.

### Experiment configuration and results

Copy the correctness-aware config and edit the paths for your setup:

```bash
cp configs/dlm/diffucoder7b_residual_cache_base_regression_search_64tasks_256tokens.yaml \
  configs/dlm/my_diffucoder_code_search.yaml
```

| YAML field | Example value |
| --- | --- |
| `run.name` | `diffucoder_code_search` |
| `model.path` | `/path/to/DiffuCoder-7B-cpGRPO` |
| `evaluation.code_tasks_file` | `/path/to/ecad/data/code/humaneval_mbpp_64_strict_python_seed0.jsonl` |
| `evaluation.evalplus_image` | `/path/to/evalplus.sif` |
| `evaluation.evalplus_humaneval_path` | `/path/to/HumanEvalPlus.jsonl` |
| `evaluation.evalplus_mbpp_path` | `/path/to/MbppPlus.jsonl` |
| `evaluation.baseline_artifact` | `null` to generate a baseline for this run |

Results default to `/scratch/<username>/ai4mat/ecad/results`. Evolution creates `dlm_ecad/<run.name>/` under that directory automatically. To change the location, add `run.output_dir: /path/to/results` in the YAML; the run will then be stored at `/path/to/results/<run.name>/`. You can also override it on the command line:

```bash
sbatch scripts/slurm/run_dlm_ecad.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --output-dir /path/to/results
```

The supplied task JSONL contains selected prompts and benchmark metadata. If you need to recreate it, the following command writes `data/code/humaneval_mbpp_64_strict_python_seed0.jsonl`:

```bash
python scripts/prepare_dlm_code_tasks.py \
  --datasets humaneval mbpp --max-tasks 64 \
  --seed 0 --prompt-style strict_python
```

The launchers use the active environment's `python` and the repository where you submit the job. Adjust their Slurm partition, account, and GPU constraints for your cluster. Logs default to `/scratch/<username>/slurm-<job-name>-<job-id>.out`; override this with `sbatch --output=/path/to/job.log` if needed.

### Generate one DiffuCoder response

Within a GPU allocation, run one prompt without a caching schedule:

```bash
python -m ecad.inference.generate_with_dlm_residual_cache_schedule \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --prompt "Write a Python function that adds two integers."
```

This runs the all-recompute baseline and writes the response, elapsed time, and cache statistics to `/scratch/<username>/ai4mat/ecad/results/dlm_inference/sample.json`, creating the directory if needed. Add `--output /path/to/sample.json` to change the destination. Add `--schedule /path/to/cand_000.json` to use an evolved schedule, keeping the same model and generation settings that produced it.

### Run evolution

The example config uses 64 tasks, 256 generated tokens, eight candidates per generation, and 20 generations. It minimizes the fraction of baseline-passing tasks that fail under caching and the estimated compute cost relative to all-recompute inference. Syntax-invalid programs make a candidate infeasible; model weights stay fixed.

Run one generation:

```bash
sbatch scripts/slurm/run_dlm_ecad.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --max-generations 1
```

After that job finishes, run the remaining generations up to `evolution.generations`. The same command starts the full search if no checkpoint exists:

```bash
sbatch scripts/slurm/run_dlm_ecad.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml
```

With `run.resume: true`, resubmitting continues from the checkpoint. `--max-generations 1` means one additional generation; `--max-generations 0` prepares and scores only the baseline. Keep the config unchanged when resuming, and use a new `run.name` for a separate experiment. If you override the output directory at submission, use that same directory in the YAML before scoring or resuming the run.

### Score a proposed schedule

Correctness-aware evolution scores each proposed schedule on the original benchmark tests. It saves the schedule as `cand_000.json` and its responses and search metrics as `cand_000.metrics.json`. After the search job finishes, score a candidate on both the original and additional Plus tests:

```bash
sbatch scripts/slurm/run_saved_dlm_code_evalplus_scoring.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml \
  --artifact generations/gen_000/candidates/cand_000.metrics.json
```

Score the baseline for comparison:

```bash
sbatch scripts/slurm/run_saved_dlm_code_evalplus_scoring.sh \
  --config configs/dlm/my_diffucoder_code_search.yaml
```

The scorer reads the run directory, container, and benchmark paths from the YAML. Relative artifact paths refer to files inside that run. It reconstructs executable Python from saved responses and writes a `.code_eval.json` beside the input artifact, without loading DiffuCoder again. A schedule JSON alone contains no generated programs to score.

The host job uses your active environment; generated programs execute with Python and EvalPlus inside the Apptainer image. The evaluator disables network access and GPU visibility, removes host home and working-directory mounts, and binds the evaluator code and benchmark data read-only. Scoring stops if the cluster cannot provide the required isolation. Plus tests are used after the search and do not guide evolution.

`base_pass_at_1` is the fraction of selected tasks passing all original tests. `plus_pass_at_1` requires passing both the original and additional tests. These are subset scores, rather than full-benchmark results. See the [DLM guide](assets/docs/dlm.md) for implementation details, experiment variants, and saved files, or the [LLaDA guide](assets/docs/dlm-llada.md) for its sampler settings and benchmark controls.

## ECAD Optimization

ECAD optimization is achieved by running the `train_nsga2_single_gpu.py` script, which implements the NSGA‑II algorithm for single GPU execution. This script runs the optimization process sequentially, executing image generation, scoring, and metrics computation steps one after another.

We **recommend** using `nohup` when running on a remote server to protect against SSH connectivity issues.

Below is an example command to optimize the PixArt-α model at 256×256 resolution. Results (all schedules by generation) will be saved under `results/genetic/pixart_alpha/pixart_alpha_256_random_init` and all benchmark results will be in `results/benchmark/genetic/pixart_alpha/pixart_alpha_256_random_init`:

```bash
nohup python ecad/genetic/train_nsga2_single_gpu.py \
    --image-generator PixArtAlphaImageGenerator \
    --name pixart_alpha_256_random_init \
    --num-cycles 50 \
    --all-populations-dir results/genetic/pixart_alpha \
    --all-benchmarks-dir results/benchmark/genetic/pixart_alpha \
    --batch-size 100 \
    --population-size 72 \
    --embedding-dir /path/to/results/embeddings/image_reward/pixart_alpha_embeddings \
    --benchmark-prompts prompts/ImageRewardPrompts.json \
    &> nohup_pixart_alpha_ecad_optimize.out & disown
```

**Note**: When starting a new optimization run without an existing population, you will be prompted to confirm random initialization.

To **resume** optimization from a previous checkpoint, replace the `--name` flag with:

```bash
--load-from results/genetic/pixart_alpha/pixart_alpha_256_random_init/gen_010/manager_config.json
```

For a full list of options and parameters, simply append the `--help` flag:

```bash
python ecad/genetic/train_nsga2_single_gpu.py --help
```

### Using an Initial Population

To use some custom initial population, instead of random initialization, see the `Initial Population` [ReadMe](assets/docs/initial-population.md).

### ECAD Schedules

For more details on schedules, see the `ECAD Schedules` [ReadMe](assets/docs/ecad-schedules.md).

### Preparing Prompts

To precompute prompt embeddings, see the `Preparing Prompts` [ReadMe](assets/docs/preparing-prompts.md).

## Image Generation

For quick inference without pre-generating embeddings, use the `inference.py` script. This tool allows you to generate images directly from prompts or prompt files, automatically handling embedding generation and image creation in one step. We provide some helpful examples.

Generate images from a single text prompt, using one of our caching schedules:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --schedule schedules/schedules_in_paper/ours_fast.json \
    --prompt "A fremen stands atop a towering sand dune, clad in a stillsuit and gripping a worm hook in each hand, facing the vast, sun-scorched desert." \
    --num-images-per-prompt 5 \
    --output-dir results/inference/single_prompt
```

Generate images for multiple prompts from a text file, without caching:

```bash
python ecad/inference/inference.py \
    PixArtAlphaImageGenerator \
    --prompt-file prompts/my_prompts.txt \
    --num-images-per-prompt 10 \
    --batch-size-generate 4 \
    --output-dir results/inference/batch_prompts
```

For more options and scripts, including how to use your precomputed text embeddings, see the `Image Generation` [ReadMe](assets/docs/image-generation.md)

## Benchmarking

For details on latency (MACs, time) and quality (Image Reward, FID, CLIP Score) metrics, see the `Benchmarking` [ReadMe](assets/docs/benchmarking.md).

## Datasets

For details on setting up and computing metrics for datasets that we use in our paper (COCO and MJHQ), see the `Datasets` [ReadMe](assets/docs/datasets.md).

## Copied repositories

Place copied or cloned upstream repositories directly at the ECAD repository root, under their own directory names. For example, [dkv_cache_official/](dkv_cache_official/) contains the official dKV-Cache source, while [dkv_repro/](dkv_repro/README.md) contains its local launchers and reproduction code. Use this layout for additional repositories as well.

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
