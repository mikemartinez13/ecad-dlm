# Official dKV-Cache reproduction

This directory is an isolated reproduction of the authors' LLaDA dKV-Cache
implementation. The unchanged author source is the root-level
[`dkv_cache_official/`](../dkv_cache_official/) copy at upstream commit
`49a76fcc43b744ec2d960137f216e419317138b1`. Do not edit that source.
Its origin and commit are recorded in `metadata/official_source.json`, which
also supplies provenance when the copy has no upstream `.git` directory.

The wrapper uses the pre-downloaded local LLaDA-8B-Instruct checkpoint and
requires `transformers==4.46.3`. Its environment is
`dkv_repro/.venv`; it inherits PyTorch/CUDA from `ecad-dlm` but installs its
own Transformers stack.

## GPU sequence

The Lily benchmark compares author dense and dKV-Decode implementations
with `L=T=128`, block length 64, and dKV refresh 4:

```bash
CASE=lily_128tokens_refresh4 LABEL=lily_a6000 \
sbatch --constraint=a6000 dkv_repro/scripts/slurm/run_official_llada_dkv_decode_benchmark.sh
```

Then run the default small dKV-Decode setting with refresh 8:

```bash
CASE=short_128tokens_refresh8 LABEL=decode_small_a6000 \
sbatch --constraint=a6000 dkv_repro/scripts/slurm/run_official_llada_dkv_decode_benchmark.sh
```

`paper_512tokens_refresh8` uses `L=T=512`, block length 32, and refresh 8 for a systems
check before a separate HumanEval/simple-evals reproduction:

```bash
CASE=paper_512tokens_refresh8 LABEL=paper_shape_a6000 \
sbatch --time=12:00:00 --constraint=a6000 dkv_repro/scripts/slurm/run_official_llada_dkv_decode_benchmark.sh
```

Each run performs one untimed warmup and three synchronized timed repeats.
Results are written as
`/scratch/gqs7zr/ai4mat/ecad/results/dkv_reproduction/<label>/<origin|decode>/reproduction.json` plus
`/scratch/gqs7zr/ai4mat/ecad/results/dkv_reproduction/<label>/comparison.json`.
