# Dream NSGA-II Single-GPU Execution Flow

This document explains what happens when you run:

```bash
python ecad/genetic/train_nsga2_single_gpu.py \
  --image-generator DreamTextGenerator \
  --name dream_7b_random_init \
  --num-cycles 50 \
  --all-populations-dir /sciclone/home/mcmartinez/ecad/results/genetic/dream \
  --all-benchmarks-dir /sciclone/home/mcmartinez/ecad/results/benchmark/genetic/dream \
  --batch-size 8 \
  --population-size 24 \
  --embedding-dir /sciclone/home/mcmartinez/ecad/gsm8k/embeddings/train
```

This walkthrough stops before the scoring section, as requested.

## 1) Entry Point And Argument Parsing

1. The entry file is `ecad/genetic/train_nsga2_single_gpu.py`.
2. `main()` calls `parse_args()`, which uses shared arguments from `ecad/genetic/train_nsga2_base.py`.
3. `validate_base_args()` runs in `ecad/genetic/train_nsga2_base.py`.
   - For Dream, `--benchmark-prompts` is not required.
4. Generator type resolution:
   - `get_image_generator_type(...)` is attempted first (from `ecad/image_generators/load_image_generator.py`).
   - For `DreamTextGenerator`, this falls through to `get_text_generator_type(...)` in `ecad/image_generators/load_text_generator.py`.

## 2) Manager Initialization (Population + Benchmark Directories)

1. `initialize_manager(...)` in `ecad/genetic/train_nsga2_base.py` picks `DreamPopulationIOManager` from `ecad/genetic/dream_population_io_manager.py`.
2. `DreamPopulationIOManager.__init__` creates a default Dream cache schedule by calling:
   - `cache_gen_default(...)` in `ecad/schedulers/cache_scheduler/generators/dream_schedule_generators.py`.
3. That default schedule is a `Dream7bCacheSchedule` from:
   - `ecad/schedulers/cache_scheduler/dream_cache_schedule.py`
   - default behavior: recompute `layer=True`, `kv=True`, `mlp=True` at all steps.
4. Base manager setup then runs in `ecad/genetic/population_io_manager.py`:
   - Creates population and benchmark directories.
   - Resolves current generation number.
   - Stores default schedule for constraints.

## 3) Algorithm Bootstrap (Checkpoint Or Generation 0)

1. `train_nsga2_single_gpu.py` tries `manager.load_algorithm()` from `ecad/genetic/population_io_manager.py`.
2. If no checkpoint exists, it calls `init_gen_0(...)` in `ecad/genetic/train_nsga2_base.py`.
3. In `init_gen_0(...)` for Dream:
   - Problem class: `Dream7bCachingScheduleProblem` from `ecad/genetic/dream_problem.py`.
   - If no seed population exists, random binary sampling uses `BinaryRandomSampling` from `ecad/genetic/sampling.py`.
   - NSGA-II (`pymoo`) is configured and `algorithm.ask()` generates candidate vectors.
4. Candidate vector -> schedule JSON conversion:
   - `DreamPopulationIOManager.save_population(...)` in `ecad/genetic/dream_population_io_manager.py`.
   - Uses `binary_vector_to_schedule_dict(...)` mapping each bit triplet to:
     - `layer`, `kv`, `mlp`.
   - Writes each candidate schedule via `Dream7bCacheSchedule.to_json(...)`.
   - Files are saved under `gen_XXX/candidates/cand_YYY.json`.

## 4) Per-Cycle Control Loop (Single GPU Orchestrator)

File: `ecad/genetic/train_nsga2_single_gpu.py`, function `train_nsga2_single_gpu(...)`.

For each cycle:

1. `train_one_cycle(...)` in `ecad/genetic/train_nsga2_base.py` checks whether offline eval artifacts already exist.
2. `checkpoint_manager_and_algorithm(...)` stores:
   - manager config JSON
   - algorithm checkpoint `.pkl`
3. Offline command strings are built by:
   - `get_offline_eval_commands_single_gpu_from_manager(...)` in `train_nsga2_single_gpu.py`.
4. For Dream text flow (`gen_images=False` path), the first executed command is:
   - `python ecad/benchmark/generate_text.py ...`
5. This command is executed synchronously via:
   - `run_single_gpu_command(...)` in `train_nsga2_single_gpu.py`.

This document stops here with respect to scoring.

## 5) Text Generation Script Cascade

The generation command enters `ecad/benchmark/generate_text.py`:

1. `main()` parses CLI args and resolves `DreamTextGenerator` via:
   - `ecad/image_generators/load_text_generator.py`.
2. `generate_all_schedules(...)` recursively iterates candidate schedule JSON files.
3. For each schedule file, `generate_for_schedule(...)`:
   - Instantiates `DreamTextGenerator(start_seed, seed_step, schedule_path=...)`.
   - Calls `text_generator.generate_from_saved_prompts(...)`.

## 6) DreamTextGenerator Setup And Schedule Loading

File: `ecad/image_generators/dream_text_generator.py`.

### Constructor Path

1. Loads tokenizer (`AutoTokenizer`) via `load_tokenizer(...)`.
2. Calls base class constructor in `ecad/image_generators/text_generator.py`.

### Base TextGenerator Schedule Loading

File: `ecad/image_generators/text_generator.py`, method `_load_schedule_file(...)`.

1. Attempts to load DiT schedule from candidate JSON using:
   - `DreamDiTSchedule.from_json(...)` in `ecad/schedulers/dit_scheduler/dream_dit_schedule.py`.
2. Attempts to load cache schedule from same JSON using:
   - `Dream7bCacheSchedule.from_json(...)` inherited from `ecad/schedulers/cache_scheduler/cache_schedule.py`.
3. Typical Dream candidate JSON has cache schedule but no dit schedule.
   - So DiT falls back to default:
     - `DreamTextGenerator._default_dit_schedule()`
     - `ecad/schedulers/dit_scheduler/generators/dream_schedule_generators.py`.
4. Cache schedule comes from the candidate file (this is the GA-evolved schedule).
5. `DiTScheduler` is created from the DiT schedule (`ecad/schedulers/dit_scheduler/dit_scheduler.py`).
6. Callback list is assembled in order:
   - `dit_scheduler.per_step_callback`
   - `cache_schedule.per_step_callback`
   - `DreamTextGenerator._reset_schedules_callback`

## 7) Prompt Loading And Batching

In `TextGenerator.generate_from_saved_prompts(...)` (`ecad/image_generators/text_generator.py`):

1. Calls `create_generation_pipeline()` if needed.
2. Loads tokenized prompt `.pt` files using:
   - `GSM8KTokenizedPtDataset` in `ecad/dataset_utils/prompt_embedding_dataset.py`.
3. Batches are collated with `pad_tokenized_batch(...)` into:
   - `input_ids` `[B, Lmax]`
   - `attention_mask` `[B, Lmax]`
4. For each batch, calls `DreamTextGenerator.generate_text(...)`.

## 8) Where Model Forward Calls Happen

### Generation Loop Entry

File: `ecad/image_generators/dream_text_generator.py`, method `generate_text(...)`.

1. Builds `DreamGenerationConfig` (`steps`, `max_new_tokens`, etc.).
2. Calls:
   - `model.diffusion_generate(...)`
   - where `model` is `DreamForCausalLMEdited`.
3. Passes `generation_tokens_hook_func`, which triggers callbacks each denoise step.

### DLM Sampling Loop

File: `ecad/lm_models/dream/generation_utils.py`, method `_sample(...)`.

1. Iterates for `steps`.
2. Each step executes model forward at:
   - `logits = self(x, attention_mask, tok_idx).logits`
3. This call enters Dream model forward stack in:
   - `ecad/lm_models/dream/modeling_dream.py`
   - `DreamForCausalLM.forward(...)`
   - `DreamBaseModel.forward(...)`
   - decoder layer loop.

## 9) Custom Dream Cache Functionality (Where Schedule Is Used)

### Cached Layer Replacement

File: `ecad/lm_models/dream/modeling_dream_cached.py`.

1. `DreamForCausalLMEdited.from_pretrained(...)` loads base Dream weights.
2. `_post_init(...)` is called with `dit_scheduler` and `cache_schedule`.
3. If cache schedule exists, `init_cached_decoder_layers()` replaces each decoder layer with `CachedDreamDecoderLayer` and copies weights.

### Per-Layer Schedule Checks

Still in `ecad/lm_models/dream/modeling_dream_cached.py`:

1. `CachedDreamDecoderLayer.forward(...)` reads:
   - `cache_schedule.get_recompute(layer_num, "layer")`
2. `_compute_attn_cached(...)` reads:
   - `cache_schedule.get_recompute(layer_num, "kv")`
3. `_compute_mlp_cached(...)` reads:
   - `cache_schedule.get_recompute(layer_num, "mlp")`
4. Based on those booleans, it either recomputes or reuses stored tensors.

This is the core custom logic that maps GA-produced schedule bits to actual compute skipping/reuse behavior.

## 10) How Step Indexing Aligns With Schedule Timesteps

1. `cache_schedule.curr_step` is defined in `ecad/schedulers/cache_scheduler/cache_schedule.py`.
2. It is advanced by `cache_schedule.per_step_callback(step, timestep, ...)`.
3. In Dream generation, callback firing is controlled by:
   - `generation_tokens_hook_func` in `DreamTextGenerator.generate_text(...)`.
4. Callback order each step:
   - scheduler step update callbacks run
   - final-step reset callback runs when `step == num_inference_steps - 1`.
5. `DreamTextGenerator` also explicitly resets schedules and model caches between sequential generations per prompt.

## 11) Role Of DreamDiTSchedule In Current Dream Text Path

`DreamDiTSchedule` (`ecad/schedulers/dit_scheduler/dream_dit_schedule.py`) is currently a minimal compatibility schedule:

1. It stores transformer blocks through `update_transformer_blocks(...)`.
2. Its `forward(...)` is a no-op passthrough on hidden states.
3. In the current Dream text path, the main forward pass is driven by Dream's own model loop, not by calling `dit_scheduler.forward(...)`.
4. Even so, `DiTScheduler` callbacks keep a synchronized per-step state so the shared scheduling infrastructure remains consistent.

## 12) Artifacts Written During This Flow

From this command path (through generation stage):

1. Population candidates:
   - `<all-populations-dir>/<name>/gen_XXX/candidates/cand_YYY.json`
2. Manager checkpoint/config:
   - `manager_config.json`
   - `checkpoint.pkl`
3. Generated text outputs:
   - `<all-benchmarks-dir>/<name>/gen_XXX/candidates/cand_YYY/.../*.txt`

These outputs are what downstream scoring expects, but scoring is outside this walkthrough.
