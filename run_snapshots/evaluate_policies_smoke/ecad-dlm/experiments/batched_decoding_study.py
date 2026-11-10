"""Fixed-batch DiffuCoder generation and batch-one reference validation.

This entrypoint does not submit jobs. Run only inside an approved allocation.
Timing benchmarks must wait for a registered speed target and correctness ladder.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from language_models.diffucoder_loader import prepare_diffucoder_prompt_batch
from language_models.code_tasks_and_response_processing import make_code_task_output
from dlm_kv_policy.batched_diffucoder_backend import BatchedDiffuCoderKVBackend
from dlm_kv_policy.dense_diffucoder_baseline import DenseDiffuCoderBaseline, BatchedDenseDiffuCoderBaseline
from dlm_kv_policy.diffucoder_active_token_backend import DiffuCoderActiveTokenKVBackend
from dlm_kv_policy.diffucoder_denoising_sampler import DiffuCoderDenoisingSampler
from dlm_kv_policy.masked_row_reuse_policy import MaskedRowReusePolicy


def build_backend(model, *, base=None, window=None, stop_ids=(), m=32, reference=False):
    if base is None:
        if window is not None:
            raise ValueError("Masked reuse requires a committed-row base")
        return DenseDiffuCoderBaseline(model) if reference else BatchedDenseDiffuCoderBaseline(model)
    if window is not None:
        policy = MaskedRowReusePolicy(base, window, m, 'split+recent', set(stop_ids))
        return policy.make_backend(model) if reference else BatchedDiffuCoderKVBackend(model, policy)
    return DiffuCoderActiveTokenKVBackend(model, base) if reference else BatchedDiffuCoderKVBackend(model, base)


def fixed_batches(tasks, tokenizer, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    lengths = []
    for task in tasks:
        ids, _ = prepare_diffucoder_prompt_batch(tokenizer, [task.prompt], 'cpu')
        lengths.append(ids.shape[1])
    order = sorted(range(len(tasks)), key=lambda i: (lengths[i], tasks[i].task_id))
    return [order[i:i + batch_size] for i in range(0, len(order), batch_size)], lengths


def source_provenance():
    root = Path(__file__).resolve().parents[2]
    def git(*args):
        return subprocess.check_output(['git', '-C', str(root), *args], text=True).strip()
    # Hash dirty and untracked runtime sources as well as the commit.
    files = [p for package in ('dlm_kv_policy', 'evolution', 'experiments', 'language_models')
        for p in sorted((root / 'ecad-dlm' / package).glob('*.py'))]
    return {'commit': git('rev-parse', 'HEAD'), 'git_status': git('status', '--short'),
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--base', default='depth_22', help='dkv_policies entry, or dense')
    parser.add_argument('--window', type=int, help='Positive reuse period; omit for base alone')
    parser.add_argument('--m', type=int, default=32)
    parser.add_argument('--verify-b1', action='store_true', help='Compare every step to the legacy reference; abort on mismatch')
    parser.add_argument('--evalplus', action='store_true', help='Score base and plus tests separately')
    parser.add_argument('--max-tasks', type=int)
    args = parser.parse_args()
    if args.verify_b1 and args.batch_size != 1:
        parser.error('--verify-b1 requires --batch-size 1')
    if args.output.exists() and any(args.output.iterdir()):
        parser.error('output must be empty to avoid mixing source versions or policies')
    from dlm_kv_policy.kv_policy_experiment_config import load_kv_policy_experiment_config
    from dlm_kv_policy.code_correctness_and_latency_evaluator import DiffuCoderKVPolicyEvaluator, seed_everything
    config = load_kv_policy_experiment_config(args.config)
    if args.max_tasks is not None:
        if args.max_tasks < 1:
            parser.error('--max-tasks must be positive')
        config.evaluation.max_prompts = args.max_tasks
    generation = config.generation
    if (generation.algorithm != 'entropy' or generation.temperature != 0
            or generation.algorithm_temperature != 0 or generation.top_p != 1 or generation.top_k is not None):
        parser.error('Use greedy entropy decoding with top_p=1 and top_k=null')
    evaluator = DiffuCoderKVPolicyEvaluator(config)
    model, tokenizer, tasks = evaluator.model, evaluator.tokenizer, evaluator.tasks
    base = None if args.base == 'dense' else config.dkv_policies[args.base]
    stops = {t for t in (tokenizer.eos_token_id, tokenizer.pad_token_id) if t is not None}
    options = dict(base=base, window=args.window, stop_ids=stops, m=args.m)
    backend = build_backend(model, **options)
    sampler = DiffuCoderDenoisingSampler(backend)
    batches, lengths = fixed_batches(tasks, tokenizer, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {**source_provenance(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(), 'config': asdict(config),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'checkpoint_sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path(config.model.path).iterdir() if p.suffix in ('.py', '.json')},
        'base': args.base, 'window': args.window, 'm': args.m,
        'batch_size': args.batch_size, 'batches': [[tasks[i].task_id for i in b] for b in batches],
        'prompt_lengths': dict(zip((t.task_id for t in tasks), lengths)),
        'tasks': [t.to_dict() for t in tasks], 'purpose': 'fidelity; elapsed times are not benchmark results'}
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2, default=str))
    settings = dict(steps=generation.steps, gen_length=generation.gen_length,
        temperature=generation.temperature, top_p=generation.top_p, top_k=generation.top_k,
        algorithm=generation.algorithm, algorithm_temperature=generation.algorithm_temperature)
    records = []
    for batch_id, indices in enumerate(batches):
        seed_everything(generation.seed)
        prompt, mask = prepare_diffucoder_prompt_batch(tokenizer, [tasks[i].prompt for i in indices], config.model.device)
        torch.cuda.reset_peak_memory_stats()
        result = sampler.generate(prompt, attention_mask=mask, collect_diagnostics=True,
                                  collect_history=args.verify_b1, **settings)
        peak = torch.cuda.max_memory_allocated()
        if args.verify_b1:
            reference = DiffuCoderDenoisingSampler(build_backend(model, **options, reference=True))
            expected = reference.generate(prompt, attention_mask=mask, collect_history=True, **settings)
            mismatches = [i for i, (a,b) in enumerate(zip(result.history, expected.history)) if not torch.equal(a,b)]
            if mismatches:
                (args.output / 'identity_failure.json').write_text(json.dumps({'task': tasks[indices[0]].task_id,
                    'steps': mismatches, 'actual': result.sequences.cpu().tolist(), 'reference': expected.sequences.cpu().tolist()}))
                raise RuntimeError(f'Batch-one identity failed at steps {mismatches}')
            del reference
        tokens = result.sequences[:, prompt.shape[1]:].cpu().tolist()
        responses = tokenizer.batch_decode(result.sequences[:, prompt.shape[1]:], skip_special_tokens=True)
        batch_record = {'batch': batch_id, 'peak_allocated_bytes': peak, 'padded_sequence_length': result.sequences.shape[1],
                        'diagnostics': result.diagnostics, 'tasks': []}
        for index, ids, response in zip(indices, tokens, responses):
            record = {'task_id': tasks[index].task_id, 'token_ids': ids,
                      'output': make_code_task_output(tasks[index], response)}
            records.append(record)
            batch_record['tasks'].append(record)
        (args.output / f'batch_{batch_id:04d}.json').write_text(json.dumps(batch_record, indent=2))
        print(f'batch {batch_id+1}/{len(batches)}: {len(indices)} tasks', flush=True)
    summary = {'tasks': len(records), 'batch_one_identity_passed': True if args.verify_b1 else None}
    if args.evalplus:
        outputs = [r['output'] for r in records]
        summary['base_tests'] = evaluator._evalplus(outputs, args.output / 'base_tests', scope='base')
        summary['plus_tests'] = evaluator._evalplus(outputs, args.output / 'plus_tests', scope='full')
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
