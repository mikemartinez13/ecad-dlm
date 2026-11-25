"""Download pinned public data and make disjoint development/OOD manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
from datasets import load_dataset
from huggingface_hub import HfApi
from language_models.reasoning_tasks import math_scorer


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('/scratch/gqs7zr/dream_reasoning/data'))
    parser.add_argument('--cache', default='/scratch/gqs7zr/cache/huggingface/datasets')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    revisions = {}
    def dataset(repo, name=None, split='train'):
        if repo not in revisions:
            revisions[repo] = HfApi().dataset_info(repo).sha
        return load_dataset(repo, name, split=split, revision=revisions[repo], cache_dir=args.cache)
    def gsm(rows, split):
        return [dict(task_id=f'gsm8k/{split}/{i}', dataset='gsm8k', source_split=split,
            prompt=f"Q: {r['question']}\nA:", question=r['question'],
            answer=r['answer'].split('####')[-1].strip(), until=['Q:', '</s>', '<|im_end|>'],
            gen_length=256) for i, r in enumerate(rows)]
    train = gsm(dataset('openai/gsm8k', 'main'), 'train')
    test = gsm(dataset('openai/gsm8k', 'main', 'test'), 'test')
    rng = random.Random(20261116)
    rng.shuffle(train)
    math500 = [dict(task_id=f"math500/{r['unique_id']}", dataset='math500', source_split='test',
        prompt=f"Problem:\n{r['problem']}\n\nSolution:", question=r['problem'], answer=r['answer'],
        subject=r['subject'], level=r['level'], until=['Problem:'], gen_length=512)
        for r in dataset('HuggingFaceH4/MATH-500', split='test')]
    # Download all MATH train/test too; split identities remain explicit. MATH-500
    # is a subset of MATH test, never a separate independent confirmation set.
    scorer = math_scorer()
    math_sets = {}
    for split in ('train', 'test'):
        rows = dataset('DigitalLearningGmbH/MATH-lighteval', split=split)
        math_sets[split] = [dict(task_id=f'math/{split}/{i}', dataset='math', source_split=split,
            prompt=f"Problem:\n{r['problem']}\n\nSolution:", question=r['problem'],
            answer=scorer.remove_boxed(scorer.last_boxed_only_string(r['solution'])),
            subject=r.get('type'), level=r.get('level'), until=['Problem:'], gen_length=512)
            for i, r in enumerate(rows)]
    rng.shuffle(math500)
    manifests = {'gsm8k_calibration': train[:16], 'gsm8k_search': train[16:48],
        'gsm8k_dev_holdout': train[48:80], 'gsm8k_test': test,
        'math500_ood_pilot': math500[:32], 'math500_test': math500,
        'math_train': math_sets['train'], 'math_test': math_sets['test']}
    # Exclude identical question text across development splits, not just IDs.
    dev = sum((manifests[k] for k in ('gsm8k_calibration','gsm8k_search','gsm8k_dev_holdout')), [])
    if len({r['question'] for r in dev}) != len(dev):
        raise ValueError('Duplicate development questions; revise selection')
    hashes = {name: write(args.output / f'{name}.jsonl', rows) for name, rows in manifests.items()}
    upstream = Path(__file__).resolve().parents[1] / 'Dream'
    metadata = dict(seed=20261116, dataset_revisions=revisions, sha256=hashes,
        counts={k:len(v) for k,v in manifests.items()},
        dream_commit=subprocess.check_output(['git','-C',str(upstream),'rev-parse','HEAD'],text=True).strip(),
        protocol='Dream eval_instruct/eval.sh: zero-shot chat; GSM8K Q/A and Minerva MATH prompts',
        ood='MATH is excluded from calibration/search; pilot exposes 32 MATH-500 test items. Do not retune on them.')
    (args.output / 'manifest.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
