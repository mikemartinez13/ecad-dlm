"""Prepare frozen domain-search manifests from pinned, already cached data."""
import argparse
import hashlib
import json
from pathlib import Path
import random
from datasets import load_dataset


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/scratch/gqs7zr/dream_reasoning/data'))
    p.add_argument('--output',type=Path,default=Path('/scratch/gqs7zr/dream_reasoning/data_v2'))
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Choose a new directory; do not change frozen manifests')
    original=json.loads((a.source/'manifest.json').read_text())
    sets={p.stem:read(p) for p in a.source.glob('*.jsonl')}
    excluded={r['question'] for k in ('gsm8k_calibration','gsm8k_search','gsm8k_dev_holdout') for r in sets[k]}
    rows=load_dataset('openai/gsm8k','main',split='train',
        revision=original['dataset_revisions']['openai/gsm8k'],
        cache_dir='/scratch/gqs7zr/cache/huggingface/datasets')
    pool=[dict(task_id=f'gsm8k/train/{i}',dataset='gsm8k',source_split='train',
        prompt=f"Q: {r['question']}\nA:",question=r['question'],
        answer=r['answer'].split('####')[-1].strip(),until=['Q:','</s>','<|im_end|>'],gen_length=256)
        for i,r in enumerate(rows) if r['question'] not in excluded]
    rng=random.Random(20261117)
    rng.shuffle(pool)
    sets['gsm8k_calibration_v2']=sets['gsm8k_calibration']+pool[:16]
    sets['gsm8k_search_v2']=sets['gsm8k_search']+pool[16:48]
    sets['gsm8k_timing']=sets['gsm8k_test'][:64]
    # MATH train is the only source of domain-specific search/calibration data.
    # Avoid identical text anywhere in its test sets and the exposed OOD pilot.
    test_questions={r['question'] for k in ('math_test','math500_test') for r in sets[k]}
    pool=[]
    seen=set(test_questions)
    for row in sets['math_train']:
        if row['question'] not in seen:
            pool.append(row)
            seen.add(row['question'])
    rng.shuffle(pool)
    sets['math_calibration']=pool[:16]
    sets['math_search']=pool[16:48]
    sets['math_dev_holdout']=pool[48:80]
    exposed={r['task_id'] for r in sets['math500_ood_pilot']}
    sets['math500_unseen']=[r for r in sets['math500_test'] if r['task_id'] not in exposed]
    sets['math500_unseen_audit']=sets['math500_unseen'][:64]
    sets['gsm8k_transfer_audit']=sets['gsm8k_test'][:64]
    for splits in [('gsm8k_calibration_v2','gsm8k_search_v2','gsm8k_dev_holdout'),
                   ('math_calibration','math_search','math_dev_holdout')]:
        selected=[r['question'] for s in splits for r in sets[s]]
        if len(selected)!=len(set(selected)):
            raise ValueError(f'Overlapping development questions: {splits}')
    a.output.mkdir(parents=True)
    hashes={}
    for name,rows in sets.items():
        path=a.output/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta={**original,'seed':20261117,'sha256':hashes,'counts':{k:len(v) for k,v in sets.items()},
          'parent_manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
          'ood':'Domain-specific GSM8K and MATH training searches; cross-domain frozen-policy audits. The 32 previously exposed MATH-500 questions are excluded from unseen audits.'}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta['counts'],indent=2))


if __name__=='__main__':
    main()
