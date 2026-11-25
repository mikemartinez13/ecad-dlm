"""Freeze 96-question domain search sets; retain independent calibration/audits."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from datasets import load_dataset


def read(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def balanced_sample(rows, count, key, seed=20261118):
    rng=random.Random(seed)
    buckets=defaultdict(list)
    for row in rows:
        buckets[key(row)].append(row)
    # Interleave short, medium and long questions within each subject/difficulty
    # cell rather than repeatedly drawing only its most common question length.
    for k, group in buckets.items():
        group.sort(key=lambda r:len(r['question']))
        bins=[group[i*len(group)//3:(i+1)*len(group)//3] for i in range(3)]
        for b in bins:
            rng.shuffle(b)
        ordered=[]
        while any(bins):
            for b in bins:
                if b:
                    ordered.append(b.pop())
        buckets[k]=ordered
    keys=sorted(buckets)
    rng.shuffle(keys)
    selected=[]
    while len(selected)<count:
        before=len(selected)
        for k in keys:
            if buckets[k] and len(selected)<count:
                selected.append(buckets[k].pop(0))
        if len(selected)==before:
            raise ValueError('Insufficient distinct questions')
    return selected


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('/scratch/gqs7zr/dream_reasoning/data_v2'))
    p.add_argument('--output',type=Path,default=Path('/scratch/gqs7zr/dream_reasoning/data_v3_96'))
    a=p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite frozen manifests')
    original=json.loads((a.source/'manifest.json').read_text())
    sets={p.stem:read(p) for p in a.source.glob('*.jsonl')}
    excluded={r['question'] for k in ('gsm8k_calibration_v2','gsm8k_dev_holdout','gsm8k_test') for r in sets[k]}
    rows=load_dataset('openai/gsm8k','main',split='train',
        revision=original['dataset_revisions']['openai/gsm8k'],
        cache_dir='/scratch/gqs7zr/cache/huggingface/datasets')
    pool=[]
    for i,r in enumerate(rows):
        if r['question'] in excluded:
            continue
        excluded.add(r['question'])
        pool.append(dict(task_id=f'gsm8k/train/{i}',dataset='gsm8k',source_split='train',
            prompt=f"Q: {r['question']}\nA:",question=r['question'],
            answer=r['answer'].split('####')[-1].strip(),until=['Q:','</s>','<|im_end|>'],gen_length=256))
    # Three approximate question-length strata with equal representation.
    pool.sort(key=lambda r:len(r['question']))
    length_bin={r['task_id']:min(2,3*i//len(pool)) for i,r in enumerate(pool)}
    sets['gsm8k_search_96']=balanced_sample(pool,96,lambda r:str(length_bin[r['task_id']]))
    excluded={r['question'] for k in ('math_calibration','math_dev_holdout','math_test','math500_test') for r in sets[k]}
    pool=[]
    for r in sets['math_train']:
        if r['question'] not in excluded:
            pool.append(r); excluded.add(r['question'])
    sets['math_search_96']=balanced_sample(pool,96,lambda r:f"{r['subject']}|{r['level']}")
    sets['math_timing']=sets['math500_unseen'][:64]
    for search,calibration,holdout in [('gsm8k_search_96','gsm8k_calibration_v2','gsm8k_dev_holdout'),
                                      ('math_search_96','math_calibration','math_dev_holdout')]:
        rows=sets[search]+sets[calibration]+sets[holdout]
        if len({r['question'] for r in rows})!=len(rows):
            raise ValueError(f'Overlapping development questions: {search}')
    a.output.mkdir(parents=True)
    hashes={}
    for name,rows in sets.items():
        path=a.output/f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    meta={**original,'seed':20261118,'sha256':hashes,'counts':{k:len(v) for k,v in sets.items()},
        'parent_manifest_sha256':hashlib.sha256((a.source/'manifest.json').read_bytes()).hexdigest(),
        'search_selection':'GSM8K question-length strata; MATH balanced subject/level cells with interleaved question-length thirds',
        'math_search_strata':dict(Counter(f"{r['subject']}|{r['level']}" for r in sets['math_search_96']))}
    (a.output/'manifest.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps({'counts':meta['counts'],'math_search_strata':meta['math_search_strata']},indent=2))


if __name__=='__main__':
    main()
