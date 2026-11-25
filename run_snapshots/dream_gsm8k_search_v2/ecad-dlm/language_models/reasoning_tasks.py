"""Versioned reasoning tasks and official Dream-Instruct answer scoring."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
import re
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[2]
SCORING_VERSION = 'reasoning_v2_latex_gold'


def load_tasks(path):
    tasks = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({t['task_id'] for t in tasks}) != len(tasks):
        raise ValueError('Duplicate task IDs')
    required = {'task_id', 'dataset', 'prompt', 'answer', 'until', 'gen_length', 'source_split'}
    for task in tasks:
        if required - task.keys():
            raise ValueError(f'Missing task fields: {required - task.keys()}')
    return tasks


@lru_cache(None)
def math_scorer():
    path = ROOT / 'Dream/eval_instruct/lm_eval/tasks/minerva_math/utils.py'
    spec = importlib.util.spec_from_file_location('ecad_dream_math_scoring', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalize_gsm(text):
    # Exact regex normalization in Dream's gsm8k-cot task YAML.
    for pattern in [',', r'\$', r'(?s).*#### ', r'\.$']:
        text = re.sub(pattern, '', text)
    return text.lower()


def gsm_extract(response, flexible=False):
    pattern = r'(-?[$0-9.,]{2,})|(-?[0-9]+)' if flexible else r'The answer is (\-?[0-9\.\,]+).'
    matches = re.findall(pattern, response)
    if not matches:
        return '[invalid]'
    value = matches[-1] if flexible else matches[0]
    if isinstance(value, tuple):
        value = next((x for x in value if x), '[invalid]')
    return value.strip()


def score_response(task, response):
    if task['dataset'] == 'gsm8k':
        strict, flexible = gsm_extract(response), gsm_extract(response, True)
        return {'exact_match': int(normalize_gsm(strict) == normalize_gsm(task['answer'])),
                'flexible_extract': int(normalize_gsm(flexible) == normalize_gsm(task['answer'])),
                'extracted_answer': strict}
    scorer = math_scorer()
    # Preserve the upstream metrics for comparison, but its exact_match extractor
    # requires a literal Minerva few-shot phrase absent from zero-shot responses.
    legacy = scorer.process_results({'answer': scorer.normalize_final_answer(task['answer'])}, [response])
    from math_verify import parse, verify, LatexExtractionConfig
    # Dataset answers are LaTeX *without delimiters*. Parsing them as prose can
    # drop operators or fail entirely (e.g. sqrt{53}); never normalize away math.
    gold = task['answer'].strip().strip('$').strip()
    parsed_gold = parse('$' + gold + '$', extraction_config=[LatexExtractionConfig()])
    parsed_response = parse(response)
    boxed = scorer.last_boxed_only_string(response)
    boxed_match = 0
    if boxed is not None:
        extracted = scorer.normalize_final_answer(scorer.remove_boxed(boxed))
        reference = scorer.normalize_final_answer(gold)
        boxed_match = int(extracted == reference or scorer.is_equiv(extracted, reference))
    return {'exact_match': legacy['exact_match'],
            'upstream_math_verify': legacy['math_verify'],
            'math_verify': int(bool(parsed_gold) and bool(parsed_response) and verify(parsed_gold, parsed_response)),
            'boxed_match': boxed_match, 'gold_parse_ok': int(bool(parsed_gold))}
