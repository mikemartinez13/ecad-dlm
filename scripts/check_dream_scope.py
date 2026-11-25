"""Prevent old launch commands from reviving the retired MATH workflow."""
import sys
import yaml

config = yaml.safe_load(open(sys.argv[1]))
splits = [config.get(key, '') for key in ('search_split', 'calibration_split', 'timing_split')]
splits += config.get('audit_splits', ['gsm8k_dev_holdout', 'math500_ood_pilot'])
if config.get('disabled') or any(s.startswith('math') for s in splits):
    raise SystemExit('Michael retired the MATH workflow on 2026-11-17. Use a GSM8K-only configuration; historical results are preserved.')
