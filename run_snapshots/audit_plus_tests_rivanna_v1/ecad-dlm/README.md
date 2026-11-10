# ecad-dlm

Diffusion language model code, separated from the image-model code in `ecad/`.
The folder name has a hyphen, so it is not itself importable: put it on the path
and import the packages inside it.

```bash
export PYTHONPATH="$PWD/ecad-dlm:$PWD"   # from the repository root
```

| Folder | Contents |
| --- | --- |
| `language_models/` | Model loading, code tasks, EvalPlus execution, residual-cache runtime, DLM schedules |
| `dlm_kv_policy/` | DiffuCoder K/V caching: policies, backends (batch one and batched), sampler, compute accounting, and the divergence proxy (`forced_trajectory_replay.py`, `validate_proxy.py`) |
| `evolution/` | NSGA-II searches over K/V policies and residual-cache schedules; see `experiments/CONSTRAINED_SEARCH.md` |
| `experiments/` | Studies and their documents, `configs/` and `scripts/`, including the proxy recheck, cost study and EAR validity study |
| `tests/` | CPU tests. Run with `cd ecad-dlm/tests && python -m unittest discover -s . -p 'test_*.py'` |

Core packages never import from `experiments/`. Files ending in `_cs` hold CS
cluster paths; the same files without the suffix hold the Rivanna paths.
`ecad-dlm/` still imports the shared `CacheSchedule` base class and schedule types from `ecad/`.
