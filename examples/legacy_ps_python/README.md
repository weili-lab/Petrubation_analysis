# Legacy PS_python Pipeline

This directory contains the original PS_python demo pipeline that existed before importing the maintained exact-fast scorer.

```text
legacy_ps_python/
├── PertPS.py
├── pertps_project/
└── demo/
```

Run it from this directory so the relative `./demo/...` paths in `PertPS.py` still resolve:

```bash
pip install -e pertps_project
python PertPS.py
```

The maintained package-level scorer now lives at repository root under `src/pertps/` and is exposed as:

```python
from pertps import run_ps_score_exact_fast
```
