# PS_python

`PS_python` contains Python tools for perturbation-response score analysis in single-cell perturbation screens.

The current maintained scorer is the streamed `exact_fast` implementation, exposed as the `pertps` Python package. It calculates perturbation scores from `.h5ad` perturb-seq data while avoiding full dense score-matrix materialization.

The original PS_python demo pipeline is retained under `examples/legacy_ps_python/` for continuity with prior usage and demo figures.

## Repository Layout

```text
PS_python/
├── pyproject.toml
├── README.md
├── src/pertps/                     # maintained exact-fast PS scorer
├── tests/                          # exact-fast unit tests
└── examples/legacy_ps_python/      # original PS_python demo pipeline
    ├── PertPS.py
    ├── pertps_project/
    └── demo/
```

## What The Exact-Fast PS Score Does

For each perturbation, the method identifies target genes, fits perturbation effect vectors from normalized expression, and scores each perturbed cell against the effect vector for its observed perturbation. Scores are bounded between `0` and `scale_factor`, then scaled to `0-1` by default.

In the default `target_mode="union_deg"`, target genes are selected per perturbation by comparing perturbed cells against control cells with streamed Welch t-statistics. Genes can first be filtered by absolute log2 fold change with `logfc_threshold`, then ranked by absolute Welch t-score.

After target genes are selected, the method solves two main optimization problems:

- Ridge regression estimates each perturbation effect vector, called `beta`, on the union target gene set.
- In multilabel mode, bounded quadratic optimization assigns per-perturbation scores for cells with multiple active perturbations.

The implementation is designed for large `.h5ad` files. It streams expression chunks from disk, preserves sparse structure where possible, and writes a long CSV table of per-cell scores.

## Input Expectations

- Input is an `.h5ad` file or an in-memory `AnnData` object.
- Expression should be raw/count-like, nonnegative values in `adata.X` or a selected `adata.layers[...]` layer.
- The method internally applies library-size normalization to `target_sum=10000` followed by `log1p`.
- If `adata.X` is already log-normalized, put raw counts in a layer and pass that layer name.
- Multilabel perturbations are represented with `+`, for example `GENE1+GENE2`.

## Install Locally

```bash
pip install -e .
```

## CLI Usage

Single-label perturbations, using counts in `adata.X`:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode single \
  --perturb-column perturbation \
  --ctrl-name control \
  --target-mode union_deg \
  --target-gene-max 500 \
  --logfc-threshold 0.1 \
  --clip-quantile 0.95 \
  --chunk-size 8192 \
  --progress
```

If raw counts are in a layer:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode single \
  --perturb-column perturbation \
  --ctrl-name control \
  --layer counts
```

Multilabel perturbations:

```bash
ps_score_exact_fast \
  --dataset-path input.h5ad \
  --output-dir ps_out \
  --mode multilabel \
  --perturb-column perturbation \
  --ctrl-name control
```

Outputs:

```text
ps_out/ps-score-exact-fast.csv
ps_out/ps-score-exact-fast-manifest.json
```

## Python API Usage

```python
from pertps import run_ps_score_exact_fast

manifest = run_ps_score_exact_fast(
    "input.h5ad",
    output_dir="ps_out",
    mode="single",
    perturb_column="perturbation",
    ctrl_name="control",
    show_progress=True,
)
```

## Legacy PS_python Pipeline

The original pipeline analyzes a 10x Genomics CRISPR perturbation screen targeting 50 transcription factors. It maps barcodes to perturbation identities, computes per-gene PS scores, trains an LDA/UMAP embedding, and generates diagnostic plots.

To run the legacy workflow, use:

```bash
cd examples/legacy_ps_python
pip install -e pertps_project
python PertPS.py
```

## Citation

Song B, Liu D, Dai W, McMyn NF, Wang Q, Yang D, Krejci A, Vasilyev A, Untermoser N, Loregger A, Song D, Williams B, Rosen B, Cheng X, Chao L, Kale HT, Zhang H, Diao Y, Buerckstuemmer T, Siliciano JD, Li JJ, Siliciano RF, Huangfu D, Li W. Decoding heterogeneous single-cell perturbation responses. Nat Cell Biol. 2025 Mar;27(3):493-504. doi: 10.1038/s41556-025-01626-9.

## License

This repository is part of the weili-lab research codebase. Please contact the lab for usage and citation information.
