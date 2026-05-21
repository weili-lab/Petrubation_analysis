# PS Score Scaling Algorithm Notes

This note summarizes three possible builds for scaling the exact Python PS score
workflow to large perturb-seq datasets. It is based on the current exact runner
in `src/perturb_effects/ps_score_exact.py` and the Replogle comparison work.

The current exact implementation is exact for a single-label AnnData setting
after target genes are selected. It fits a ridge model with an intercept and
perturbation indicators, then computes bounded PS scores from the fitted
perturbation beta vectors.

## Shared Problem Structure

Let:

```text
N = number of cells
P = number of perturbations
G = number of union target genes
X = cell by perturbation design matrix, with an intercept column
Y = cell by target-gene expression matrix
B = ridge beta matrix
```

The ridge fit is:

```text
B = inv(X.T @ X + lr_lambda * I) @ X.T @ Y
```

The major scale issue is not the formula itself. The issue is avoiding dense
`N x G` expression arrays and dense `N x P` score outputs when `N` is in the
millions.

For the current code, `apply_quantile_clip=True` disables the sparse closed-form
path and forces a dense selected expression matrix. For million-cell data this
is usually not acceptable. Disabling clipping is the simplest scalable path. If
clipping is required later, it should be reimplemented as a streaming or
blockwise operation.

Cholesky is not directly limited by millions of cells because it operates on
`X.T @ X`, whose size is `(P + 1) x (P + 1)`. The blocker for Cholesky is very
large `P`, not large `N`.

## Path A: Scalable Single-Label Exact PS

### Intended Use

Use this when each cell has exactly one perturbation label or is control. This
matches the current Replogle-style workflow and the current exact implementation
most closely.

### Key Simplification

For one perturbation per cell, the design matrix is intercept plus one-hot
perturbation labels. We do not need to materialize `X`.

The ridge sufficient statistics can be accumulated from sparse expression
chunks:

```text
total cell count
cell count per perturbation
sum expression over all cells
sum expression per perturbation
```

These statistics are enough to form `X.T @ X` and `X.T @ Y`.

### Scoring

After beta is fit, the own score for a perturbed cell with perturbation `j` can
be computed independently:

```text
raw_score = ((y_i - beta_0) dot beta_j - score_lambda) / (beta_j dot beta_j)
score = clip(raw_score, 0, scale_factor) / scale_factor
```

If per-perturbation max scaling is enabled, scoring needs two passes or a
temporary per-cell unscaled score file:

```text
pass 1: stream cells and collect max score per perturbation
pass 2: stream cells and write normalized own scores
```

### Required Scale Choices

Use sparse/chunked expression reads.

Do not build dense `Y`.

Do not build dense `X`.

Do not write full `N x P` wide scores unless explicitly requested.

Prefer own-score output:

```text
cell_id, perturbation_id, score
```

Disable quantile clipping for the first scalable implementation.

### Cholesky / Solve Choice

Generic Cholesky can work for moderate `P`, but the single-label design has
special structure. A structured solve should be faster and use less memory than
forming a fully dense generic system.

### Main Bottlenecks

Target-gene selection can dominate runtime, especially `scanpy_de` target genes.

Sparse h5ad I/O can dominate runtime once the math is optimized.

Per-perturbation max scaling needs an extra scoring pass.

### Expected Memory Shape

Memory should scale mainly with:

```text
P x G beta/statistics
chunk_cells x chunk_genes working block
```

It should not scale with full:

```text
N x G
N x P
```

## Path B: General Multi-Perturbation Exact PS

### Intended Use

Use this when cells may contain multiple perturbations or guides. This is a more
general model than the current single-label exact path.

### Design Matrix

Each cell has a multi-hot perturbation vector:

```text
cell i: perturbations A, B, C
X_i = [intercept, 1_A, 1_B, 1_C, ...]
```

This changes both the ridge fit and the scoring step.

### Ridge Statistics

`X.T @ X` becomes a perturbation co-occurrence matrix:

```text
X.T @ X[j, k] = number of cells containing both perturbation j and perturbation k
```

`X.T @ Y` can still be accumulated from sparse expression chunks:

```text
for each active perturbation in a cell, add that cell's expression to that perturbation row
```

### Memory Blockers

Dense `X.T @ X` can become large when `P` is large:

```text
P = 4,000:  about 128 MB for one float64 dense matrix
P = 10,000: about 800 MB for one float64 dense matrix
P = 50,000: about 20 GB for one float64 dense matrix
```

The beta matrix can also become large:

```text
beta shape = (P + 1) x G
```

Full `N x P` score output is not acceptable at scale. Output should contain only
active perturbation scores:

```text
cell_id, perturbation_id, score
```

Dense `N x G` expression remains unacceptable, so quantile clipping has the same
problem as in Path A unless rewritten in streaming form.

### Time Blockers

Constructing `X.T @ X` costs roughly:

```text
sum over cells of active_perturbation_count_i squared
```

If each cell has 2 to 5 perturbations, this can be manageable. If many cells
have 10 or more perturbations, co-occurrence construction can become expensive.

Dense Cholesky has about cubic cost in `P`:

```text
O(P^3)
```

For large `P`, a sparse or iterative ridge solve may be required.

### Scoring

For a multi-perturbation cell, scoring is no longer one scalar projection. The
cell needs a small bounded regression over its active perturbations:

```text
y_i - beta_0 ~= s_A beta_A + s_B beta_B + s_C beta_C
0 <= s_j <= scale_factor
```

If each cell has few perturbations, this is a small per-cell problem. If each
cell has many perturbations, millions of bounded least-squares solves become a
major time blocker.

Using `scipy.optimize.lsq_linear` one cell at a time would likely be too slow
for million-cell datasets. This scoring step needs batching, vectorization, or a
custom small-solver implementation.

### Required Inputs

This path needs an explicit cell-to-perturbation assignment table, not just one
label per cell.

Useful metadata to inspect before implementation:

```text
number of perturbations per cell distribution
number of unique perturbations
number of co-occurring perturbation pairs
number of target genes
desired output shape
```

## Path C: GPU-Based Minibatch Build

### Intended Use

Use this when CPU streaming is correct but scoring or dense block math becomes
the bottleneck, or when multi-perturbation scoring needs batched small solves.

This path should not start as stochastic gradient descent unless we intentionally
change the objective. The better first GPU design is exact minibatch
accumulation and exact minibatch scoring.

### Exact Minibatch Ridge Accumulation

Read cells in chunks and accumulate:

```text
X.T @ X += X_batch.T @ X_batch
X.T @ Y += X_batch.T @ Y_batch
```

This gives the same ridge solution as the full closed-form calculation, without
holding all cells in memory.

For Path A, this is mostly group sums and may be faster on CPU than GPU.

For Path B, GPU can help if the multi-hot design and expression blocks are
already represented efficiently.

### GPU Scoring

GPU is more promising for scoring than for h5ad I/O.

For single-label scoring:

```text
score_batch = projection of expression batch onto matched beta rows
```

For multi-label scoring:

```text
solve many small bounded regressions in batches
```

The multi-label case is the main reason to consider PyTorch seriously.

### Main Blockers

Sparse h5ad I/O may dominate and keep the GPU underused.

Moving large expression chunks from CPU to GPU can erase gains.

GPU memory still cannot hold full `N x G` or `N x P` arrays.

Batched bounded least-squares needs a clear solver strategy. Options include
projected gradient, coordinate descent, active-set methods, or a small custom
solver for common active-set sizes.

Numerical agreement needs validation against the CPU closed-form path on small
panels.

### Practical GPU Strategy

Start with CPU streaming for target-gene selection and ridge statistics.

Use GPU only for the scoring pass if profiling shows scoring dominates.

For multi-label cells, group cells by active perturbation count and run batched
solvers per group.

Write compact long output instead of wide score matrices.

## Recommended Development Order

Build Path A first. It gives the clearest route to million-cell single-label PS
scores and directly addresses the current scaling bottlenecks.

Profile Path A on a full-size dataset with clipping disabled and own-score-only
output.

Before building Path B, inspect the perturbations-per-cell distribution in the
target multi-perturbation datasets. This determines whether the general case is
easy or expensive.

Add Path C only after CPU streaming identifies a real compute bottleneck. GPU is
most likely useful for multi-label batched scoring, not for basic single-label
ridge statistics.
