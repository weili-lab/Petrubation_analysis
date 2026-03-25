import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from multiprocessing import Pool

# ==============================================================================
# HELPER FUNCTION FOR PARALLEL LEIDEN
# ==============================================================================
def run_leiden_parallel(args_tuple):
    """
    Independent worker function. 
    Uses igraph flavor to match future Scanpy defaults and increase speed.
    """
    adata_subset, res = args_tuple
    
    # Surgical check: Ensure neighbors exist in this process's memory
    if 'neighbors' not in adata_subset.uns:
        return res, None
        
    sc.tl.leiden(
        adata_subset, 
        resolution=res, 
        key_added=f'leiden_res_{res}',
        flavor="igraph", 
        n_iterations=2,
        directed=False
    )
    
    return res, adata_subset.obs[f'leiden_res_{res}']

# ==============================================================================
# 0. ARGUMENT PARSER
# ==============================================================================
parser = argparse.ArgumentParser(description='Parallel Surgical scRNA-seq Preprocessing')
parser.add_argument('--input', type=str, required=True, help='Path to input .h5ad')
parser.add_argument('--output', type=str, default='processed_data.h5ad', help='Path to output .h5ad')
parser.add_argument('--min_genes', type=int, default=200, help='Min genes per cell')
parser.add_argument('--max_genes', type=int, default=10000, help='Max genes per cell')
parser.add_argument('--max_mt', type=float, default=5.0, help='Max MT%')
parser.add_argument('--threads', type=int, default=4, help='Parallel threads for clustering')
args = parser.parse_args()

os.makedirs("plots_qc", exist_ok=True)

# ==============================================================================
# 1. LOAD & PRESERVE
# ==============================================================================
print(f"--- LOADING: {args.input} ---")
adata = sc.read_h5ad(args.input)
adata.layers["raw_counts"] = adata.X.copy()

# ==============================================================================
# 2. QC & FILTERING (SCATTER DIAGNOSTICS)
# ==============================================================================
print("\n--- CALCULATING QC & GENERATING SCATTERS ---")
adata.var['mt'] = adata.var_names.str.startswith('MT-') 
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], inplace=True)

# Pre-filter Scatters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=ax1, show=False)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=ax2, show=False)
plt.tight_layout()
plt.savefig("plots_qc/01_FeatureScatter_PRE.png", dpi=300)
plt.close()

# Surgical Filter
initial_cells = adata.n_obs
adata = adata[(adata.obs.n_genes_by_counts >= args.min_genes) & 
              (adata.obs.n_genes_by_counts <= args.max_genes) & 
              (adata.obs.pct_counts_mt < args.max_mt)].copy()

# Post-filter Scatters
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
sc.pl.scatter(adata, x='total_counts', y='pct_counts_mt', ax=ax1, show=False)
sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', ax=ax2, show=False)
plt.tight_layout()
plt.savefig("plots_qc/02_FeatureScatter_POST.png", dpi=300)
plt.close()

print(f"STDOUT: Filtered {initial_cells - adata.n_obs} cells. {adata.n_obs} remaining.")

# ==============================================================================
# 3. NORMALIZATION & PCA/NEIGHBORS
# ==============================================================================
print("\n--- NORMALIZING & COMPUTING GRAPH ---")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)

# Flag 3000 HVGs (subset=False ensures we keep all genes for PS analysis)
sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)
adata.raw = adata

# CRITICAL FIX: Neighbors must be computed BEFORE the Pool starts
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)

# ==============================================================================
# 4. MULTIPROCESSING LEIDEN SWEEP
# ==============================================================================
resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
print(f"\n--- STARTING PARALLEL LEIDEN SWEEP ({args.threads} threads) ---")

tasks = [(adata, res) for res in resolutions]

with Pool(processes=args.threads) as pool:
    results = pool.map(run_leiden_parallel, tasks)

# Join results back
for res, cluster_series in results:
    if cluster_series is not None:
        adata.obs[f'leiden_res_{res}'] = cluster_series
        print(f"STDOUT: Joined Resolution {res}")

# ==============================================================================
# 5. VISUALIZATION & EXPORT
# ==============================================================================
print("\n--- GENERATING 2x5 UMAP GRID ---")
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
axes = axes.flatten()
for i, res in enumerate(resolutions):
    res_key = f'leiden_res_{res}'
    if res_key in adata.obs.columns:
        sc.pl.umap(adata, color=res_key, ax=axes[i], show=False, title=f"Res {res}")
plt.tight_layout()
fig.savefig("plots_qc/03_UMAP_All_Resolutions_Grid.pdf", format='pdf')
fig.savefig("plots_qc/03_UMAP_All_Resolutions_Grid.png", dpi=300)
plt.close()

print(f"\n--- SAVING: {args.output} ---")
adata.write_h5ad(args.output)
print("--- PIPELINE COMPLETE ---")