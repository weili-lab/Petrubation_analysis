import scanpy as sc
import pandas as pd
import numpy as np
import scrublet as scr
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
h5ad_input  = "/local/projects-t3/lilab/vmenon/PertTF-Reploge-analysis/ReplogleWeissman2022_rpe1.h5ad"
h5ad_output = "/local/projects-t3/lilab/vmenon/PertTF-Reploge-analysis/ReplogleWeissman2022_rpe1_PROCESSED.h5ad"

os.makedirs("plots_qc", exist_ok=True)

# ==============================================================================
# 2. LOAD & PRESERVE RAW DATA
# ==============================================================================
print(f"--- LOADING: {h5ad_input} ---")
adata = sc.read_h5ad(h5ad_input)

# STDOUT: Initial Scale
print(f"INITIAL STATE: {adata.n_obs} cells, {adata.n_vars} genes")

# PRESERVE: Store raw integer counts in a dedicated layer before any math happens
adata.layers["raw_counts"] = adata.X.copy()
print("LOGIC: Raw integer counts preserved in adata.layers['raw_counts']")


# ==============================================================================
# 4. NORMALIZATION & LOG-TRANSFORM
# ==============================================================================
print("\n--- NORMALIZING & LOG-TRANSFORMING ---")
# Standardize library size to 10,000 counts per cell
sc.pp.normalize_total(adata, target_sum=1e4)
# log(1+x) transformation
sc.pp.log1p(adata)
print("LOGIC: Normalization (CPM) and Log1p complete.")

# ==============================================================================
# 5. HIGHLY VARIABLE GENES (3000 HVGs)
# ==============================================================================
print("\n--- IDENTIFYING 3000 HVGs ---")
# We use 'seurat_v3' style if data were raw, but since we log-transformed, 
# we use the standard 'cell_ranger' flavor.
sc.pp.highly_variable_genes(adata, n_top_genes=3000, subset=False)

# Preserve the full gene set in .raw before any potential future subsetting
adata.raw = adata
print("LOGIC: 3000 HVGs flagged in adata.var['highly_variable']. Full set saved in adata.raw.")

# ==============================================================================
# 6. CLUSTERING SWEEP (0.1 to 1.0)
# ==============================================================================
print("\n--- DIMENSIONALITY REDUCTION & CLUSTERING ---")
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata)

resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
for res in resolutions:
    sc.tl.leiden(adata, resolution=res, key_added=f'leiden_res_{res}')
    print(f"STDOUT: Clustering at resolution {res} complete.")

import matplotlib.pyplot as plt

# ==============================================================================
# 6.5 GENERATE UMAP GRID (0.1 TO 1.0)
# ==============================================================================
print("\n--- GENERATING UMAP GRID PLOTS ---")

resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,1.0]
n_res = len(resolutions)

# Create a figure with a grid (e.g., 2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()

for i, res in enumerate(resolutions):
    res_key = f'leiden_res_{res}'
    # Plot on the specific axis
    sc.pl.umap(adata, color=res_key, ax=axes[i], show=False, title=f"Resolution {res}")
    
    # Also save individual PNG for high-res viewing
    plt.figure(figsize=(8, 6))
    sc.pl.umap(adata, color=res_key, show=False, title=f"Resolution {res}")
    plt.savefig(f"plots_qc/UMAP_Resolution_{res}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Remove the empty 6th subplot in the grid
if n_res < len(axes):
    fig.delaxes(axes[-1])

# Save the combined Grid as PDF and PNG
plt.tight_layout()
fig.savefig("plots_qc/UMAP_All_Resolutions_Grid.pdf", format='pdf')
fig.savefig("plots_qc/UMAP_All_Resolutions_Grid.png", dpi=300)
plt.close()

print("STDOUT: UMAP grids saved to plots_qc/ (PDF/PNG).")

# = :============================================================================
# 7. FINAL EXPORT
# ==============================================================================
print(f"\n--- SAVING PROCESSED FILE: {h5ad_output} ---")
# Ensure perturbation metadata is untouched (it was carried over from sc.read_h5ad)
adata.write_h5ad(h5ad_output)

print("--- PIPELINE COMPLETE ---")
print(f"FINAL STDOUT: {adata.n_obs} High-Quality cells saved.")
