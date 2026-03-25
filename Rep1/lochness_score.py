import os
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors

# ==============================================================================
# 1. THE LOCHNESS ENGINE (Single-Pool Exact)
# ==============================================================================
def calculate_lochness_single_pool(adata, perturb_col='perturbation', control_label='control'):
    """
    Calculates lochNESS for a single-pool experiment, 
    actively removing the self-match bias.
    """
    k_adj = int(round(0.5 * np.sqrt(adata.n_obs)))
    k_query = k_adj + 1 
    
    if 'X_pca_l2' not in adata.obsm:
        X_pca = adata.obsm['X_pca']
        adata.obsm['X_pca_l2'] = X_pca / np.linalg.norm(X_pca, axis=1, keepdims=True)
    
    space = adata.obsm['X_pca_l2']
    
    nn = NearestNeighbors(n_neighbors=k_query, algorithm='kd_tree', n_jobs=-1).fit(space)
    distances, indices = nn.kneighbors(space)
    
    # Remove self-match
    true_neighbor_indices = indices[:, 1:]
    
    labels = adata.obs[perturb_col].values
    knn_labels = labels[true_neighbor_indices]
    
    is_perturbed = (knn_labels != control_label)
    obs_mutant_counts = np.sum(is_perturbed, axis=1)
    
    bg_freq = np.sum(adata.obs[perturb_col] != control_label) / adata.n_obs
    
    lochness_scores = (obs_mutant_counts / k_adj) / bg_freq - 1
    
    return pd.Series(lochness_scores, index=adata.obs_names)

# ==============================================================================
# 2. EXECUTION
# ==============================================================================
print("--- LOADING DATA ---")
# Use your subset file if you generated one, otherwise the full processed file
input_file = "ReplogleWeissman2022_rpe1_processed.h5ad" 
adata = sc.read_h5ad(input_file)

print("--- CALCULATING LOCHNESS ---")
perturb_col = 'perturbation'
control_label = 'control'

adata.obs['lochNESS'] = calculate_lochness_single_pool(
    adata, 
    perturb_col=perturb_col, 
    control_label=control_label
)

print("--- SAVING UPDATED H5AD ---")
adata.write_h5ad("ReplogleWeissman2022_rpe1_processed_lochness.h5ad")

# ==============================================================================
# 3. VISUALIZATION SUITE
# ==============================================================================
print("--- GENERATING FIGURES ---")
os.makedirs("plots_lochness", exist_ok=True)

# ==============================================================================
# UPDATED FIGURE 1: SYMMETRICAL LOCHNESS UMAP
# ==============================================================================
# 1. Find the absolute maximum deviation from 0 to force symmetry
max_abs_val = np.max(np.abs(adata.obs['lochNESS']))

# 2. Plot with strict vmin and vmax boundaries
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    adata, 
    color='lochNESS', 
    cmap='coolwarm', 
    vcenter=0, 
    vmin=-max_abs_val,  # Forces symmetric bottom
    vmax=max_abs_val,   # Forces symmetric top
    title=f"Global lochNESS (Symmetric Scale: ±{max_abs_val:.3f})", 
    ax=ax, 
    show=False
)
plt.savefig("plots_lochness/01_UMAP_lochNESS_symmetric.png", dpi=300, bbox_inches='tight')
plt.close()

# Filter out controls to analyze the targets specifically
target_adata = adata[adata.obs[perturb_col] != control_label]

# Sort targets by their median effect size for clean plotting
order = target_adata.obs.groupby(perturb_col)['lochNESS'].median().sort_values(ascending=False).index

# Figure 2: Ranked Violin Plots
# Shows the variance and spread of the perturbation effect for each target
plt.figure(figsize=(14, 6))
sns.violinplot(
    data=target_adata.obs, 
    x=perturb_col, 
    y='lochNESS', 
    order=order,
    palette="viridis",
    inner="quartile",
    linewidth=1
)
plt.axhline(0, color='red', linestyle='--', label='Baseline (0)')
plt.title("Distribution of lochNESS Scores per CRISPR Target")
# Only rotate labels if there are a lot of them; for 200, 90 degrees might be needed
plt.xticks(rotation=90, ha='center', fontsize=6) 
plt.ylabel("lochNESS Score")
plt.legend()
plt.tight_layout()
plt.savefig("plots_lochness/02_Violin_lochNESS_per_Target.png", dpi=300)
plt.close()

# Figure 3: Mean Effect Size Bar Chart
# A quick ranking of which targets drive the strongest overall phenotype
plt.figure(figsize=(12, 6))
mean_scores = target_adata.obs.groupby(perturb_col)['lochNESS'].mean().sort_values(ascending=False)
sns.barplot(x=mean_scores.index, y=mean_scores.values, palette="mako")
plt.axhline(0, color='red', linestyle='--')
plt.title("Mean lochNESS Score by CRISPR Target")
plt.xticks(rotation=90, ha='center', fontsize=6)
plt.ylabel("Mean lochNESS Score")
plt.tight_layout()
plt.savefig("plots_lochness/03_Bar_Mean_lochNESS.png", dpi=300)
plt.close()

print("--- PIPELINE COMPLETE. PLOTS SAVED IN 'plots_lochness/' ---")
