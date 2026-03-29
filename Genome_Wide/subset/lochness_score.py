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
input_file = "ReplogleWeissman2022_K562_gwps_processed_top100_dev.h5ad" 
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
adata.write_h5ad("ReplogleWeissman2022_K562_gwps_processed_top100_dev_lochness.h5ad")

# ==============================================================================
# 3. VISUALIZATION SUITE
# ==============================================================================
print("--- GENERATING FIGURES ---")
os.makedirs("plots_lochness", exist_ok=True)

# ------------------------------------------------------------------------------
# FIGURE 1A: UMAP - ALL TARGETS (Raw Asymmetric Scale)
# ------------------------------------------------------------------------------
# We use vcenter=0 to keep the baseline neutral, but do NOT force vmin/vmax.
# This shows the exact, raw mathematical limits of the global dataset.
fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    adata, 
    color='lochNESS', 
    cmap='coolwarm', 
    vcenter=0, 
    title="Global lochNESS (All Targets, Raw Scale)", 
    ax=ax, 
    show=False
)
plt.savefig("plots_lochness/01a_UMAP_lochNESS_All.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------------
# EXTRACT TOP 15 TARGETS
# ------------------------------------------------------------------------------
# Filter out controls to identify true top targets based on median score
target_adata = adata[adata.obs[perturb_col] != control_label]
order_all = target_adata.obs.groupby(perturb_col)['lochNESS'].median().sort_values(ascending=False).index
top_15_targets = order_all[:15].tolist()

# ------------------------------------------------------------------------------
# FIGURE 1B: UMAP - TOP 15 vs CONTROL
# ------------------------------------------------------------------------------
# Strictly isolate the top 15 targets and the baseline control cells
keep_list = top_15_targets + [control_label]
adata_top15 = adata[adata.obs[perturb_col].isin(keep_list)]

fig, ax = plt.subplots(figsize=(10, 8))
sc.pl.umap(
    adata_top15, 
    color='lochNESS', 
    cmap='coolwarm', 
    vcenter=0, 
    title="lochNESS Distribution (Top 15 Targets vs Control)", 
    ax=ax, 
    show=False
)
plt.savefig("plots_lochness/01b_UMAP_lochNESS_Top15.png", dpi=300, bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------------
# FIGURE 2: RANKED VIOLIN PLOT (TOP 15)
# ------------------------------------------------------------------------------
# Subset the target-only data to just the top 15
adata_top15_targets = target_adata[target_adata.obs[perturb_col].isin(top_15_targets)]

plt.figure(figsize=(10, 6)) # Scaled perfectly for 15 variables
sns.violinplot(
    data=adata_top15_targets.obs, 
    x=perturb_col, 
    y='lochNESS', 
    order=top_15_targets, # Uses the median-sorted order
    palette="viridis",
    inner="quartile",
    linewidth=1
)
plt.axhline(0, color='red', linestyle='--', label='Baseline (0)')
plt.title("Distribution of lochNESS Scores (Top 15 Targets)")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("lochNESS Score")
plt.legend()
plt.tight_layout()
plt.savefig("plots_lochness/02_Violin_lochNESS_Top15.png", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# FIGURE 3: MEAN EFFECT SIZE BAR CHART (TOP 15)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
# Calculate mean specifically for the top 15 and sort them by mean for the bar chart
mean_scores_top15 = adata_top15_targets.obs.groupby(perturb_col)['lochNESS'].mean().sort_values(ascending=False)

sns.barplot(x=mean_scores_top15.index, y=mean_scores_top15.values, palette="mako")
plt.axhline(0, color='red', linestyle='--')
plt.title("Mean lochNESS Score (Top 15 Targets)")
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.ylabel("Mean lochNESS Score")
plt.tight_layout()
plt.savefig("plots_lochness/03_Bar_Mean_lochNESS_Top15.png", dpi=300)
plt.close()

print("--- PIPELINE COMPLETE. PLOTS SAVED IN 'plots_lochness/' ---")
