import os
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
loch_h5ad_path = "ReplogleWeissman2022_K562_essential_processed_top100_dev_lochness.h5ad"
ps_h5ad_path   = "ReplogleWeissman2022_K562_essential_processed_top100_dev_PS.h5ad"
output_h5ad    = "ReplogleWeissman2022_K562_essential_top100_INTEGRATED.h5ad"

perturb_col   = 'perturbation' 
negative_ctrl = 'control'

os.makedirs("plots_integrated", exist_ok=True)
os.makedirs("tables_batch", exist_ok=True)

# ==============================================================================
# 2. LOAD & VERIFY ALIGNMENT
# ==============================================================================
print(f"--- LOADING DATA ---")
adata_loch = sc.read_h5ad(loch_h5ad_path)
adata_ps   = sc.read_h5ad(ps_h5ad_path)

if not adata_loch.obs_names.equals(adata_ps.obs_names):
    print("STDOUT: Cell barcodes do not strictly align. Forcing alignment...")
    adata_ps = adata_ps[adata_loch.obs_names].copy()

# ==============================================================================
# 3. COLLAPSE PS SCORE & INTEGRATE
# ==============================================================================
print("--- COLLAPSING PS SCORES TO 1D ARRAY ---")

ps_1d_array = pd.Series(0.0, index=adata_loch.obs_names, dtype=float)
unique_targets = [t for t in adata_loch.obs[perturb_col].unique() if t != negative_ctrl]

for target in unique_targets:
    score_col = f"{target}_eff"
    
    if score_col in adata_ps.obs.columns:
        mask = adata_ps.obs[perturb_col] == target
        ps_1d_array[mask] = adata_ps.obs.loc[mask, score_col]
    else:
        print(f"WARNING: '{score_col}' not found in PS h5ad. Skipping {target}.")

adata_loch.obs['PS_score_collapsed'] = ps_1d_array
print("STDOUT: PS Scores successfully collapsed and merged.")

# ==============================================================================
# 4. SAVE UNIFIED H5AD
# ==============================================================================
print(f"--- SAVING INTEGRATED DATA TO {output_h5ad} ---")
adata_loch.write_h5ad(output_h5ad)

# ==============================================================================
# 5. DUAL-METRIC VISUALIZATION SUITE
# ==============================================================================
print("--- GENERATING DUAL-METRIC PLOTS ---")

target_adata = adata_loch[adata_loch.obs[perturb_col] != negative_ctrl].copy()
median_ps = target_adata.obs['PS_score_collapsed'].median()

# ------------------------------------------------------------------------------
# PLOT 1: Cell-Level 2D Density (The Quadrant View)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 8))
sns.histplot(
    data=target_adata.obs, 
    x='lochNESS', 
    y='PS_score_collapsed', 
    bins=75, 
    pmax=0.9, 
    cmap='mako', 
    cbar=True,
    cbar_kws={'label': 'Cell Count Density'}
)

plt.axvline(0, color='red', linestyle='--', linewidth=1.5, label='lochNESS Baseline (0)')
plt.axhline(median_ps, color='orange', linestyle='--', linewidth=1.5, label=f'Median PS ({median_ps:.2f})')

y_max = target_adata.obs['PS_score_collapsed'].max()
plt.text(0.02, y_max * 0.95, "Holy Grail\n(High Shift, High Consistency)", color='white', weight='bold')
plt.text(-0.15, y_max * 0.95, "Noisy/Dying\n(High Shift, Low Consistency)", color='black', weight='bold')

plt.title("Cell-Level Phenotypic Landscape: lochNESS vs PS Score")
plt.xlabel("lochNESS (Phenotypic Consistency)")
plt.ylabel("Collapsed PS Score (Phenotypic Magnitude)")
plt.legend(loc='lower right')
plt.tight_layout()

# Save as PNG and PDF
plt.savefig("plots_integrated/01_Cell_Density_lochNESS_vs_PS.png", dpi=300, bbox_inches='tight')
plt.savefig("plots_integrated/01_Cell_Density_lochNESS_vs_PS.pdf", format='pdf', bbox_inches='tight')
plt.close()

# ------------------------------------------------------------------------------
# PLOT 2: Target-Level Aggregated Scatter (Color-Coded & Filtered)
# ------------------------------------------------------------------------------
target_stats = target_adata.obs.groupby(perturb_col)[['lochNESS', 'PS_score_collapsed']].median()

def assign_quadrant_color(row):
    if row['lochNESS'] > 0 and row['PS_score_collapsed'] > median_ps:
        return 'red'
    elif row['lochNESS'] <= 0 and row['PS_score_collapsed'] <= median_ps:
        return 'skyblue'
    else:
        return 'grey'

target_stats['Quadrant_Color'] = target_stats.apply(assign_quadrant_color, axis=1)

plt.figure(figsize=(12, 8))
plt.scatter(
    target_stats['lochNESS'], 
    target_stats['PS_score_collapsed'], 
    c=target_stats['Quadrant_Color'], 
    alpha=0.8, 
    s=100,
    edgecolors='white', 
    linewidth=0.5
)

plt.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.axhline(median_ps, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Extract hits, filter boring genes, calculate Hit Power
red_quadrant_hits = target_stats[target_stats['Quadrant_Color'] == 'red'].copy()
mask_boring = red_quadrant_hits.index.str.contains('^RP[LS]|^PSM', regex=True)
interesting_hits = red_quadrant_hits[~mask_boring].copy()

interesting_hits['Hit_Power'] = interesting_hits['lochNESS'] * (interesting_hits['PS_score_collapsed'] - median_ps)
top_5_genes = interesting_hits.sort_values(by='Hit_Power', ascending=False).head(5)

# Print the findings to the terminal
print("\n--- TOP 5 BIOLOGICAL HITS (Filtered) ---")
print(top_5_genes[['lochNESS', 'PS_score_collapsed', 'Hit_Power']])
print("----------------------------------------\n")

for gene in top_5_genes.index:
    plt.text(
        target_stats.loc[gene, 'lochNESS'] + 0.002, 
        target_stats.loc[gene, 'PS_score_collapsed'], 
        gene, 
        fontsize=10, 
        weight='bold',
        color='darkred'
    )

red_patch = mpatches.Patch(color='red', label='Target Hits (High PS, High lochNESS)')
blue_patch = mpatches.Patch(color='skyblue', label='Escapers (Low PS, Low lochNESS)')
grey_patch = mpatches.Patch(color='grey', label='Discordant / Noisy')
plt.legend(handles=[red_patch, blue_patch, grey_patch], loc='lower left')

plt.title("Target-Level Evaluation: Median lochNESS vs Median PS Score")
plt.xlabel("Median lochNESS (Consistency)")
plt.ylabel("Median PS Score (Magnitude)")
plt.tight_layout()

# Save as PNG and PDF
plt.savefig("plots_integrated/02_Target_Scatter_lochNESS_vs_PS_Colored_Filtered.png", dpi=300, bbox_inches='tight')
plt.savefig("plots_integrated/02_Target_Scatter_lochNESS_vs_PS_Colored_Filtered.pdf", format='pdf', bbox_inches='tight')
plt.close()

# ==============================================================================
# 6. EXPORT TARGET METRICS TO CSV
# ==============================================================================
print("--- EXPORTING TARGET METRICS TO CSV ---")
target_stats['Hit_Power'] = target_stats['lochNESS'] * (target_stats['PS_score_collapsed'] - median_ps)
target_stats_sorted = target_stats.sort_values(by=['Quadrant_Color', 'Hit_Power'], ascending=[True, False])
csv_out_path = "tables_batch/Integrated_Target_Metrics.csv"
target_stats_sorted.to_csv(csv_out_path)

print(f"STDOUT: Target metrics successfully saved to {csv_out_path}")
print("--- PIPELINE COMPLETE. READY FOR DOWNSTREAM MODELING. ---")
