import os
import scanpy as sc
import pandas as pd
import numpy as np
from pertps import PerturbAnalyzer, plot_ps_on_lda, plot_global_summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# 1. USER CONFIGURATION
# ==============================================================================
h5ad_filepath    = "ReplogleWeissman2022_K562_essential_processed_top100_dev.h5ad"
output_h5ad_path = "ReplogleWeissman2022_K562_essential_processed_top100_dev_PS.h5ad"
barcode_filepath = "barcode10X_merge.txt"

# BLUNT TRUTH: Based on your log, h5ad uses 'control'. We force this globally.
negative_ctrl    = "control" 

# Create output directories
for folder in ["plots_batch", "tables_batch", "plots_fixed_lda", "plots_scatter_validation"]:
    os.makedirs(folder, exist_ok=True)

# ==============================================================================
# 2. LOAD DATA & DYNAMIC PREP (ALIGNED VERSION)
# ==============================================================================
print("Loading AnnData...")
adata = sc.read_h5ad(h5ad_filepath)

# --- TRUTH CHECK: Use 'perturbation' from h5ad metadata ---
if 'perturbation' not in adata.obs.columns:
    print("CRITICAL: 'perturbation' column not found in h5ad. Check your metadata names.")
    print(f"Available columns: {list(adata.obs.columns)}")
    exit()

# Extract targets from h5ad 'perturbation' column
all_entries = adata.obs['perturbation'].unique().astype(str).tolist()
gene_list = [g for g in all_entries if g.lower() not in [negative_ctrl.lower(), 'other', 'nan', 'unknown']]
print(f"Detected {len(gene_list)} active perturbation targets in h5ad.")

print("Loading Barcode Table...")
# Using sep='\s+' to handle the space-separated format you showed
bc_frame = pd.read_csv(barcode_filepath, sep='\s+') 

# Standardize column names to lowercase for safety
bc_frame.columns = bc_frame.columns.str.lower()

# --- FIX: Mapping Logic ---
# Your file has 'gene', 'sgrna', 'barcode', 'readcount', 'umicount'
if 'gene' not in bc_frame.columns:
    print(f"CRITICAL: 'gene' column not found in barcode file. Found: {list(bc_frame.columns)}")
    exit()

print(f"Synchronizing 'non-targeting' in barcode file to '{negative_ctrl}'...")
bc_frame['gene'] = bc_frame['gene'].replace(['non-targeting', 'non'], negative_ctrl)

# Strip prefixes (S1L1_ etc) if present
bc_frame['cell'] = bc_frame['cell'].astype(str).str.split('_').str[-1]

# Create the map: Cell Barcode -> Gene Target
barcode_map = bc_frame.set_index('cell')['gene'].to_dict()

# Apply the map to a new column 'gene' in adata.obs for the analyzer to use
adata.obs['gene'] = adata.obs_names.map(barcode_map).fillna('Other')

print("--- MAPPING VERIFICATION ---")
print(adata.obs['gene'].value_counts().head(10))

# ==============================================================================
# 3. SURGICAL LOOP (CALCULATE PS SCORES)
# ==============================================================================
analyzer = PerturbAnalyzer(adata, neg_ctrl=negative_ctrl)

print(f"--- STARTING ANALYSIS FOR {len(gene_list)} GENES ---")

# 1. Initialize a dictionary to hold all new columns in memory
new_scores_dict = {}

for target_gene in tqdm(gene_list, desc="Calculating PS Scores"):
    scores = analyzer.calculate_ps_score(target_gene)
    
    if scores is not None:
        score_name = f"{target_gene}_eff"
        # Store the aligned scores in the dictionary instead of writing to adata.obs directly
        new_scores_dict[score_name] = scores.reindex(adata.obs_names).fillna(0)
        scores.to_csv(f"tables_batch/{target_gene}_PS_Scores.csv")

# 2. Convert the dictionary to a DataFrame and concatenate it to adata.obs all at once
if new_scores_dict:
    new_scores_df = pd.DataFrame(new_scores_dict)
    adata.obs = pd.concat([adata.obs, new_scores_df], axis=1)

# 3. Force defragmentation of the resulting DataFrame
# This forces Pandas to reallocate the underlying C-arrays into a single contiguous block
adata.obs = adata.obs.copy()

# ==============================================================================
# 4. GLOBAL LDA & UMAP GENERATION
# ==============================================================================
print("--- GENERATING GLOBAL LDA MAP ---")
try:
    analyzer.compute_lda_umap(gene_list)
    print("Generating Fixed UMAP plots...")
    plot_ps_on_lda(adata, gene_list, output_dir="plots_fixed_lda", neg_ctrl=negative_ctrl)
    plot_global_summary(adata, output_dir="plots_fixed_lda", score_threshold=0.8, downsample_bg=0.05)
except Exception as e:
    print(f"ERROR in Global Visualization: {e}")

# ==============================================================================
# 5. GENERATING LABELED DIAGNOSTIC SCATTER PLOTS
# ==============================================================================
print("--- GENERATING LABELED DIAGNOSTIC SCATTER PLOTS ---")

for target_gene in tqdm(gene_list, desc="Generating Scatters"):
    score_col = f"{target_gene}_eff"
    
    if score_col not in adata.obs.columns or target_gene not in adata.var_names:
        continue
        
    # FIX: Use obs_vector to guarantee a 1D array regardless of sparse/dense status
    exp_data = adata.obs_vector(target_gene)
    
    plot_df = pd.DataFrame({
        'PS_Score': adata.obs[score_col],
        'Expression': exp_data,
        'Group': adata.obs['gene']
    }).fillna(0)

    df_ctrl = plot_df[plot_df['Group'] == negative_ctrl].copy()
    df_target = plot_df[plot_df['Group'] == target_gene].copy()
    
    if len(df_target) < 5: # Minimum threshold to plot
        continue

    # Downsample background for visual clarity
    if len(df_ctrl) > 2000:
        df_ctrl = df_ctrl.sample(n=2000, random_state=42)

    plt.figure(figsize=(9, 7))
    
    # Layer 1: Background
    plt.scatter(df_ctrl['PS_Score'], df_ctrl['Expression'], c='lightgrey', s=30, alpha=0.3, label=negative_ctrl)
    # Layer 2: Signal
    plt.scatter(df_target['PS_Score'], df_target['Expression'], c='#e74c3c', s=40, alpha=0.7, label=target_gene, edgecolors='white', linewidth=0.5)

    # Thresholds
    h_thresh = df_ctrl['Expression'].median()
    v_thresh = 0.5
    plt.axvline(x=v_thresh, color='black', linestyle='--', alpha=0.5)
    plt.axhline(y=h_thresh, color='black', linestyle='--', alpha=0.5)

    # Quadrant Labels
    y_max = plot_df['Expression'].max()
    plt.text(0.75, y_max * 0.9, 'ESCAPERS', fontsize=10, fontweight='bold', color='darkred', ha='center')
    plt.text(0.25, y_max * 0.9, 'CONTROL / WT', fontsize=10, fontweight='bold', color='blue', ha='center')

    plt.title(f"Validation: {target_gene}")
    plt.xlabel("Perturbation Score (PS)")
    plt.ylabel(f"Raw {target_gene} Expression")
    plt.legend(loc='upper right')
    
    plt.savefig(f"plots_scatter_validation/{target_gene}_labeled_scatter.png", dpi=300)
    plt.close()

    # ==============================================================================
# 6. SAVE THE FINAL H5AD FILE
# ==============================================================================
print(f"--- SAVING PS SCORES TO H5AD ---")
adata.write_h5ad(output_h5ad_path)
print(f"STDOUT: Pipeline complete. Successfully saved to {output_h5ad_path}")
