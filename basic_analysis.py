import scanpy as sc
import pandas as pd

# 1. Load the structure in backed mode
adata = sc.read_h5ad('ReplogleWeissman2022_K562_gwps.h5ad', backed='r')

print("--- EXPERIMENTAL LOGIC SUMMARY ---")

# 2. Perturbation Type 
if 'perturbation_type' in adata.obs:
    p_types = adata.obs['perturbation_type'].unique()
    print(f"Perturbation Type(s): {list(p_types)}")
else:
    print("Warning: 'perturbation_type' column not found.")

# 3. System Info
print(f"Biological System: {adata.obs['cell_line'].unique()[0]} ({adata.obs['tissue_type'].unique()[0]})")

# 4. Perturbation Diversity 
n_targets = adata.obs['perturbation'].nunique()
# Subtracting 1 because 'control' is in that list
print(f"Total Unique Genes Targeted: {n_targets - 1}")

print("\n--- CELL DISTRIBUTION PER TARGET ---")
print(adata.obs['perturbation'].value_counts().head(10))

# 6. Identifying the Baseline (The 'Control')
# Hardcoded to 'control' based on the dataset's ground truth
control_label = 'control'
print(f"\nDetected Control Label: ['{control_label}']")

print("\n--- DATA SCALE ---")
print(f"Total Cells: {adata.n_obs}")
print(f"Total Features (Genes): {adata.n_vars}")
print(f"Mean UMI per cell: {adata.obs['UMI_count'].mean():.1f}")

# ==========================================
# --- QUALITY CONTROL & EXPRESSION DATA ---
# ==========================================

print("\n--- QUALITY CONTROL (QC) METRICS (Mito & UMI) ---")
qc_summary = adata.obs[['percent_mito', 'UMI_count']].describe().loc[['min', 'max', 'mean']]
print(qc_summary)

print("\n--- TRUE EXPERIMENTAL GROUPS ---")
# Strict binary logic: It is either 'control' or it is perturbed.
actual_control_count = adata.obs[adata.obs['perturbation'] == control_label].shape[0]
perturbation_cells = adata.obs[adata.obs['perturbation'] != control_label].shape[0]

print(f"Baseline Control Cells: {actual_control_count}")
print(f"Perturbed Cells (Active Targets): {perturbation_cells}")

print("\n--- THE TRANSCRIPTOMIC DATA (The RNA Assay) ---")
genes_sample = adata.var_names[:5]
expression_sample = adata[:5, :5].X

if hasattr(expression_sample, "toarray"):
    expression_sample = expression_sample.toarray()

sample_df = pd.DataFrame(expression_sample, columns=genes_sample, index=adata.obs_names[:5])
print("Sample RNA Expression Matrix (First 5 cells x 5 genes):")
print(sample_df)
print("\nBlunt Truth: The numbers in the matrix above are your Single Cell RNA expression values, NOT the metadata columns.")
