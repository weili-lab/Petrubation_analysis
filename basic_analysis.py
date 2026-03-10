import scanpy as sc
import pandas as pd
import numpy as np

# 1. Load the structure in backed mode
adata = sc.read_h5ad('ReplogleWeissman2022_rpe1.h5ad', backed='r')

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

# ==========================================
# --- LOG-NORMALIZATION EVALUATION ---
# ==========================================
print("\n--- NORMALIZATION STATUS ---")

# Check 1: Metadata analysis
has_log1p_metadata = 'log1p' in adata.uns
if has_log1p_metadata:
    print("Metadata Check: 'log1p' key FOUND in adata.uns.")
else:
    print("Metadata Check: 'log1p' key NOT FOUND in adata.uns.")

# Check 2: Numerical distribution analysis
# Extract 100 cells across all genes to ensure a statistically significant sample
sample_X = adata[:100, :].X
if hasattr(sample_X, "toarray"):
    sample_X = sample_X.toarray()

max_val = np.max(sample_X)
# Mean of non-zero entries to prevent artificial skewing from matrix sparsity
mean_val = np.mean(sample_X[sample_X > 0]) 

print(f"\nValue Metrics:")
print(f"  Maximum matrix value: {max_val:.2f}")
print(f"  Mean non-zero value:  {mean_val:.2f}")

# The Final Verdict
print("\nBLUNT TRUTH VERDICT:")
if max_val < 25 and has_log1p_metadata:
    print("Confirmed: The data is LOG-NORMALIZED.")
elif max_val < 25 and not has_log1p_metadata:
    print("Likely: The data is LOG-NORMALIZED (Matrix contains small floats, but Scanpy metadata is missing).")
elif max_val > 25 and np.all(sample_X == np.floor(sample_X)):
    print("Confirmed: The data contains RAW COUNTS (Matrix consists entirely of large integers).")
else:
    print("Ambiguous: The numerical distribution does not cleanly align with standard raw or log-normalized profiles. Manual inspection of the matrix is required.") 