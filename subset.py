import os
import anndata as ad
import scanpy as sc
from scipy.sparse import csr_matrix

input_file = "PertTF_Combined_Clean_Merged.h5ad"
output_file = "PertTF_Subset_100MB.h5ad"

print("Loading dataset...")
adata = ad.read_h5ad(input_file)

# Clear the non-unique cell names warning
if not adata.obs_names.is_unique:
    adata.obs_names_make_unique()

# Convert to sparse format
if not isinstance(adata.X, csr_matrix):
    adata.X = csr_matrix(adata.X)

# Downsample further to 15,000 cells (All genes remain intact)
print("Downsampling to 15000 cells...")
sc.pp.subsample(adata, n_obs=15000, random_state=42)

# Save with high-ratio gzip compression
print("Saving compressed file...")
adata.write_h5ad(output_file, compression="gzip")

final_size = os.path.getsize(output_file) / (1024 * 1024)
print(f"Done! New file size: {final_size:.2f} MB")
