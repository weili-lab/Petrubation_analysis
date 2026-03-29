#!/usr/bin/env python3
import scanpy as sc
import os

def convert_h5ad_to_zarr(h5ad_path, zarr_path):
    print(f"Loading {h5ad_path} into memory...")
    # This requires enough RAM to hold the initial file once
    adata = sc.read_h5ad(h5ad_path)
    
    print(f"Writing to Zarr format at {zarr_path}...")
    # Chunking is crucial: (all_cells, 100_genes)
    # This optimizes disk reads for gene-wise (column-wise) operations
    adata.write_zarr(zarr_path, chunks=(adata.n_obs, 100))
    
    print("Conversion complete.")

if __name__ == "__main__":
    h5ad_file = "ReplogleWeissman2022_K562_gwps.h5ad"
    zarr_dir = "ReplogleWeissman2022_K562_gwps.zarr"
    
    if not os.path.exists(zarr_dir):
        convert_h5ad_to_zarr(h5ad_file, zarr_dir)
    else:
        print(f"Zarr directory {zarr_dir} already exists. Skipping conversion.")
