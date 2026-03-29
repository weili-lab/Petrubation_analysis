#!/usr/bin/env python3
import os
import tiledbsoma
import tiledbsoma.io

def convert_h5ad_to_tiledb(h5ad_path: str, soma_path: str):
    """
    Converts a massive .h5ad file into a SOMA (TileDB) Experiment directory.
    This process is memory-efficient but heavily disk-bound.
    """
    if os.path.exists(soma_path):
        print(f"Error: SOMA URI '{soma_path}' already exists. Please remove it or choose a new path.")
        return

    print(f"Starting conversion of {h5ad_path} to TileDB-SOMA at {soma_path}...")
    print("This will take time depending on your disk write speed. RAM usage will remain stable.")

    # tiledbsoma.io.from_h5ad streams the data in chunks.
    # 'measurement_name' dictates the sub-directory where the X matrix and var dataframe live. 
    # 'RNA' is the standard convention.
    soma_uri = tiledbsoma.io.from_h5ad(
        experiment_uri=soma_path,
        input_path=h5ad_path,
        measurement_name="RNA"
    )

    print(f"Conversion absolute. Data successfully written to: {soma_uri}")

if __name__ == "__main__":
    # Define your paths
    h5ad_file = "ReplogleWeissman2022_K562_gwps.h5ad"
    tiledb_dir = "ReplogleWeissman2022_K562_gwps.soma"

    convert_h5ad_to_tiledb(h5ad_file, tiledb_dir)
