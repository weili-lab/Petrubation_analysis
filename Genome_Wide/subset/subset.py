import scanpy as sc

def create_top100_subset(input_h5ad, output_h5ad, perturb_col='perturbation', control_label='control'):
    print(f"--- LOADING FULL DATASET: {input_h5ad} ---")
    adata = sc.read_h5ad(input_h5ad)
    
    # 1. Count frequencies of all perturbations
    counts = adata.obs[perturb_col].value_counts()
    
    # 2. Separate true targets from the control baseline
    targets_only = counts.drop(control_label, errors='ignore')
    
    # 3. Identify exactly the top 100 targets with the highest cell counts
    top_100_targets = targets_only.head(100).index.tolist()
    
    # 4. Re-append the control label (Strictly required for baseline math)
    keep_list = top_100_targets + [control_label]
    
    # 5. Execute the surgical subset (.copy() prevents memory view errors)
    adata_subset = adata[adata.obs[perturb_col].isin(keep_list)].copy()
    
    # 6. Verify and Save
    print("\n--- SUBSET METRICS ---")
    print(f"Original Cell Count: {adata.n_obs}")
    print(f"Subset Cell Count:   {adata_subset.n_obs}")
    reduction = 100 - (adata_subset.n_obs / adata.n_obs * 100)
    print(f"Data Reduction:      {reduction:.1f}%")
    print(f"Total Categories:    {len(adata_subset.obs[perturb_col].unique())} (100 Targets + 1 Control)")
    
    print(f"\n--- SAVING SUBSET TO: {output_h5ad} ---")
    adata_subset.write_h5ad(output_h5ad)
    print("STDOUT: Subsetting complete.")

if __name__ == '__main__':
    create_top100_subset(
        input_h5ad="ReplogleWeissman2022_K562_gwps_processed.h5ad", 
        output_h5ad="ReplogleWeissman2022_K562_gwps_processed_top100_dev.h5ad"
    )
