import scanpy as sc
import pandas as pd
import numpy as np

def interrogate_guide_architecture(h5ad_path):
    print("--- INITIATING MULTIPLEXED GUIDE DATA HUNT ---")
    
    # Load in backed mode to save memory during metadata extraction
    adata = sc.read_h5ad(h5ad_path, backed='r')
    
    # Define target keywords that pipelines (like Cell Ranger or Perturb-seq) use
    target_keywords = ['guide', 'sgrna', 'umi', 'read', 'count', 'perturbation', 'target']
    
    # ==========================================
    # SEARCH VECTOR 1: .obs (Delimited Strings)
    # ==========================================
    print("\n[1] SCANNING .obs FOR DELIMITED MULTI-GUIDE STRINGS...")
    found_obs_candidates = {}
    
    # We look for columns that contain characters like '|' or ',' 
    # which denote multiple guides squeezed into one cell's row.
    for col in adata.obs.columns:
        if any(keyword in col.lower() for keyword in target_keywords):
            sample_val = str(adata.obs[col].iloc[0])
            if ',' in sample_val or '|' in sample_val:
                found_obs_candidates[col] = "Delimited String (Likely Multi-Guide)"
            elif adata.obs[col].dtype == 'object':
                found_obs_candidates[col] = "Categorical/String (Possibly Single Guide)"
                
    if found_obs_candidates:
        print("POTENTIAL MULTI-GUIDE COLUMNS FOUND IN .obs:")
        for k, v in found_obs_candidates.items():
            print(f"  - '{k}': {v} (Sample: {str(adata.obs[k].iloc[0])[:50]})")
    else:
        print("BLUNT TRUTH: No delimited string columns found in .obs.")

    # ==========================================
    # SEARCH VECTOR 2: .obsm (Guide Count Matrices)
    # ==========================================
    print("\n[2] SCANNING .obsm FOR GUIDE COUNT MATRICES...")
    # .obsm stores multidimensional arrays aligned to cells. 
    # Raw guide UMIs are frequently stored here as a separate matrix.
    obsm_candidates = []
    for key in adata.obsm.keys():
        if any(keyword in key.lower() for keyword in target_keywords):
            obsm_candidates.append(key)
            
    if obsm_candidates:
        print("POTENTIAL GUIDE MATRICES FOUND IN .obsm:")
        for key in obsm_candidates:
            shape = adata.obsm[key].shape
            print(f"  - '{key}': Matrix Shape {shape} (Rows: Cells, Cols: Guides)")
    else:
        print("BLUNT TRUTH: No dedicated guide count matrices found in .obsm.")

    # ==========================================
    # SEARCH VECTOR 3: .uns (Mapping Dictionaries)
    # ==========================================
    print("\n[3] SCANNING .uns FOR SGRNA-TO-GENE MAPPINGS...")
    # To map 'CD81.2' to 'CD81', we need a dictionary. These live in .uns.
    uns_candidates = []
    for key in adata.uns.keys():
        if any(keyword in key.lower() for keyword in target_keywords):
            uns_candidates.append(key)
            
    if uns_candidates:
        print("POTENTIAL MAPPING DATA STRUCTURES FOUND IN .uns:")
        for key in uns_candidates:
             print(f"  - '{key}': Type {type(adata.uns[key])}")
    else:
        print("BLUNT TRUTH: No obvious sgRNA-to-Gene mapping dictionaries found in .uns.")
    
    # ==========================================
    # SEARCH VECTOR 4: UMI COUNT VALIDATION
    # ==========================================
    print("\n[4] VALIDATING UMI COUNT PRESENCE & INTEGRITY...")
    
    umi_found = False
    # Check .obsm for numerical arrays that aren't just weights/embeddings
    for key in adata.obsm.keys():
        data = adata.obsm[key]
        
        # Logic: UMI counts must be numeric and usually integers
        is_numeric = np.issubdtype(data.dtype, np.number)
        
        if is_numeric:
            # Check if values are discrete (integers) or continuous (normalized)
            # We take a small sample to save time on large files
            sample = data[:100, :].toarray() if hasattr(data, "toarray") else data[:100, :]
            is_integer = np.all(np.equal(np.mod(sample, 1), 0))
            
            if is_integer and np.max(sample) > 0:
                print(f"  - MATCH: '{key}' contains integer values. High probability of UMI counts.")
                print(f"    Max UMI in sample: {np.max(sample)}")
                umi_found = True
            elif is_numeric:
                print(f"  - NOTE: '{key}' is numeric but contains floats. Likely normalized data, not raw UMIs.")

    # Check if a separate 'layers' entry exists (common in some Perturb-seq objects)
    if hasattr(adata, 'layers'):
        for layer in adata.layers.keys():
            if any(k in layer.lower() for k in ['counts', 'raw', 'umi']):
                print(f"  - MATCH: Found potential UMI layer in .layers['{layer}']")
                umi_found = True

    if not umi_found:
        print("BLUNT TRUTH: No explicit integer UMI count matrices identified in .obsm or .layers.")

    # ==========================================
    # LOGICAL VERDICT & EXTRACTION STRATEGY
    # ==========================================
    print("\n--- EXTRACTION STRATEGY VERDICT ---")
    if any("Delimited" in v for v in found_obs_candidates.values()):
        print("LOGIC: Data is stored as delimited lists in .obs. Use pandas.DataFrame.explode() to split strings into rows.")
    elif obsm_candidates:
        print("LOGIC: Data is stored as a matrix in .obsm. Use pandas.melt() to convert the wide matrix into your long target format.")
    else:
        print("LOGIC: CRITICAL FAILURE. The h5ad file does not appear to contain raw, multiplexed guide tracking data. It only contains the pre-collapsed, single-guide assignments.")

interrogate_guide_architecture('ReplogleWeissman2022_K562_essential.h5ad')
