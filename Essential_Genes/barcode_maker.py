import scanpy as sc
import pandas as pd
import re

def export_nontargeting_report(h5ad_path, output_filename="guide_report.txt"):
    # 1. Load data
    adata = sc.read_h5ad(h5ad_path)
    
    all_entries = []
    for cell_id, row in adata.obs.iterrows():
        # Skip empty cells
        if pd.isna(row['guide_id']) or str(row['guide_id']) == 'nan':
            continue
            
        # Split the string (e.g., MRPS31_A|non-targeting_02989)
        guides = str(row['guide_id']).split('|')
        
        # Determine the target gene for every guide in this cell to calculate frequency
        cell_targets = []
        for g in guides:
            g = g.strip()
            if not g: continue
            # Check if the guide is a control (non)
            if 'non' in g.lower():
                cell_targets.append("non-targeting")
            else:
                cell_targets.append(re.split(r'[-_+.]', g)[0])
        
        # Calculate frequencies for THIS cell
        target_to_count = pd.Series(cell_targets).value_counts().to_dict()
        
        # 2. Create the row-level data
        unique_guides_in_cell = set([g.strip() for g in guides if g.strip()])
        for g in unique_guides_in_cell:
            # Logic: Identify if current guide is a control or gene
            is_control = 'non' in g.lower()
            current_target = "non-targeting" if is_control else re.split(r'[-_+.]', g)[0]
            
            all_entries.append({
                'cell': cell_id,
                'barcode': g,                # The specific ID (e.g., non-targeting_02989)
                'sgRNA': current_target,     # Renamed to non-targeting
                'gene': current_target,      # Renamed to non-targeting
                'readCount': target_to_count[current_target],
                'UMIcount': target_to_count[current_target]
            })
    
    # 3. Create DataFrame and Export
    final_df = pd.DataFrame(all_entries)
    
    # Standardize column order and headers
    final_df = final_df[['cell', 'barcode', 'sgRNA', 'gene', 'readCount', 'UMIcount']]
    
    # Export with Space Separator and Header
    final_df.to_csv(output_filename, sep=' ', index=False, header=True)
    
    print(f"SUCCESS: {output_filename} created.")

# Execute
export_nontargeting_report('ReplogleWeissman2022_K562_essential.h5ad')
