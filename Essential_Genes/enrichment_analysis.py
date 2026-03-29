import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
input_h5ad = "/local/projects-t3/lilab/vmenon/PertTF-Reploge-analysis/Essential_Genes/ReplogleWeissman2022_K562_essential_processed.h5ad"
cluster_res = 'leiden_res_0.5'  # Based on your UMAP grid analysis
perturb_col = 'perturbation'    
output_dir  = "enrichment_results"

os.makedirs(output_dir, exist_ok=True)

# ==============================================================================
# 2. ENRICHMENT LOGIC
# ==============================================================================
def run_full_enrichment():
    print(f"--- LOADING PROCESSED DATA: {input_h5ad} ---")
    adata = sc.read_h5ad(input_h5ad)
    
    # Validation Check: Ensure all genes are present
    print(f"STDOUT: Total genes in matrix: {adata.n_vars}")
    print(f"STDOUT: HVGs flagged: {adata.var['highly_variable'].sum()}")

    # 1. Create the cross-tabulation
    ct = pd.crosstab(adata.obs[perturb_col], adata.obs[cluster_res])
    total_cells = len(adata)
    results = []

    print(f"--- ANALYZING {len(ct.index)} TARGETS ---")

    for cluster in tqdm(ct.columns, desc="Clusters"):
        cluster_total = ct[cluster].sum()
        
        for perturb in ct.index:
            a = ct.loc[perturb, cluster]
            b = ct.loc[perturb].sum() - a
            c = cluster_total - a
            d = (total_cells - cluster_total) - b
            
            # Fisher's Exact Test for Over-representation
            odds_ratio, p_val = fisher_exact([[a, b], [c, d]], alternative='greater')
            
            results.append({
                'perturbation': perturb,
                'cluster': f"Cluster_{cluster}",
                'observed_count': a,
                'odds_ratio': odds_ratio,
                'p_value': p_val
            })

    # 2. Statistical Correction
    enrich_df = pd.DataFrame(results)
    enrich_df['adj_p_value'] = multipletests(enrich_df['p_value'], method='fdr_bh')[1]
    enrich_df = enrich_df.sort_values(['adj_p_value', 'odds_ratio'], ascending=[True, False])
    
    # Save CSV
    enrich_df.to_csv(f"{output_dir}/Enrichment_Stats_Res0.5.csv", index=False)
    
    # 3. Visualize Top 5 per Cluster
    significant_hits = enrich_df[enrich_df['adj_p_value'] < 0.05]
    top_genes = significant_hits.groupby('cluster').head(5)
    pivot_df = top_genes.pivot(index='perturbation', columns='cluster', values='odds_ratio').fillna(0)
    
    plt.figure(figsize=(14, 10))
    sns.heatmap(np.log1p(pivot_df), annot=pivot_df, fmt=".1f", cmap="YlOrRd")
    plt.title("Top Enriched Perturbations per Cluster (Res 0.5)")
    plt.savefig(f"{output_dir}/Enrichment_Heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"--- ANALYSIS COMPLETE: Results in {output_dir}/ ---")

if __name__ == "__main__":
    run_full_enrichment()
