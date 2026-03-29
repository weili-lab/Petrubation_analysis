#!/usr/bin/env python3
import os
import math
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
import statsmodels.stats.multitest as smm
import multiprocessing as mp
from functools import partial
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib
matplotlib.use('Agg')

def ensure_dir(d: str): 
    os.makedirs(d, exist_ok=True)

def worker_calc_stats(target_gene, expr_matrix, var_names, perturbation_obs, min_cells=20):
    """
    Worker function compatible with 'spawn'. 
    Note: expr_matrix is passed as a sparse CSR matrix to save memory.
    """
    try:
        # Find index of the gene
        gene_idx = np.where(var_names == target_gene)[0][0]
        
        # Get indices for KO and Control
        ko_idx = np.where(perturbation_obs == target_gene)[0]
        ctrl_idx = np.where(perturbation_obs == 'control')[0]

        if len(ko_idx) < min_cells:
            return None

        # Slice sparse matrix and convert to dense only for the specific cells
        # This is the most memory-efficient way to handle large matrices
        expr_ko = np.asarray(expr_matrix[ko_idx, gene_idx].todense()).flatten()
        expr_ctrl = np.asarray(expr_matrix[ctrl_idx, gene_idx].todense()).flatten()

        ks_stat, p_val = ks_2samp(expr_ko, expr_ctrl, alternative='two-sided')
        
        return {
            "Target": target_gene,
            "N_KO": len(expr_ko),
            "N_Ctrl": len(expr_ctrl),
            "KS_Stat": float(ks_stat),
            "P_Value": float(p_val),
            "expr_ko": expr_ko.astype(np.float32), 
            "expr_ctrl": expr_ctrl.astype(np.float32)
        }
    except Exception as e:
        return None

def generate_3x3_grid_pdf(results_list, outdir, filename="ECDF_Grids_3x3.pdf"):
    n_targets = len(results_list)
    n_pages = math.ceil(n_targets / 9)
    pdf_path = os.path.join(outdir, filename)
    
    print(f"Generating {n_pages} pages of ECDF grids...")
    with PdfPages(pdf_path) as pdf:
        for p in range(n_pages):
            fig, axes = plt.subplots(3, 3, figsize=(15, 14))
            axes = axes.flatten()
            for i in range(9):
                idx = p * 9 + i
                if idx >= n_targets:
                    axes[i].axis('off')
                    continue
                res = results_list[idx]
                ax = axes[i]
                x_ko, y_ko = np.sort(res['expr_ko']), np.arange(1, res['N_KO'] + 1) / res['N_KO']
                x_ctrl, y_ctrl = np.sort(res['expr_ctrl']), np.arange(1, res['N_Ctrl'] + 1) / res['N_Ctrl']
                ax.step(x_ko, y_ko, color='red', label=f'KO (n={res["N_KO"]})', where='post')
                ax.step(x_ctrl, y_ctrl, color='black', label='Ctrl', where='post')
                ax.set_title(f"{res['Target']}\nKS: {res['KS_Stat']:.3f} | FDR: {res['FDR']:.2e}", fontsize=10, fontweight='bold')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=8)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

def main():
    # SET START METHOD TO SPAWN
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing context set to 'spawn'.")
    except RuntimeError:
        pass

    h5ad_path = "ReplogleWeissman2022_K562_gwps.h5ad" 
    outdir = "target_validation_results_genome_wide"
    n_cores = 30  # You can try 53 again with spawn, but 32 is safer
    ensure_dir(outdir)

    print("Loading AnnData...")
    adata = sc.read_h5ad(h5ad_path)
    
    print("Normalizing and log-transforming...")
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Extract raw data to reduce pickling overhead
    targets = [t for t in adata.obs['perturbation'].unique() if t != 'control']
    var_names = adata.var_names.to_numpy()
    perturbation_obs = adata.obs['perturbation'].to_numpy()
    
    # Ensure CSR format for efficient slicing
    expr_matrix = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    if not isinstance(expr_matrix, sp.csr_matrix):
        expr_matrix = expr_matrix.tocsr()
    
    # Free up memory before starting workers
    del adata 

    print(f"Parallelizing with {n_cores} cores using SPAWN...")
    worker_func = partial(worker_calc_stats, expr_matrix=expr_matrix, 
                          var_names=var_names, perturbation_obs=perturbation_obs)

    # Using 'with' statement ensures the pool is cleaned up properly
    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker_func, targets)

    results = [r for r in results if r is not None]
    
    if results:
        # FDR and Sorting
        pvals = [r['P_Value'] for r in results]
        _, fdrs, _, _ = smm.multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, res in enumerate(results):
            res['FDR'] = fdrs[i]
            res['-log10_FDR'] = -np.log10(max(fdrs[i], 1e-300))

        results.sort(key=lambda x: (x['FDR'], -x['KS_Stat']))
        
        # Save CSV
        summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['expr_ko', 'expr_ctrl']} for r in results])
        summary_df.to_csv(os.path.join(outdir, "KS_test_summary.csv"), index=False)

        # Generate the PDF grids
        generate_3x3_grid_pdf(results, outdir)
        print(f"Analysis Complete. Results in {outdir}")

if __name__ == "__main__":
    main()
