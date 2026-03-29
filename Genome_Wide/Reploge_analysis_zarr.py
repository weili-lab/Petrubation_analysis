#!/usr/bin/env python3
import os
import math
import zarr
import numpy as np
import pandas as pd
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

def worker_calc_stats(target_gene, zarr_path, var_names, ko_indices, ctrl_indices, size_factors, min_cells=20):
    """
    Worker function utilizing Zarr for lock-free, parallel chunk reads.
    """
    try:
        # Open Zarr store in read-only mode
        root = zarr.open(zarr_path, mode='r')
        
        # Locate gene index
        gene_idx = np.where(var_names == target_gene)[0][0]
        ko_idx = ko_indices.get(target_gene, [])

        if len(ko_idx) < min_cells:
            return None

        # Direct disk read of the specific gene column.
        # This bypasses the need for large memory structures.
        gene_expr_all = np.asarray(root['X'][:, gene_idx]).flatten()

        # Extract specific cells
        expr_ko_raw = gene_expr_all[ko_idx]
        expr_ctrl_raw = gene_expr_all[ctrl_indices]

        # Apply target_sum=1e4 normalization and log1p on-the-fly
        expr_ko = np.log1p(expr_ko_raw * size_factors[ko_idx])
        expr_ctrl = np.log1p(expr_ctrl_raw * size_factors[ctrl_indices])

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
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    zarr_path = "ReplogleWeissman2022_K562_gwps.zarr" 
    outdir = "target_validation_results_rpe1"
    n_cores = 30 
    ensure_dir(outdir)

    print("Reading metadata from Zarr store...")
    root = zarr.open(zarr_path, mode='r')
    
    # ---------------------------------------------------------
    # DYNAMIC METADATA LOOKUP
    # ---------------------------------------------------------
    var_group = root['var']
    # Extract the actual index key from the group's attributes
    var_index_key = var_group.attrs.get('_index', 'index')
    
    # Validation step to ensure the key exists
    if var_index_key not in var_group:
        available_keys = list(var_group.keys())
        raise KeyError(f"Index '{var_index_key}' not found in var. Available keys: {available_keys}")
        
    var_names = np.asarray(var_group[var_index_key]).astype(str)
    
    # Extract perturbation observations
    obs_group = root['obs']
    if 'perturbation' not in obs_group:
        raise KeyError(f"'perturbation' column not found in obs. Available keys: {list(obs_group.keys())}")
    perturbation_obs = np.asarray(obs_group['perturbation']).astype(str)
    # ---------------------------------------------------------

    print("Calculating size factors for normalization...")
    # Sum across columns (axis=1) in chunks to get library sizes without crashing RAM
    if 'total_counts' in root['obs']:
        total_counts = np.asarray(root['obs']['total_counts'])
    else:
        # Fallback: Compute row sums iteratively. 
        # Zarr handles this sequentially without loading the full matrix.
        X_array = root['X']
        total_counts = np.zeros(X_array.shape[0])
        # Read in blocks of 50000 cells to compute row sums
        step = 50000
        for i in range(0, X_array.shape[0], step):
            total_counts[i:i+step] = np.sum(X_array[i:i+step, :], axis=1)

    size_factors = 10000.0 / (total_counts + 1e-9)

    print("Pre-calculating indices for perturbations...")
    ctrl_indices = np.where(perturbation_obs == 'control')[0]
    unique_targets = np.unique(perturbation_obs)
    targets = [t for t in unique_targets if t != 'control']
    ko_indices = {t: np.where(perturbation_obs == t)[0] for t in targets}

    print(f"Parallelizing with {n_cores} cores. Target count: {len(targets)}")
    worker_func = partial(worker_calc_stats, 
                          zarr_path=zarr_path, 
                          var_names=var_names, 
                          ko_indices=ko_indices, 
                          ctrl_indices=ctrl_indices,
                          size_factors=size_factors)

    with mp.Pool(processes=n_cores) as pool:
        results = pool.map(worker_func, targets)

    results = [r for r in results if r is not None]
    
    if results:
        print("Calculating FDR and sorting results...")
        pvals = [r['P_Value'] for r in results]
        _, fdrs, _, _ = smm.multipletests(pvals, alpha=0.05, method='fdr_bh')
        for i, res in enumerate(results):
            res['FDR'] = fdrs[i]
            res['-log10_FDR'] = -np.log10(max(fdrs[i], 1e-300))

        results.sort(key=lambda x: (x['FDR'], -x['KS_Stat']))
        
        print("Saving CSV...")
        summary_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['expr_ko', 'expr_ctrl']} for r in results])
        summary_df.to_csv(os.path.join(outdir, "KS_test_summary.csv"), index=False)

        generate_3x3_grid_pdf(results, outdir)
        print(f"Analysis Complete. Results in {outdir}")
    else:
        print("No valid results found. Verify min_cells threshold and target mapping.")

if __name__ == "__main__":
    main()