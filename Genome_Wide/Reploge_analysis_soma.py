#!/usr/bin/env python3
import os, math, tiledbsoma, matplotlib, time
import numpy as np
import pandas as pd
import multiprocessing as mp
from scipy.stats import ks_2samp
import statsmodels.stats.multitest as smm
from functools import partial
matplotlib.use('Agg')

shared_mem_buffer = None

def init_worker(buffer):
    global shared_mem_buffer
    shared_mem_buffer = buffer

def worker_calc_stats(target_gene, soma_path, gene_to_id, pert_col, ctrl_indices):
    try:
        full_sf = np.frombuffer(shared_mem_buffer, dtype=np.float64)
        
        # Open the experiment ONCE per worker task
        with tiledbsoma.Experiment.open(soma_path, "r") as exp:
            X = exp.ms["RNA"].X["data"]
            gene_soma_id = gene_to_id[target_gene]

            # 1. Get KO Indices
            obs_query = exp.obs.read(value_filter=f"{pert_col} == '{target_gene}'", 
                                     column_names=["soma_joinid"]).concat().to_pandas()
            target_ko_ids = obs_query['soma_joinid'].to_numpy()
            
            if len(target_ko_ids) < 20: return None

            # 2. Optimized Read: Pull KO and Ctrl in a single indexed query
            # Combining these into one 'read' call reduces disk seek latency by 50%
            all_ids = np.concatenate([target_ko_ids, ctrl_indices])
            raw_table = X.read(coords=(all_ids, [gene_soma_id])).tables().concat().to_pandas()
            
            if raw_table.empty:
                return {"Target": target_gene, "KS_Stat": 0.0, "P_Value": 1.0}

            # Map results
            lookup = dict(zip(raw_table['soma_dim_0'], raw_table['soma_data']))
            
            expr_ko_raw = np.array([lookup.get(i, 0.0) for i in target_ko_ids])
            expr_ctrl_raw = np.array([lookup.get(i, 0.0) for i in ctrl_indices])

            # Normalized stats
            expr_ko = np.log1p(expr_ko_raw * full_sf[target_ko_ids])
            expr_ctrl = np.log1p(expr_ctrl_raw * full_sf[ctrl_indices])

            ks_stat, p_val = ks_2samp(expr_ko, expr_ctrl, alternative='two-sided')
            return {"Target": target_gene, "KS_Stat": float(ks_stat), "P_Value": float(p_val)}
    except:
        return None

def main():
    start_time = time.time()
    soma_path = "ReplogleWeissman2022_K562_gwps.soma"
    outdir = "target_validation_results_soma"
    n_cores = 16  # Moderate increase
    os.makedirs(outdir, exist_ok=True)

    with tiledbsoma.Experiment.open(soma_path, "r") as exp:
        var_df = exp.ms["RNA"].var.read(column_names=["soma_joinid", "gene_name"]).concat().to_pandas()
        gene_to_id = dict(zip(var_df['gene_name'], var_df['soma_joinid']))
        
        obs_df = exp.obs.read(column_names=["soma_joinid", "perturbation", "ncounts"]).concat().to_pandas()
        ctrl_indices = obs_df[obs_df['perturbation'] == 'control']['soma_joinid'].to_numpy()
        targets = [t for t in obs_df['perturbation'].unique() if t != 'control' and t in gene_to_id]
        
        # Shared Memory setup
        total_counts = obs_df['ncounts'].to_numpy()
        sf_raw = 10000.0 / (total_counts + 1e-9)
        shm = mp.RawArray('d', len(sf_raw))
        np.copyto(np.frombuffer(shm, dtype=np.float64), sf_raw)

    print(f"Processing {len(targets)} targets with {n_cores} cores...")
    
    with mp.Pool(processes=n_cores, initializer=init_worker, initargs=(shm,)) as pool:
        func = partial(worker_calc_stats, soma_path=soma_path, gene_to_id=gene_to_id, 
                       pert_col='perturbation', ctrl_indices=ctrl_indices)
        
        results = []
        for i, res in enumerate(pool.imap_unordered(func, targets, chunksize=10)):
            if res: results.append(res)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - start_time
                print(f"Progress: {i+1}/{len(targets)} | Time: {elapsed:.1f}s")

    if results:
        df = pd.DataFrame(results)
        _, df['FDR'], _, _ = smm.multipletests(df['P_Value'], method='fdr_bh')
        df.sort_values(['FDR', 'KS_Stat'], ascending=[True, False]).to_csv(f"{outdir}/KS_results_SOMA.csv", index=False)
        print(f"Total Run Time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()