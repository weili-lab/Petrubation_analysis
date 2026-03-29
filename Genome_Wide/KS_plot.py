#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def generate_fdr_barplot(csv_path, outdir, top_n=50):
    # 1. Load the pre-calculated summary
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
    
    df = pd.read_csv(csv_path)
    
    # 2. Filtering and Sorting
    # We sort by FDR (ascending) and KS_Stat (descending) to get the best hits
    df_sorted = df.sort_values(by=['FDR', 'KS_Stat'], ascending=[True, False]).head(top_n)
    
    # Reverse for the horizontal bar plot (highest bars at the top)
    df_plot = df_sorted.iloc[::-1]

    # 3. Create Figure
    # Height is dynamic: 0.25 inches per bar, minimum 5 inches
    fig_height = max(5, len(df_plot) * 0.25)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Plotting -log10_FDR
    bars = ax.barh(df_plot['Target'], df_plot['-log10_FDR'], 
                   color='steelblue', edgecolor='black', alpha=0.8)

    # 4. Formatting
    # Add significance threshold line at FDR = 0.05
    threshold = -np.log10(0.05)
    ax.axvline(threshold, color='red', linestyle='--', linewidth=1.5, label='FDR = 0.05')
    
    ax.set_xlabel('-log10(FDR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Perturbation Target', fontsize=12, fontweight='bold')
    ax.set_title(f'Top {top_n} Significant Perturbations (Replogle 2022)', fontsize=14)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    ax.legend(loc='lower right')

    # Add KS Stat labels to the end of bars for extra context
    for i, bar in enumerate(bars):
        ks_val = df_plot.iloc[i]['KS_Stat']
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'KS: {ks_val:.3f}', va='center', fontsize=8, color='dimgray')

    plt.tight_layout()
    
    # 5. Save
    out_file = os.path.join(outdir, "Top50_FDR_Barplot.pdf")
    fig.savefig(out_file, bbox_inches='tight')
    fig.savefig(out_file.replace(".pdf", ".png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"✅ Bar plot generated: {out_file}")

if __name__ == "__main__":
    # Update these paths to match your current directory
    CSV_FILE = "target_validation_results_genome_wide/KS_test_summary.csv"
    OUTPUT_DIR = "target_validation_results_genome_wide"
    
    generate_fdr_barplot(CSV_FILE, OUTPUT_DIR, top_n=50)
