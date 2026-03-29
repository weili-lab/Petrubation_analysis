import tiledbsoma
with tiledbsoma.Experiment.open("ReplogleWeissman2022_K562_gwps.soma", "r") as exp:
    var_df = exp.ms["RNA"].var.read().concat().to_pandas()
    obs_df = exp.obs.read().concat().to_pandas()
    print("--- VAR COLUMNS ---")
    print(var_df.columns.tolist())
    print(var_df.head(3))
    print("\n--- OBS COLUMNS ---")
    print(obs_df.columns.tolist())
    print(obs_df['perturbation'].unique()[:5])
