from pathlib import Path
import pandas as pd
import numpy as np


dim_data = {}
rho_m = {}
folder = Path("/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/results/new_results/different_sp/finegrain/dim20/epi")
# --- Load CSVs into Dask ---
for path in folder.glob("*.csv"):
        filename = path.stem
        name = "_".join(filename.split("_")[:2]) #extracts teh first twowords of the filename
        cols = pd.read_csv(path, nrows=0).columns  
        dim_data[name] = dd.read_csv(path, usecols=cols[-50:]) 
        
rho_m = pd.read_csv(path, usecols=range(2))
    # --- Compute required columns (row-wise means of last 50 columns) ---
mean_memory = dim_data["mean_memory"]
mean_neutral = dim_data["mean_neutral"]
epi_fitness = dim_data["epi_fitness"]
gen_fitness = dim_data["genetic_fitness"]

df_out = pd.DataFrame({
        "rho": rho_m.iloc[:, 0],
        "m": rho_m.iloc[:, 1],
        "initial_mem": rho_m[:,2],
        "mean_last_50_genetic_fitness": gen_fitness.mean(axis=1),
        "mean_last_50_mean_memory": mean_memory.mean(axis=1),
        "mean_last_50_epi_fitness": epi_fitness.mean(axis=1),
        "mean_last_50_mean_neutral": mean_neutral.mean(axis=1)
    })

# --- Save CSV ---
out_path = folder / f"{dim}_summary.csv"
df_out.to_csv(out_path, index=False)