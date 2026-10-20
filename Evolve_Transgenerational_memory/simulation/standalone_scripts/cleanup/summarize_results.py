from pathlib import Path
import pandas as pd
import numpy as np
import dask.dataframe as dd


base = Path("/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/results/new_results/different_sp/finegrain/dim20/epi")

for dim in ["dim1"]:
    folder = base / dim
    print(f"Processing {folder}")

    dim_data = {}

    # --- Load CSVs into Dask ---
    for path in folder.glob("*.csv"):
        filename = path.stem
        name = "_".join(filename.split("_")[:2]) #extracts teh first twowords of the filename
        dim_data[name] = dd.read_csv(path)

    # --- Compute required columns (row-wise means of last 50 columns) ---
    mean_memory = dim_data["mean_memory"]
    mean_neutral = dim_data["mean_neutral"]
    epi_fitness = dim_data["epi_fitness"]

    df_out = pd.DataFrame({
        "rho": mean_memory.iloc[:, 0],
        "m": mean_memory.iloc[:, 1],
        "epi_mem_p": mean_memory.iloc[:, -50:].mean(axis=1),
        "neutral_trait_p": mean_neutral.iloc[:, -50:].mean(axis=1),
        "epi_fitness": epi_fitness.iloc[:, -50:].mean(axis=1)
    })

    # --- Save CSV ---
    out_path = folder / f"{dim}_summary.csv"
    df_out.to_csv(out_path, index=False)

    print(f"Saved → {out_path}")
