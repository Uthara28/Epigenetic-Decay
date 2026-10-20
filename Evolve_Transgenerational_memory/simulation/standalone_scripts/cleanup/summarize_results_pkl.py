import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image
import seaborn as sns
import statistics
import math
import pickle 
import os
import csv
import multiprocessing as mp
import argparse
import concurrent.futures
import matplotlib.colors as mcolors
import re
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pathlib import Path
from dask import dataframe as df1
import dask.dataframe as dd


#for epi


folder = Path("/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/results/new_results/finegrain_30_30_envs/1D_constant_env/with_seed/epi")

epi_obj = {}

for path in folder.glob("*.pkl"):
    print(path)

    # extract dimension from filename
    match = re.search(r"dims(\d+)", path.stem)
    if match:
        dim = int(match.group(1))
        with open(path, "rb") as file:
            results_array_epi = pickle.load(file)
        
        epi_obj[f"epi_{dim}"] = results_array_epi

print(epi_obj.keys())

folder = Path("/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/results/new_results/finegrain_30_30_envs/1D_constant_env/with_seed/onlygen")

onlygen_obj = {}

for path in folder.glob("*.pkl"):
    print(path)

    match = re.search(r"dims(\d+)", path.stem)
    if match:
        dim = int(match.group(1))
        with open(path, "rb") as file:
            results_array_gen = pickle.load(file)
        
        onlygen_obj[f"gen_{dim}"] = results_array_gen

print(onlygen_obj.keys())


dims=[1]  #list of dimensions under consideration

for i in dims:
    
    concat_epi = epi_obj[f"epi_{i}"]
    concat_gen = onlygen_obj[f"gen_{i}"]   # adjust key if needed
    
    keys = list(concat_epi[0].keys())

    concatenated_results_epi = {}
    concatenated_results_gen = {}

    for k in keys:
            concatenated_results_epi[k] = np.vstack([r[k] for r in concat_epi])
            
    for k in keys:
            concatenated_results_gen[k] = np.vstack([r[k] for r in concat_gen])
            
    df_epi = pd.DataFrame({
        
    'rho': concatenated_results_epi['rho_m_alpha_beta'][:,0],
    'm': concatenated_results_epi['rho_m_alpha_beta'][:,1],
    'fitness':np.mean(concatenated_results_epi['meanw'][:,-50:], axis=1),
    'epi_mem_p': np.mean(concatenated_results_epi['meanmemory_p'][:,-50:], axis=1),
    'neutral_trait_p' : np.mean(concatenated_results_epi['meanneutral_p'][:,-50:], axis=1 ),
    'stdw': np.mean(concatenated_results_epi['stdw'][:,:-100], axis=1),
    'mean_phenotype' : np.mean(concatenated_results_epi['mean_phenotype'][:,:-50], axis=1 ),
    'std_phenotype' : np.mean(concatenated_results_epi['std_phenotype'][:,:-50], axis=1 ),
    
    })
    
    df_gen = pd.DataFrame({
            
        'rho': concatenated_results_gen['rho_m_alpha_beta'][:,0],
        'm': concatenated_results_gen['rho_m_alpha_beta'][:,1],
        'fitness':np.mean(concatenated_results_gen['meanw'][:,-50:], axis=1),
        'stdw':np.mean(concatenated_results_gen['stdw'][:,-50:], axis=1),
        'mean_phenotype' : np.mean(concatenated_results_gen['mean_phenotype'][:,:-50], axis=1 ),
        'std_phenotype' : np.mean(concatenated_results_gen['std_phenotype'][:,:-50], axis=1 )
        })
    
    
    out_path = folder / f"dim{i}_epi_summary.csv"
    df_epi.to_csv(out_path, index=False)
    
    out_path = folder / f"dim{i}_gen_summary.csv"
    df_gen.to_csv(out_path, index=False)
    print(f"Saved → {out_path}")
