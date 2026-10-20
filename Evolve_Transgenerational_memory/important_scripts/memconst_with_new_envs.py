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
import multiprocessing as mp
import argparse
import concurrent.futures
import matplotlib.colors as mcolors
import re
import sys
import secrets


nof_dimensions  = 2

initial_memory  =  0
   
rho_cases = 100
m_cases = 100

max_gen= 20000


results_array = {}  ### DONT FORGET TO RUN BEFORE NEW PARAM CONDITIONS

popsize = 5000

# Mutation rates: [epi-memory mutation rate, geno mutation rate]
mutrate = [0.001, 0.001]

# Mutation sizes: [epi-memory mutation size, geno mutation size]
mutsize = [1, 1]

epsilon = 0.05 # phenotypic noise



def get_cube_dims(nof_dimensions):
    """
    Generate index ranges representing different sections (geno, pheno, dev)
    of a cube structure based on the number of dimensions provided.
    """
    memory = 0
    neutral = 1

    geno = list(range(2, 2 + nof_dimensions))
    pheno = list(range(2 + nof_dimensions, 2 + 2 * nof_dimensions))
    dev = list(range(2 + 2 * nof_dimensions, 2 + 3 * nof_dimensions))

    total_layers = dev[-1] + 1

    return {
        "memory": memory,
        "neutral": neutral,
        "geno": geno,
        "pheno": pheno,
        "dev": dev,
        "total_layers": total_layers
    }


def create_cube(popsize, dims, nof_dimensions, epsilon=0.1, initial_memory=-3):
    
    """
    Create and initialize a population cube N with genotype, phenotype,
    and developmental layers.

    Parameters:se
        popsize (int): Number of individuals in the population.
        dims (dict): Output from get_cube_dims().
        nof_dimensions (int): Number of dimensions (traits).
        epsilon (float): Random noise factor for phenotype initialization.

    Returns:
        np.ndarray: Initialized cube with dimensions [popsize, total_layers].
    """
    
    # Initialize cube
    N = np.empty((popsize, dims['total_layers']))
    
    # Initialize memory & neutral layers
    N[:, dims['memory']:dims['neutral'] + 1] = initial_memory

    # Initialize genotype layer
    N[:, dims['geno']] = np.zeros((popsize, nof_dimensions))

    # Initialize phenotype layer with mut
    N[:, dims['pheno']] = N[:, dims['geno']] + epsilon * np.random.randn(popsize, nof_dimensions)

    # Initialize development layer with NaN
    N[:, dims['dev']] = np.nan

    return N


        

def run_simulation(params):
    scenarios, seed, popsize, maxgen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, memory = params
    
    rho_m_alpha_beta = scenarios[0:4]
    env_states = scenarios[4:]
    
    np.random.seed(seed)
    print(f"Using seed: {seed}")
    
    env_states = scenarios

    meanw = np.zeros(maxgen)
    meanmemory_g = np.zeros(maxgen)
    meanmemory_p= np.zeros(maxgen)
    meanneutral_g = np.zeros(maxgen)
    meanneutral_p = np.zeros(maxgen)
    mean_gen = np.zeros(maxgen)

    
    for t in range(0, (maxgen-1)):
        #if t%10 == 0:
        #    print(t)
        for d in range(0, nof_dimensions):
            N[:,dims['dev'][d]] = N[:,dims['pheno'][d]] - env_states[t]
            
        dist_from_optima = np.sqrt(np.sum(N[:,dims['dev']]**2, axis =1))
        
        #------distance correction ------
       
        W = np.exp((-dist_from_optima**2)/ (2))
        
        meanw[t] = np.mean(W)    
    
        
        meanmemory_g[t] = np.mean(N[:, dims['memory']])  # the actual memory
        meanmemory_p[t] = np.mean(1/(1 + np.exp(-N[:,dims['memory']]))) #with the logistc correction
        
        meanneutral_g[t] = np.mean(N[:,dims['neutral']])
        meanneutral_p[t] = np.mean(1/(1 + np.exp(-N[:,dims['neutral']])))
        mean_gen[t] = np.mean(N[:,dims['geno']])
        

        
        #current gen offspring (the cube gets stored here per gen)
        
        offspring = np.zeros((popsize, dims['total_layers'])) # empty 3D matrix with dimensions of the cube
        
        #sample offspring for each scenario weighted by fitness
        
        
        parents_idx = np.random.choice(popsize, size=popsize, p = (W/np.sum(W)) )#pick #popsize sized random numbers
        offspring[:,:] = N[parents_idx, :] #similar to how it works in matlab
        
        mutate_memories = np.random.uniform(low=0, high=1, size= popsize) < mutrate[0]
        #mutate_neutral = np.random.uniform(low=0, high=1, size=popsize) < mutrate[0]
        mutate_geno = np.random.uniform(low=0, high=1, size=(popsize, nof_dimensions)) < mutrate[1]
        
        # adding a mutation of size mutsize[1], rate mutate_memories, to the epi-memory
        # the epignetic memory evolves; 
        
        offspring[:,dims['memory']] = (
                                offspring[:,dims['memory']] 
                                + mutate_memories * np.random.normal(0, mutsize[0], popsize)
        )
        
        # adding a mutation of size mutsize[1] (epi-mutation size), rate mutate_memories, to the neutral phenotype
        # the neutral trait recieves random mutation but is not implemented in the calculation for fitness; so is evolving neutrally
        offspring[:,dims['neutral']] = (
                                offspring[:,dims['neutral']]  
                                + mutate_memories * np.random.normal(0, mutsize[0], popsize)
        ) 
        
        # adding a mutation of size mutsize[2] (genetic mutation size), rate mutate_memories, to the genotype

        offspring[:,dims['geno']] = (
                            offspring[:,dims['geno']] 
                            + mutate_geno * np.random.normal(0, mutsize[1], size=(popsize, nof_dimensions) )
        )
        
        
        # Construct the phenotype
        
        if t % 10000 == 0:
            print(f"Gen {t}: Running EPIGENETIC mode")
            
        for i in range(0, nof_dimensions):
                    offspring[:, dims['pheno'][i]] = (
                        offspring[:, dims['geno'][i]] #genotype
                        + np.random.normal(0, epsilon, size=popsize) # adding random noise (developemental noise, nromally distributed with epsilon variance)
                        + (memory*(N[parents_idx, dims['pheno'][i]] - N[parents_idx, dims['geno'][i]]))# adding the deviation -> read as weight (phenotypic component * deviation)
                    )
        N = offspring
        
        
    return {                ## Make sure this is indented!!
        'rho_m_alpha_beta' :rho_m_alpha_beta,
        'maxgen_popsize': [maxgen,popsize],
        'meanw': meanw,
        'meanmemory_p': meanmemory_p,
        'meanneutral_p': meanneutral_p

        }
        




    # ---------------
    # Set params for simulation
    #-----------------

# get the cube dimensions
dims = get_cube_dims(nof_dimensions)

N = create_cube(popsize, dims=dims, nof_dimensions=nof_dimensions, epsilon=epsilon, initial_memory=initial_memory)

print("no_dims", nof_dimensions)
print("case (0-epi or 1-og in bash):", 0)


memory = 0.99
    
file_path_envs = '/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/Plot_results/clean_data_env.txt'

if __name__ == '__main__':
    
    for i in range(0, 7): 
        
            rows_per_job = 1000
            start_row = i * rows_per_job  # 200
            end_row = min(start_row + rows_per_job, 6976)  # 300 or 6976
            n_rows = end_row - start_row  # 100 (or fewer for last job)

            # Read only the needed chunk
            params_df = pd.read_csv(
                file_path_envs,
                sep='\t',
                skiprows=range(1, start_row + 1),  # skip all rows *before* start_row (header at 0)
                nrows=n_rows,
                header= None
            )
            
            seed_array = np.array(
                [secrets.randbits(32) for _ in range(params_df.shape[0])],
                dtype=np.uint32)
            
            # Create all combinations of sigma_mut and sigma_alpha
            param_grid = [
                (np.array(scenario), seed, popsize, max_gen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, False) #only genetics can either be True or false
                for (_, scenario), seed in zip(params_df.iterrows(), seed_array)
            ]

            num_cpus = 10
        
            with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor: 
                results_array = list(executor.map(run_simulation, param_grid))


            folder_path  = '/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Evolve_Transgenerational_memory/results/new_results/finegrain_100_100_env/mem_const/mem_const/'
                    

            ## Collect keys
            keys = list(results_array[0].keys())



            og = 'epi'
            filename = f'rhos{rho_cases}_ms{m_cases}_gens{max_gen}_dims{nof_dimensions}_taskid{i}'

            concatenated_results = {}

            for k in keys:
                        concatenated_results[k] = np.vstack([r[k] for r in results_array])

            df = pd.DataFrame({
                        'rho': concatenated_results['rho_m_alpha_beta'][:, 0],
                        'm': concatenated_results['rho_m_alpha_beta'][:, 1],
                        'fitness': np.mean(concatenated_results['meanw'][:, -100:], axis=1),
                        'std_fitness': np.std(concatenated_results['meanw'][:, -100:], axis=1),
                        'epi_mem_p': np.mean(concatenated_results['meanmemory_p'][:, -100:], axis=1),
                        'neutral_trait_p': np.mean(concatenated_results['meanneutral_p'][:, -100:], axis=1)
                    })
                    
            # ---- Saving (common for both branches) ----
            save_dir = os.path.join(folder_path, og)
            os.makedirs(save_dir, exist_ok=True) #make the directory if needed 

            full_path_csv = os.path.join(save_dir, filename + '.csv')

            df.to_csv(full_path_csv, index=False)
