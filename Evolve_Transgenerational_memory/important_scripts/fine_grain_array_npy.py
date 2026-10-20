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

nof_dimensions  = 1
only_gen        = 0# 0-epi or 1-og in bash

initial_memory  =  0
   
rho_cases = 50
m_cases = 50

max_gen= 10000


results_array = {}  ### DONT FORGET TO RUN BEFORE NEW PARAM CONDITIONS

popsize = 5000

# Mutation rates: [epi-memory mutation rate, geno mutation rate]
mutrate = [0.001, 0.001]

# Mutation sizes: [epi-memory mutation size, geno mutation size]
mutsize = [1, 1]

epsilon = 0.05 # phenotypic noise


#split into 'scenarios' chunks


task_id = int(sys.argv[1])
chunks = np.load("/home/usriniva/Epi_kok/chunks_10K_7_chunks.npy", allow_pickle=True)

array_scenarios_chunk = chunks[task_id]


#array_scenarios_chunk = chunks[task_id]


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

    Parameters:
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


def create_world(rho_cases, m_cases, max_gen):
    

    # Define rho and m grid
    rho = np.linspace(-0.99, 0.99, rho_cases)
    m = np.linspace(0.51, 0.99, m_cases)

    RHO, M = np.meshgrid(rho, m)

    # Define ALPHA and BETA
    ALPHA = (1 - RHO) * (1 - M)
    BETA = (1 - RHO) * M

    # Flatten and create DataFrame
    data_env = np.column_stack((RHO.flatten(),
                                M.flatten(),
                                ALPHA.flatten(),
                                BETA.flatten()))

    n_scenarios= len(RHO.flatten())
    n_gen = max_gen

    opt = np.array([0, 1])
    Env = np.zeros((n_scenarios, n_gen), dtype=int) #same as hannas E()

    p = np.random.uniform(low=0, high=1, size=(n_scenarios, n_gen))
    alpha = ALPHA.flatten()
    beta =  ALPHA.flatten()

    for g in range(0, (n_gen-1)):
        
        Env[:, g+1] = Env[:, g]
        
        # Transitions
        cAB = (Env[:, g] == 0) & (p[:,g] < alpha)  #use boolean operator & to ask; which rows have a0 and
        cBA = (Env[:, g] == 1) & (p[:,g] < beta)

        Env[cAB, g+1] = 1  # A→B transitions
        Env[cBA, g+1] = 0  # B→A transitions
        
    env_values = opt[Env]  # final result
    
    full_data_env_env_values = np.concatenate((data_env,env_values),axis=1)
    
    # Mask where ALPHA > 1 or BETA > 1
    mask = (full_data_env_env_values[:, 2] > 1) | (full_data_env_env_values[:, 3] > 1)

    # Set ALPHA and BETA (columns 2 and 3) to NaN where mask is True
    full_data_env_env_values[mask, 2:] = np.nan


    # Equilibrium stats
    return full_data_env_env_values
        

def run_simulation(params):
    scenarios, seed, popsize, maxgen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, only_genetics = params
    
    rho_m_alpha_beta = scenarios[0:4]
    env_states = scenarios[4:]
    
    np.random.seed(seed)
    print(f"Using seed: {seed}")
    

    meanw = np.zeros(maxgen)
    meanw0 = np.zeros(maxgen)
    meanw1 = np.zeros(maxgen)
    meanmemory_g = np.zeros(maxgen)
    meanmemory_p= np.zeros(maxgen)
    meanneutral_g = np.zeros(maxgen)
    meanneutral_p = np.zeros(maxgen)
    mean_gen = np.zeros(maxgen)
    
    for t in range(0, (maxgen)):
        #if t%10 == 0:
        #    print(t)
        for d in range(0, nof_dimensions):
            N[:,dims['dev'][d]] = N[:,dims['pheno'][d]] - env_states[t]
            
        
        dist_from_optima = np.sqrt(np.sum(N[:,dims['dev']]**2, axis =1))
        
        #------distance correction ------
        def between_optima_dist(dims):
            multi_d_diff = np.ones(dims) - np.zeros(dims)
            
            return np.sqrt(np.sum(multi_d_diff**2))
        #-----------------------------
        
        normalised_dist = dist_from_optima/between_optima_dist(nof_dimensions) 
        
        #Distances from either optima
        Dist0 = np.sqrt(np.sum((N[:,dims['pheno']] - 0)**2, axis = 1))/between_optima_dist(nof_dimensions)
        Dist1 = np.sqrt(np.sum((N[:,dims['pheno']] - 1 )**2, axis =1 )) /between_optima_dist(nof_dimensions)
        
        W = np.exp((-normalised_dist**2)/ (2))
        W0 = np.exp((-Dist0**2)/ (2))
        W1 = np.exp((-Dist1**2)/ (2))
        
        meanw[t] = np.mean(W)    
        meanw0[t] = np.mean(W0)
        meanw1[t] = np.mean(W1)
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
        if only_genetics :
            if t % 10000 == 0:
                print(f"Gen {t}: Running ONLY genetics mode")
            
            for i in range(0, nof_dimensions):
                offspring[:, dims['pheno'][i]] = (
                offspring[:, dims['geno'][i]]
                + np.random.normal(0, epsilon, size=popsize) )
        else:
            if t % 10000 == 0:
                print(f"Gen {t}: Running EPIGENETIC mode")
            
            for i in range(0, nof_dimensions):
                    offspring[:, dims['pheno'][i]] = (
                        offspring[:, dims['geno'][i]] #genotype
                        + np.random.normal(0, epsilon, size=popsize) # adding random noise (developemental noise, nromally distributed with epsilon variance)
                        + ((1 / (1 + np.exp(-offspring[:, dims['memory']]))) #phenotype version of the memory (y axis) 
                        * (N[parents_idx, dims['pheno'][i]] - N[parents_idx, dims['geno'][i]]))# adding the deviation -> read as weight (phenotypic component * deviation)
                    )
        N = offspring
        
        
    return {
        'rho_m_alpha_beta': rho_m_alpha_beta,
        'maxgen_popsize': [maxgen, popsize],
        'meanw': meanw,
        'meanw0': meanw0,
        'meanw1': meanw1,
        'meanmemory_p': meanmemory_p,
        'meanneutral_p': meanneutral_p
    }
        
        

# ---------------
# Set params for simulation
#-----------------

full_env_values = create_world(rho_cases, m_cases, max_gen)   # create world (input: generations, no. rhos and no. ms) 

clean_data_env = full_env_values[~np.isnan(full_env_values).any(axis=1)]        
    
  
# get the cube dimensions
dims = get_cube_dims(nof_dimensions)

N = create_cube(popsize, dims=dims, nof_dimensions=nof_dimensions, epsilon=epsilon, initial_memory=initial_memory)

print("no_dims", nof_dimensions)
print("case (0-epi or 1-og in bash):", only_gen)
print("task_id", task_id)


seed_array = np.array(
    [secrets.randbits(32) for _ in range(array_scenarios_chunk.shape[0])],
    dtype=np.uint32
)


#seed_array = np.linspace(2000, 12000, num=array_scenarios_chunk.shape[0], dtype=int)


if __name__ == '__main__':
    
    # Create all combinations of sigma_mut and sigma_alpha
    param_grid = [
        (scenario, seed, popsize, max_gen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, False) #only genetics can either be True or false
        for scenario, seed in zip(array_scenarios_chunk, seed_array) #where each row corresponds to one scenario
    ]

    num_cpus = 15
    
    
    with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor: 
        results_array = list(executor.map(run_simulation, param_grid))


folder_path  = '//home/usriniva/Epi_kok/newseed_dim1'
    

## Collect keys
keys = list(results_array[0].keys())

if only_gen:
    og = 'onlygen'
    filename = f'rhos{rho_cases}_ms{m_cases}_gens{max_gen}_dims{nof_dimensions}_taskid{task_id}_onlygen'

    concatenated_results = {}
    for k in keys:
        concatenated_results[k] = np.vstack([r[k] for r in results_array])

    df = pd.DataFrame({
        
    'rho': concatenated_results['rho_m_alpha_beta'][:,0],
    'm': concatenated_results['rho_m_alpha_beta'][:,1],
    'fitness':np.mean(concatenated_results['meanw'][:,-100:], axis=1),
    'fitness0': np.mean(concatenated_results['meanw0'][:, -100:], axis=1),
    'fitness1': np.mean(concatenated_results['meanw1'][:, -100:], axis=1),
    'std_fitness': np.std(concatenated_results['meanw'][:, -100:], axis=1)
    })

else:
    og = 'epi'
    filename = f'rhos{rho_cases}_ms{m_cases}_gens{max_gen}_dims{nof_dimensions}_taskid{task_id}'

    concatenated_results = {}
    for k in keys:
        concatenated_results[k] = np.vstack([r[k] for r in results_array])

    df = pd.DataFrame({
        'rho': concatenated_results['rho_m_alpha_beta'][:, 0],
        'm': concatenated_results['rho_m_alpha_beta'][:, 1],
        'fitness': np.mean(concatenated_results['meanw'][:, -100:], axis=1),
        'fitness0': np.mean(concatenated_results['meanw0'][:, -100:], axis=1),
        'fitness1': np.mean(concatenated_results['meanw1'][:, -100:], axis=1),
        'epi_mem_p': np.mean(concatenated_results['meanmemory_p'][:, -100:], axis=1),
        'neutral_trait_p': np.mean(concatenated_results['meanneutral_p'][:, -100:], axis=1),
        'std_fitness': np.std(concatenated_results['meanw'][:, -100:], axis=1)
    })
    
# ---- Saving (common for both branches) ----
save_dir = os.path.join(folder_path, og)
os.makedirs(save_dir, exist_ok=True) #make the directory if needed 

full_path_csv = os.path.join(save_dir, filename + '.csv')

df.to_csv(full_path_csv, index=False)
