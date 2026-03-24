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

nof_dimensions  = 1
only_gen        = 1 # 0-epi or 1-og in bash

initial_memory  =  0
   
rho_cases = 30
m_cases = 30

max_gen= 10000


results_array = {}  ### DONT FORGET TO RUN BEFORE NEW PARAM CONDITIONS

popsize = 5000

# Mutation rates: [epi-memory mutation rate, geno mutation rate]
mutrate = [0.001, 0.001]

# Mutation sizes: [epi-memory mutation size, geno mutation size]
mutsize = [1, 1]

epsilon = 0 # phenotypic noise




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
        

def run_simulation(args):
    env_param_space, popsize, maxgen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, only_genetics = args
    
    rho_m_alpha_beta = env_param_space[0:4]
    env_states = env_param_space[4:]

    meanw = np.zeros(maxgen)
    meanmemory_g = np.zeros(maxgen)
    meanmemory_p= np.zeros(maxgen)
    meanneutral_g = np.zeros(maxgen)
    meanneutral_p = np.zeros(maxgen)
    mean_gen = np.zeros(maxgen)
    stdmemory_p = np.zeros(maxgen)
    stdneutral_p = np.zeros(maxgen)
    
    for t in range(0, (maxgen-1)):
        #if t%10 == 0:
        #    print(t)
        for d in range(0, nof_dimensions):
            N[:,dims['dev'][d]] = N[:,dims['pheno'][d]] - env_states[t]
            
        dev_combined = np.sqrt(np.sum(N[:,dims['dev']]**2, axis =1))     
        W = np.exp((-dev_combined**2)/ (2))
        
        
        meanw[t] = np.mean(W)    
        
        meanmemory_g[t] = np.mean(N[:, dims['memory']])  # the actual memory
        meanmemory_p[t] = np.mean(1/(1 + np.exp(-N[:,dims['memory']]))) #with the logistc correction
        
        meanneutral_g[t] = np.mean(N[:,dims['neutral']])
        meanneutral_p[t] = np.mean(1/(1 + np.exp(-N[:,dims['neutral']])))
        mean_gen[t] = np.mean(N[:,dims['geno']])
        
        stdmemory_p[t] = np.std(1/(1 + np.exp(-N[:,dims['memory']])))
        stdneutral_p[t] = np.std(1/(1 + np.exp(-N[:,dims['neutral']])))
        
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
                                + mutate_memories * np.random.# The `normal` function in NumPy
                                # generates random numbers from a normal
                                # (Gaussian) distribution. It takes
                                # three parameters: `loc` (mean of the
                                # distribution), `scale` (standard
                                # deviation of the distribution), and
                                # `size` (output shape).
                                normal(0, mutsize[0], popsize)
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
        
        
    return {                ## Make sure this is indented!!
        'maxgen_popsize': [maxgen,popsize],
        'rho_m_alpha_beta' :rho_m_alpha_beta,
        'meanw': meanw,
        'meanmemory_g': meanmemory_g,
        'meanmemory_p': meanmemory_p,
        'meanneutral_g': meanneutral_g,
        'meanneutral_p': meanneutral_p,
        'mean_geno': mean_gen,
        'std_memory_p': stdmemory_p,
        'std_neutral_p': stdneutral_p
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
  

if __name__ == '__main__':
    
   # Create all combinations of sigma_mut and sigma_alpha
    param_grid = [
        (scenario, popsize, max_gen, dims, N, mutsize, mutrate, epsilon, nof_dimensions, only_gen) #only genetics can either be True or false
        for _, scenario in enumerate(clean_data_env) #where each row corresponds to one scenario
    ]

    num_cpus = 10
    
    with concurrent.futures.ProcessPoolExecutor(num_cpus) as executor:

        result_array = list(executor.map(run_simulation, param_grid))
        #If disk output is all you need, then don’t collect results:No list(...) needed.


folder_path  = '/home/usriniva/Desktop/Phd/Epigenetics/Simulations/New_Hanna_eqns/Hanna_simu/results/new_results/env_off'
    
print("no_dims", nof_dimensions)
print("case:", only_gen)
       
## Collect keys
keys = list(result_array[0].keys())

if only_gen:
    og = 'onlygen'
    filename = f'rhos{rho_cases}_ms{m_cases}_gens{max_gen}_dims{nof_dimensions}_mutrate{mutrate[0]}_onlygen_envoff'

    
else:
    og = 'epi'
    filename = f'rhos{rho_cases}_ms{m_cases}_gens{max_gen}_dims{nof_dimensions}_mutrate{mutrate[0]}_envoff'

# ---- Saving (common for both branches) ----
save_dir = os.path.join(folder_path, og)
os.makedirs(save_dir, exist_ok=True)

full_path_pkl = os.path.join(save_dir, filename + '.pkl')


with open(full_path_pkl, 'wb') as f:
    pickle.dump(result_array, f)


