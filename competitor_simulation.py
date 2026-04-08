import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import chi2
from joblib import Parallel, delayed
from tqdm import tqdm
# Import your algorithms (make sure to use the vectorized LGR)
from src.algorithms import run_complete_randomization, run_acceptance_rejection, run_psrr, run_brain, run_lgr, run_lgr_adaptive, run_lgr_normalized, run_lgr_barrier
# Note: Assuming you added the function above to src.algorithms or defined it locally
# from src.algorithms import run_lgr_vectorized as run_lgr 
from src.quadratic_form import quadratic_form_distance
from src.data import generate_data
from src.estimator import DifferenceInMeans
from src.threshold import DefineThreshold
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Dimensions to test (The X-axis)
# We stop Rejection sampling early because it will hang forever at high d
d_values = [10,50,100,150,200,250]
# d_values = [2,3,4,5,6,7,8,9,10]  
B = 1000  # Number of repetitions per dimension
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.5 # True ATE
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperature = 0.5  # LGR temperature - [0.01, 0.1, 0.5, 1.0, 10.0]
eta = 0.01  # LGR eta parameter [0.01, 0.1, 1.0, 10.0]
linear = True # linear DGP

def run_CR_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)

    # Store results
    row = {'d': d}

    # 0. Complete Randomization Baseline
    start_time = time.time()
    Z, success, iters, _ = run_complete_randomization(n, n1)
    row['CR_Time'] = time.time() - start_time
    row['CR_Iter'] = iters
    row['CR_Bias'] = tau-DifferenceInMeans(Y0=Y0, Y1=Y1, Z=Z)
    row['CR_Distance'] = quadratic_form_distance(Z, X, A)

    return row

def run_ARR_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)
    
    # Store results
    row = {'d': d}

    # 1. Rejection Sampling
    start_time = time.time()
    Z, success, iters, M = run_acceptance_rejection(X, n1, a, A, max_iter=100000)
    row['ARR_Time'] = time.time() - start_time
    row['ARR_Iter'] = iters
    row['ARR_Bias'] = tau-DifferenceInMeans(Y0=Y0, Y1=Y1, Z=Z)
    row['ARR_Distance'] = quadratic_form_distance(Z, X, A)

    return row

def run_PSRR_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)
    
    # Store results
    row = {'d': d}

    # 2. PSRR
    start_time = time.time()
    Z, success, iters, M = run_psrr(X, n1, a, A, max_iter=100000)
    row['PSRR_Time'] = time.time() - start_time
    row['PSRR_Iter'] = iters
    row['PSRR_Bias'] = tau-DifferenceInMeans(Y0=Y0, Y1=Y1, Z=Z)
    row['PSRR_Distance'] = quadratic_form_distance(Z, X, A)

    return row

def run_BRAIN_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)
    
    # Store results
    row = {'d': d}

    # 3. BRAIN
    start_time = time.time()
    Z, success, iters, M = run_brain(X, n1, a, A, max_iter=100000)
    row['BRAIN_Time'] = time.time() - start_time
    row['BRAIN_Iter'] = iters
    row['BRAIN_Bias'] = tau-DifferenceInMeans(Y0=Y0, Y1=Y1, Z=Z)
    row['BRAIN_Distance'] = quadratic_form_distance(Z, X, A)

    return row

def run_LGR_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)
    
    # Store results
    row = {'d': d}

    # 4. LGR
    start_time = time.time()
    Z, success, iters, M = run_lgr(X, n1, a, A, max_iter=100000, *kwargs)
    row['LGR_Time'] = time.time() - start_time
    row['LGR_Iter'] = iters
    row['LGR_Bias'] = tau-DifferenceInMeans(Y0=Y0, Y1=Y1, Z=Z)
    row['LGR_Distance'] = quadratic_form_distance(Z, X, A)

    return row



# ==========================================
# EXECUTION
# ==========================================

file = f'results/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pkl'

try:
    df = pd.read_pickle(file)
    print(f"Loaded existing results from {file}")
except FileNotFoundError:
    print("No existing results found. Running simulations...")
    print(f"Running Benchmark across dimensions: {d_values} - Temperature: {temperature} - Eta: {eta} - Distance Metric: {distance_metric}")
    tasks = []
    for d in d_values:

        X, Y0, Y1 = generate_data(n, d, tau, linear)
        if distance_metric == "euclidean":
            A = np.eye(d)
        elif distance_metric == "mahalanobis":
            A = np.linalg.inv(np.cov(X.T) * (n / (n1 * (n - n1))))
        a = DefineThreshold(X, A, p_val, n1)


        for b in range(B):
            tasks.append((X, A, Y1, Y0, a, d, b, temperature, eta))  # Additional args for LGR

    print("Running CR simulations...")
    results_CR = Parallel(n_jobs=n_jobs)(
        delayed(run_CR_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )

    print("Running ARR simulations...")
    results_ARR = Parallel(n_jobs=n_jobs)(
        delayed(run_ARR_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )

    print("Running PSRR simulations...")
    results_PSRR = Parallel(n_jobs=n_jobs)(
        delayed(run_PSRR_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )

    print("Running BRAIN simulations...")
    results_BRAIN = Parallel(n_jobs=n_jobs)(
        delayed(run_BRAIN_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )

    print("Running LGR simulations...")
    results_LGR = Parallel(n_jobs=n_jobs)(
        delayed(run_LGR_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )

    
    df_CR = pd.DataFrame(results_CR)
    df_ARR = pd.DataFrame(results_ARR)
    df_PSRR = pd.DataFrame(results_PSRR)
    df_BRAIN = pd.DataFrame(results_BRAIN)
    df_LGR = pd.DataFrame(results_LGR)

    df = pd.concat([df_CR, df_ARR.drop(columns=['d']), df_PSRR.drop(columns=['d']), df_BRAIN.drop(columns=['d']), df_LGR.drop(columns=['d'])], axis=1)
    pd.to_pickle(df, f'results/competitor_d{d_values}_metric{distance_metric}_pval{p_val}_n{n}_temperature{temperature}_eta{eta}_linear{linear}.pkl')



# SUMMARY STATISTICS

summary_df = df.groupby('d').agg({
    'CR_Time': 'mean',
    'CR_Iter': 'mean',
    'CR_Bias': 'mean',
    'CR_Distance': 'mean',
    'ARR_Time': 'mean',
    'ARR_Iter': 'mean',
    'ARR_Bias': 'mean',
    'ARR_Distance': 'mean',
    'PSRR_Time': 'mean',
    'PSRR_Iter': 'mean',
    'PSRR_Bias': 'mean',
    'PSRR_Distance': 'mean',
    'BRAIN_Time': 'mean',
    'BRAIN_Iter': 'mean',
    'BRAIN_Bias': 'mean',
    'BRAIN_Distance': 'mean',
    'LGR_Time': 'mean',
    'LGR_Iter': 'mean',
    'LGR_Bias': 'mean',
    'LGR_Distance': 'mean',
}).reset_index()

print("Average Time by Dimension:")
print(summary_df[['d', 'CR_Time', 'ARR_Time', 'PSRR_Time', 'BRAIN_Time', 'LGR_Time']])

print("\nAverage Iterations by Dimension:")
print(summary_df[['d', 'CR_Iter', 'ARR_Iter', 'PSRR_Iter', 'BRAIN_Iter', 'LGR_Iter']])
