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
from src.data import generate_data
from src.estimator import DifferenceInMeans
from src.threshold import DefineThreshold

# ==========================================
# CONFIGURATION
# ==========================================
# Dimensions to test (The X-axis)
# We stop Rejection sampling early because it will hang forever at high d
d_values = [5, 10, 50, 100, 150, 200, 250] 
d_values = [10, 50, 100, 150, 200, 250] 
B = 1000  # Number of repetitions per dimension
B_frt = 100  # Number of FRT permutations
n = 500 # Total units
n1 = 250 # Treated units
p_val = 0.01 # Top 1% balance
n_jobs = -2  # Parallel cores
tau = 0.5 # True ATE
alpha = 0.05 # Significance level
distance_metric = "mahalanobis" # "mahalanobis" or "euclidean"
temperature = 0.5  # LGR temperature
eta = 1.0  # LGR eta parameter
linear = True # linear DGP


def run_frt_test(Y_obs, Z_obs, X, func_algo, algo_args, null_tau=0, B_frt=2):
    """
    Runs the Fisher Randomization Test for a specific null hypothesis (tau = null_tau).
    
    Returns:
        p_value: The two-sided p-value.
    """
    # 1. Impute potential outcomes under the sharp null H0: Y_i(1) = Y_i(0) + null_tau
    # If the null is true, then Y_i(0) is fixed regardless of Z.
    # We recover Y_i(0) from the observed data assuming the null is true.
    # Y_obs = Z*Y(1) + (1-Z)*Y(0)
    # Under H0: Y(1) = Y(0) + null_tau
    # Y_obs = Z*(Y(0) + null_tau) + (1-Z)*Y(0) = Y(0) + Z*null_tau
    # Therefore: Y_null_response (which is Y(0)) = Y_obs - Z_obs * null_tau
    
    Y_null_response = Y_obs - Z_obs * null_tau
    
    # Calculate the observed statistic (Difference in Means)
    # Note: We compute the diff in means on the *observed* Y, testing against the null distribution
    # Actually, standard FRT compares T(Z_obs, Y_obs) vs T(Z_perm, Y_imputed)
    # But usually we just fix the potential outcomes Y_null_response and permute Z.
    
    # Stat: Difference in means on the adjusted residuals (or raw outcomes? Standard is residuals)
    # Let's stick to the simplest: Diff in Means of Y_obs - Z_obs*null_tau (which is Y(0))
    # If H0 is true, the treatment effect on Y(0) is 0. 
    # So we compute DiffMeans(Y_null_response, Z) and check if it deviates from 0.
    
    obs_stat = DifferenceInMeans(Y0=Y_null_response, Y1=Y_null_response, Z=Z_obs)
    
    null_stats = []
    
    # 2. FRT Loop
    # We do NOT run this in parallel here if the outer loop is already parallel
    for _ in range(B_frt):
        # Generate a new balanced assignment using the SAME algorithm
        # We unpack algo_args to pass X, n1, a, etc.
        # Ensure 'algo_args' contains: (X, n1, a, A)
        
        # Note: We don't care about the other returns (t, success, etc)
        Z_null, _, _, _, _ = func_algo(*algo_args)
        
        # Calculate statistic on the FIXED null potential outcomes
        stat_null = DifferenceInMeans(Y0=Y_null_response, Y1=Y_null_response, Z=Z_null)
        null_stats.append(stat_null)
        
    null_stats = np.array(null_stats)
    
    # 3. Compute P-value (Two-sided)
    # P = Mean( |T_null| >= |T_obs| )
    p_val = np.mean(np.abs(null_stats) >= np.abs(obs_stat))
    
    return p_val


def run_simulation_step(X, A, Y1, Y0, a, d, seed, *kwargs):
    np.random.seed(seed)
    
    row = {'d': d}
    
    # Algorithms
    methods = [
        ('CR', run_complete_randomization),
        # ('Rejection', run_acceptance_rejection),
        # ('PSRR', run_psrr),
        ('BRAIN', run_brain),
        ('LGR', run_lgr)
    ]
    
    for name, func in methods:
        # Prepare arguments tuple for the FRT function
        # Arguments expected by algorithms: (X, n1, a, A, max_iter)
        algo_args = (X, n1, a, A) 
        if name == 'LGR':
            algo_args += (kwargs)  #  any extra kwargs like temperature
        elif name == 'CR':
            algo_args = (n, n1)  # CR needs n instead of X
        
        # --- 1. Run Algorithm to get Observation ---
        # Note: LGR/Rejection might need max_iter passed directly or handled in wrapper
        # Assuming your functions accept *algo_args
        Z, t, success, iters, M = func(*algo_args)
        
        # Observed Outcome
        Y_obs = Y1 * Z + Y0 * (1 - Z)
        
        # Basic Stats
        est = DifferenceInMeans(Y0=Y_obs, Y1=Y_obs, Z=Z) # Naive diff in means
        row[f'{name}_Time'] = t
        row[f'{name}_Iter'] = iters
        row[f'{name}_Dist'] = M
        
        # --- 2. FRT for COVERAGE (Test H0: tau = true_tau) ---
        # If p > alpha, we DO NOT reject the truth -> Covered = True        
        # Check Coverage: Test H0: tau = true tau
        p_val_cov = run_frt_test(Y_obs, Z, X, func, algo_args, null_tau=tau, B_frt=B_frt)
        row[f'{name}_Covered'] = 1 if p_val_cov > alpha else 0
        
        # --- 3. FRT for POWER (Test H0: tau = 0) ---
        # If p < alpha, we REJECT the null -> Power = 1
        p_val_power = run_frt_test(Y_obs, Z, X, func, algo_args, null_tau=0, B_frt=B_frt)
        row[f'{name}_Power'] = 1 if p_val_power < alpha else 0

    return row

# ==========================================
# EXECUTION
# ==========================================

file = f'results/coverage_d{d_values}_metric{distance_metric}_B{B}_Bfrt{B_frt}_pval{p_val}_temperature{temperature}_eta{eta}_linear{linear}.pkl'


try:
    df = pd.read_pickle(file)
    print(f"Loaded existing results from {file}")
except FileNotFoundError:
    print("File not found, running new simulations...")
    print(f"Running Benchmark across dimensions: {d_values} - Temperature: {temperature} - Eta: {eta} - Distance Metric: {distance_metric}")
    tasks = []
    for d in d_values:

        X, Y0, Y1 = generate_data(n, d, tau, linear)
        if distance_metric == "euclidean":
            A = np.eye(d) * (n / (n1 * (n - n1)))
        elif distance_metric == "mahalanobis":
            A = np.linalg.inv(np.cov(X.T) * (n / (n1 * (n - n1))))
        a = DefineThreshold(X, A, p_val, n1)

        for b in range(B):
            tasks.append((X, A, Y1, Y0, a, d, b, temperature, eta)),

    results_list = Parallel(n_jobs=n_jobs)(
        delayed(run_simulation_step)(X, A, Y1, Y0, a, d, 2 + d*1000 + b, temperature, eta) for X, A, Y1, Y0, a, d, b, temperature, eta in tqdm(tasks)
    )
    
    df = pd.DataFrame(results_list)
    pd.to_pickle(df, f'results/coverage_d{d_values}_metric{distance_metric}_B{B}_Bfrt{B_frt}_pval{p_val}_temperature{temperature}_eta{eta}_linear{linear}.pkl')