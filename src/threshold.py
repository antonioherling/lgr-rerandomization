# a = chi2.ppf(p_val, df=d)

import numpy as np


def DefineThreshold(X, A, p_val, n1, L=10000):
    """Define the threshold 'a' based on L random samples."""
    n, d = X.shape
    distances = []
    for _ in range(L):
        # Randomly assign n1 treated units
        Z = np.zeros(n)
        treated_indices = np.random.choice(n, n1, replace=False)
        Z[treated_indices] = 1

        # Compute Mahalanobis distance
        x_treated = X[Z == 1]
        x_control = X[Z == 0]
        mean_diff = np.mean(x_treated, axis=0) - np.mean(x_control, axis=0)
        M = mean_diff.T @ A @ mean_diff
        distances.append(M)
    
    distances = np.array(distances)
    a = np.quantile(distances, p_val)
    return a