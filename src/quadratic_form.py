import numpy as np

def quadratic_form_distance(Z, X, A):
    """Computes M = tau_hat' * Sigma_inv * tau_hat [cite: 47]"""
    n = len(Z)
    n1 = np.sum(Z)
    n0 = n - n1
    # Difference in means [cite: 50]
    tau_x = (X[Z==1].mean(axis=0) - X[Z==0].mean(axis=0))
    M = (tau_x.T @ A @ tau_x)

    # M = ((Z-n1/n*np.ones(n)) @ A @ (Z-n1/n*np.ones(n)).T)
    return M