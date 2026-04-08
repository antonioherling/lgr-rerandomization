import numpy as np

def generate_data(n, d, tau, linear):
    """Generates synthetic data for experiments."""
    X = np.random.normal(0, 1, (n, d))
    if linear:
        eta = np.sum(X, axis=1)
    else:
        eta = np.sum(X**2, axis=1)

    epsilon = np.random.normal(0, 1, n)
    Y0 = eta + epsilon
    Y1 = Y0 + tau
    
    return X, Y0, Y1