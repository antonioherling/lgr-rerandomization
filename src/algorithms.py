import numpy as np
from scipy.special import expit  # Logistic function
from src.quadratic_form import quadratic_form_distance
import time

def run_complete_randomization(n, n1):
    
    Z = np.zeros(n)
    Z[np.random.choice(n, n1, replace=False)] = 1
    return Z, True, 1, None

def run_acceptance_rejection(X, n1, a, A, max_iter=3000):
    n = X.shape[0]

    for iter in range(max_iter):
        Z = np.zeros(n)
        Z[np.random.choice(n, n1, replace=False)] = 1
        M = quadratic_form_distance(Z, X, A)
        if M <= a:
            return Z, True, iter+1, M
    return Z, False, max_iter, M

def run_psrr(X, n1, a, A, max_iter=3000, gamma = 10.0):
    """Pair-Switching Rerandomization (PSRR) - simplified search """
    n = X.shape[0]
    
    # Start with random Z
    Z = np.zeros(n)
    Z[np.random.choice(n, n1, replace=False)] = 1
    
    for t in range(max_iter):
        M = quadratic_form_distance(Z, X, A)
        if M <= a:
            return Z, True, t+1, M
        
        # Switch one treated and one control unit
        treated_idx = np.where(Z == 1)[0]
        control_idx = np.where(Z == 0)[0]
        i = np.random.choice(treated_idx)
        j = np.random.choice(control_idx)
        
        # Propose swap
        Z[i], Z[j] = 0, 1 
        Mstar = quadratic_form_distance(Z, X, A)
        # if Mstar <= M:
        #     continue  # Accept swap
        # else:  # Accept swap with some probability
        #     if np.random.uniform(0, 1) < (M / Mstar)**gamma: 
        #         continue # Accept swap
        #     else:
        #         Z[i], Z[j] = 1, 0 # Revert swap

        J = np.min([(M / Mstar)**gamma, 1]) # Acceptance probability
        if np.random.uniform(0, 1) < J: 
            continue # Accept swap
        else:
            Z[i], Z[j] = 1, 0 # Revert swap

    return Z, False, max_iter, M

def run_brain(X, n1, a, A, max_iter=3000, L=None, S=None):
    """
    BRAIN algorithm (Lu et al., 2025)
    A optimization method using iterative refinement.
    """
    n, d = X.shape

    if L is None: # Default number of swaps to consider per iteration
        L = n//2  
    if S is None: # Default number of random starts
        S = 1    
    
    # Initialize with a random allocation
    Z = np.zeros(n)
    Z[np.random.choice(n, n1, replace=False)] = 1
    
    for t in range(max_iter):
        # Check current balance
        M = quadratic_form_distance(Z, X, A)
        if M <= a:
            return Z, True, t+1, M
        
        # Iterative Refinement: 
        # Identify the pair swap that yields the greatest decrease in M
        # For efficiency in simulations, we look for the best swap among a subset
        treated_indices = np.where(Z == 1)[0]
        control_indices = np.where(Z == 0)[0]
        Stilde = S

        # Greedy search for a swap that improves balance
        for _ in range(L): # Sample swaps for speed
            i = np.random.choice(treated_indices)
            j = np.random.choice(control_indices)
            
            Z[i], Z[j] = 0, 1
            Mstar = quadratic_form_distance(Z, X, A)
            if Mstar < M:
                Stilde = 0
                M = Mstar
                if M <= a:
                    return Z, True, t+1, M
            else:
                Z[i], Z[j] = 1, 0 # Revert swap

        if Stilde > 0:   
            for _ in range(Stilde):
                i = np.random.choice(treated_indices)
                j = np.random.choice(control_indices)
                Z[i], Z[j] = 0, 1
                M = quadratic_form_distance(Z, X, A)
        else:
            continue
            
    return Z, False, t+1, M

def run_lgr(X, n1, a, A, eta = 1, temperature=0.5,  max_iter=10000):
    """
    Langevin-Gradient Rerandomization (LGR)
    """
    n, d = X.shape 
    # theta = np.random.normal(0, 0.1, n)
    theta = np.random.normal(0, 1, n)
    
    for t in range(max_iter):
        # 1. Discrete Projection Check
        # Indices of n1 largest elements
        S = np.argsort(theta)[-n1:]
        Z = np.zeros(n)
        Z[S] = 1
        
        M = quadratic_form_distance(Z, X, A)
        if M <= a:
            return Z, True, t+1, M
        
        # 2. Soft Relaxation
        z_tilde = expit(theta / temperature)
        nt = np.sum(z_tilde)
        nc = n - nt
        
        # 3. Gradient Computation
        # Simplified gradient logic based on Eq 2
        mean_t = (X.T @ z_tilde) / nt
        mean_c = (X.T @ (1 - z_tilde)) / nc
        Delta = mean_t - mean_c
        
        g_M = 2 * A @ Delta

        # Chain rule with sigmoid derivative
        # grad_theta = np.zeros(n)
        # for i in range(n):
        #     term = (1/nt * (X[i] - mean_t) + 1/nc * (X[i] - mean_c))
        #     gamma = (1/temperature * z_tilde[i] * (1 - z_tilde[i]))
        #     grad_theta[i] = (g_M.T @ term) * gamma

        term_matrix = (1/nt * (X - mean_t) + 1/nc * (X - mean_c))
        grad_dot = term_matrix @ g_M
        gamma = (1/temperature) * z_tilde * (1 - z_tilde)
        grad_theta = grad_dot * gamma
            
        # 4. Update Step (SGLD)
        noise = np.random.normal(0, 1, n)
        theta = theta - eta*grad_theta + np.sqrt(2*eta*temperature)*noise
        
    return Z, False, max_iter, M




def run_lgr_barrier(X, n1, a, A, temperature=0.1, eta=0.01, max_iter=10000, lambda_barrier=50.0):
    """
    Langevin-Gradient Rerandomization (LGR) with Barrier Potential
    
    Parameters:
    - lambda_barrier: Strength of the "wall" when M > a. 
      Needs to be high enough to overpower the prior drift.
    - eta: Step size (learning rate).
    """
    n, d = X.shape 
    # Initialize theta from standard normal
    theta = np.random.normal(0, 1, n)
    
    start_time = time.time()
    
    for t in range(max_iter):
        # 1. Discrete Projection Check
        S = np.argsort(theta)[-n1:]
        Z = np.zeros(n)
        Z[S] = 1
        
        M_hard = quadratic_form_distance(Z, X, A)
        
        # STOPPING CRITERION:
        # If we are in the set, we are done. 
        # (In a pure sampling context, you might keep running to mix, 
        # but for rerandomization search, finding one valid Z is the goal).
        if M_hard <= a:
            return Z, time.time() - start_time, True, t+1, M_hard
        
        # 2. Soft Relaxation
        z_tilde = expit(theta / temperature)
        nt = np.sum(z_tilde)
        nc = n - nt
        
        # 3. Gradient Calculation
        
        # A. The Prior Gradient (Ornstein-Uhlenbeck Drift)
        # Gradient of U_prior = 0.5 * ||theta||^2  => grad = theta
        # This ensures we target N(0,1) distribution
        grad_total = theta.copy()
        
        # B. The Barrier Gradient (Balance Constraint)
        # Only apply if the *soft* metric violates the constraint
        mean_t = (X.T @ z_tilde) / nt
        mean_c = (X.T @ (1 - z_tilde)) / nc
        Delta = mean_t - mean_c
        M_soft = Delta.T @ A @ Delta
        
        if M_soft > a:
            # Gradient of Mahalanobis distance w.r.t theta
            g_M = 2 * A @ Delta
            
            # Chain rule terms
            term_matrix = (1/nt * (X - mean_t) + 1/nc * (X - mean_c))
            grad_dot = term_matrix @ g_M
            gamma = (1/temperature) * z_tilde * (1 - z_tilde)
            
            grad_barrier = grad_dot * gamma
            
            # Add barrier force to total gradient
            grad_total += lambda_barrier * grad_barrier
            
        # 4. Update Step (SGLD)
        # theta_{t+1} = theta_t - eta * grad_U + sqrt(2*eta) * noise
        # This relationship is strictly required for correct sampling physics.
        noise = np.random.normal(0, 1, n)
        theta = theta - eta * grad_total + np.sqrt(2 * eta) * noise
        
    return Z, time.time() - start_time, False, max_iter, M_hard

def run_lgr_adaptive(X, n1, a, A, temperature=100, eta_init=1.0, max_iter=10000, beta=0.99, epsilon=1e-8):
    """
    Langevin-Gradient Rerandomization (LGR) with Adaptive Learning Rate (RMSprop-style)
    """
    n, d = X.shape 
    theta = np.random.normal(0, 1.0, n) # Initial latent scores [cite: 75]
    
    # Adaptive learning rate state
    v = np.zeros(n) 
    
    start_time = time.time()
    for t in range(max_iter):
        # 1. Discrete Projection Check [cite: 81]
        S = np.argsort(theta)[-n1:] # Indices of n1 largest elements [cite: 84]
        Z = np.zeros(n)
        Z[S] = 1
        
        M = quadratic_form_distance(Z, X, A) # Compute Mahalanobis Distance [cite: 88]
        if M <= a:
            return Z, time.time() - start_time, True, t+1, M
        
        # 2. Soft Relaxation [cite: 78, 55]
        z_tilde = expit(theta / temperature)
        nt = np.sum(z_tilde)
        nc = n - nt
        
        # 3. Gradient Computation [cite: 107]
        mean_t = (X.T @ z_tilde) / nt
        mean_c = (X.T @ (1 - z_tilde)) / nc
        Delta = mean_t - mean_c
        
        g_M = 2 * A @ Delta # Gradient of distance w.r.t soft mean [cite: 109]

        # Chain rule with sigmoid derivative [cite: 59, 110]
        term_matrix = (1/nt * (X - mean_t) + 1/nc * (X - mean_c))
        grad_dot = term_matrix @ g_M
        gamma = (1/temperature) * z_tilde * (1 - z_tilde)
        grad_theta = grad_dot * gamma
            
        # 4. Adaptive Update Step (RMSprop + SGLD)
        # Update moving average of squared gradients
        v = beta * v + (1 - beta) * (grad_theta**2)
        
        # Adapt eta: normalize the step size by the gradient's magnitude
        eta_t = eta_init / (np.sqrt(v) + epsilon)
        
        # Langevin noise scaled by the square root of the step size
        # This ensures we sample from the correct distribution 
        noise = np.random.normal(0, np.sqrt(2 * eta_t), n)
        
        theta = theta - (eta_t / 2) * grad_theta + noise
        
    return Z, time.time() - start_time, False, max_iter, M

def run_lgr_normalized(X, n1, a, A, temperature=100, eta=1.0, max_iter=10000, epsilon=1e-8):
    """
    Langevin-Gradient Rerandomization (LGR) with Global Gradient Normalization.
    This version removes the beta parameter to avoid "memory" lag.
    """
    n, d = X.shape 
    theta = np.random.normal(0, 1.0, n)
    
    start_time = time.time()
    for t in range(max_iter):
        # 1. Discrete Projection Check
        S = np.argsort(theta)[-n1:]
        Z = np.zeros(n)
        Z[S] = 1
        
        M = quadratic_form_distance(Z, X, A)
        if M <= a:
            return Z, time.time() - start_time, True, t+1, M
        
        # 2. Soft Relaxation
        z_tilde = expit(theta / temperature)
        nt = np.sum(z_tilde)
        nc = n - nt
        
        # 3. Gradient Computation
        mean_t = (X.T @ z_tilde) / nt
        mean_c = (X.T @ (1 - z_tilde)) / nc
        Delta = mean_t - mean_c
        
        g_M = 2 * A @ Delta 

        # Chain rule with sigmoid derivative
        term_matrix = (1/nt * (X - mean_t) + 1/nc * (X - mean_c))
        grad_dot = term_matrix @ g_M
        gamma = (1/temperature) * z_tilde * (1 - z_tilde)
        grad_theta = grad_dot * gamma
            
        # 4. Global Gradient Normalization
        # We normalize the entire gradient vector by its L2 norm
        grad_norm = np.linalg.norm(grad_theta)
        
        # Normalized step: the update magnitude is controlled strictly by 'eta'
        # This prevents gradient explosion at low temperatures
        step_direction = grad_theta / (grad_norm + epsilon)
        
        # Langevin noise - now we can use a fixed scale or tie it to eta
        noise = np.random.normal(0, np.sqrt(2 * eta), n)
        
        theta = theta - eta * step_direction + noise
        
    return Z, time.time() - start_time, False, max_iter, M