import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def simulate_cir(T, dt, kappa, theta, sigma, lambda_0):
    """
    Simulate a CIR process with given parameters.
    """
    n_steps = int(T / dt)
    intensity = np.zeros(n_steps)
    intensity[0] = lambda_0
    
    for t in range(1, n_steps):
        dW = np.sqrt(dt) * np.random.normal()
        intensity[t] = max(intensity[t-1] + kappa * (theta - intensity[t-1]) * dt + sigma * np.sqrt(intensity[t-1]) * dW, 0)
    
    return intensity

def log_likelihood(params, intensity, dt):
    """
    Compute the negative log-likelihood for the CIR model given observed intensity.
    
    Args:
    - params (list): Parameters [kappa, theta, sigma] to estimate.
    - intensity (np.array): Observed intensity values.
    - dt (float): Time increment.
    
    Returns:
    - float: Negative log-likelihood for MLE.
    """
    kappa, theta, sigma = params
    log_likelihood_sum = 0.0
    
    for t in range(1, len(intensity)):
        mu = intensity[t-1] + kappa * (theta - intensity[t-1]) * dt
        var = sigma**2 * intensity[t-1] * dt
        if var > 0:
            log_likelihood_sum += -0.5 * np.log(2 * np.pi * var) - 0.5 * ((intensity[t] - mu)**2 / var)
    
    return -log_likelihood_sum  # Return negative for minimization
