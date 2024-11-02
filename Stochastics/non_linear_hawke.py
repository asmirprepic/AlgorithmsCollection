import numpy as np
import matplotlib.pyplot as plt

def power_law_kernel(t, alpha, beta, gamma):
    """
    Power-law decay kernel function for a nonlinear Hawkes process.
    
    Args:
    - t (float): Time difference (t - t_i).
    - alpha (float): Scale parameter for self-excitation.
    - beta (float): Controls the decay rate.
    - gamma (float): Shape parameter for the power-law decay.
    
    Returns:
    - float: Kernel value for the given time difference.
    """
    return alpha / (1 + beta * t) ** (1 + gamma)

def simulate_nonlinear_hawkes(mu, alpha, beta, gamma, T):
    """
    Simulate a univariate nonlinear Hawkes process with a power-law kernel.
    
    Args:
    - mu (float): Baseline intensity.
    - alpha (float): Scale of self-excitation.
    - beta (float): Decay rate for power-law kernel.
    - gamma (float): Power-law exponent.
    - T (float): End time for the simulation.
    
    Returns:
    - events (list): List of event times.
    """
    # Initialize list of events and first event time
    events = []
    t = 0
    
    while t < T:
        # Calculate the current intensity
        lambda_t = mu + sum(power_law_kernel(t - ti, alpha, beta, gamma) for ti in events)
        
        # Generate the next inter-arrival time using the current intensity
        u = np.random.uniform()
        w = -np.log(u) / lambda_t
        t = t + w
        
        # Calculate the intensity at the proposed time to accept/reje
