import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def simulate_fractional_poisson(T, alpha, lam, n_paths=1, dt=0.01, seed=42):
    """
    Simulates paths from a fractional Poisson process using thinning approach.

    Parameters:
    - T: total time
    - alpha: fractional order (0 < alpha <= 1)
    - lam: rate parameter
    - n_paths: number of paths
    - dt: discretization step
    """
    np.random.seed(seed)
    N = int(T / dt)
    t_grid = np.linspace(0, T, N)
    paths = np.zeros((n_paths, N))

    for p in range(n_paths):
        count = 0
        for i, t in enumerate(t_grid):

            prob = lam * dt**alpha / gamma(1 + alpha)
            if np.random.rand() < prob:
                count += 1
            paths[p, i] = count
    return t_grid, paths
