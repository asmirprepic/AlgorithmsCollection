import numpy as np
import matplotlib.pyplot as plt

def simulate_ou_lambda(t_grid, theta=2, mu=15, sigma=3, lam0=5):
    dt = t_grid[1] - t_grid[0]
    lam = np.zeros_like(t_grid)
    lam[0] = lam0
    for i in range(1, len(t_grid)):
        dW = np.random.normal(0, np.sqrt(dt))
        lam[i] = lam[i-1] + theta * (mu - lam[i-1]) * dt + sigma * dW
        lam[i] = max(lam[i], 0)
    return lam

def simulate_cox_conditional(lambda_t, t_grid):
    dt = t_grid[1] - t_grid[0]

    Lambda = np.sum(lambda_t) * dt

    N = np.random.poisson(Lambda)

    pdf = lambda_t / Lambda
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]

    u_samples = np.random.rand(N)
    arrival_times = np.interp(u_samples, cdf, t_grid)
    return arrival_times, Lambda
