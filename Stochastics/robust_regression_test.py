import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, HuberRegressor, TheilSenRegressor
from tqdm import tqdm

# Simulation settings
np.random.seed(42)
n_sim = 500
n = 100
contamination_levels = np.linspace(0, 0.3, 10)  # 0% to 30% contamination

def simulate_location_estimators(contamination):
    true_value = 0
    mean_estimates, median_estimates = [], []
    
    for _ in range(n_sim):
        x = np.random.normal(loc=true_value, scale=1, size=n)
        n_contaminated = int(contamination * n)
        if n_contaminated > 0:
            x[:n_contaminated] = 10  # Large outliers

        mean_estimates.append(np.mean(x))
        median_estimates.append(np.median(x))
    
    def mse(estimates):
        return np.mean((np.array(estimates) - true_value)**2)

    return {
        "contamination": contamination,
        "mean_mse": mse(mean_estimates),
        "median_mse": mse(median_estimates),
        "mean_bias": np.mean(mean_estimates) - true_value,
        "median_bias": np.mean(median_estimates) - true_value
    }
