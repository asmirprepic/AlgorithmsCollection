import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


D = 0.1       # Diffusion coefficient
alpha = 0.5   # Growth rate
K = 1.0       # Carrying capacity
L = 50        # Number of assets (spatial grid points)
T = 1.0       # Time horizon (years)
dx = 1.0      # Spatial step size
dt = 0.001    # Time step size
steps = int(T / dt)  # Total time steps


v = np.zeros(L)
v[L//2] = 0.5  # Initial volatility localized in the center


def simulate_reaction_diffusion(v, D, alpha, K, dx, dt, steps):
    v_evolution = [v.copy()]
    for _ in range(steps):
        laplacian = (np.roll(v, -1) - 2*v + np.roll(v, 1)) / dx**2
        reaction = alpha * v * (1 - v / K)
        v += dt * (D * laplacian + reaction)
        v_evolution.append(v.copy())
    return np.array(v_evolution)

# Simulate PDE
v_evolution = simulate_reaction_diffusion(v, D, alpha, K, dx, dt, steps)

plt.figure(figsize=(10, 6))
plt.imshow(v_evolution.T, aspect='auto', origin='lower', cmap='hot',
           extent=[0, T, 0, L])
plt.colorbar(label="Volatility")
plt.xlabel("Time (years)")
plt.ylabel("Assets")
plt.title("Spatial and Temporal Evolution of Volatility")
plt.show()

def logistic_growth(v, alpha, K):
    return alpha * v * (1 - v / K)

# Estimate Parameters from Simulated Data
def estimate_parameters(v_evolution, dt):
    time_series = v_evolution[:, L//2]  # Extract volatility at the center asset
    dv_dt = np.gradient(time_series, dt)  # Time derivative of volatility
    
    # Fit the logistic growth model
    popt, _ = curve_fit(logistic_growth, time_series, dv_dt, p0=[0.5, 1.0])
    return popt
