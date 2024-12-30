import numpy as np
import matplotlib.pyplot as plt

def simulate_energy_prices(alpha, beta, gamma, L, T, dx, dt, E0):
    """
    Simulate electricity prices using nonlinear PDE.

    Parameters:
    - alpha: Speed of price diffusion
    - beta: Nonlinear price growth parameter
    - gamma: Price decay parameter
    - L: Maximum spatial domain length (e.g., regions)
    - T: Total time (years)
    - dx: Spatial step size
    - dt: Time step size
    - E0: Initial price profile (function of x)

    Returns:
    - x: Spatial grid
    - t: Time grid
    - E: Simulated electricity prices
    """
    nx = int(L / dx) + 1  # Number of spatial points
    nt = int(T / dt) + 1  # Number of time points
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    E = np.zeros((nt, nx))
    
    # Initial condition
    E[0, :] = E0(x)
    
    # Time stepping
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            dE_dx = (E[n, i+1] - E[n, i-1]) / (2 * dx)
            nonlinear_term = beta * E[n, i]**2 - gamma * E[n, i]
            
            E[n+1, i] = E[n, i] - alpha * dt * dE_dx + dt * nonlinear_term
        
        # Boundary conditions (no flux at boundaries)
        E[n+1, 0] = E[n+1, 1]
        E[n+1, -1] = E[n+1, -2]
    
    return x, t, E
