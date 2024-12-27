import numpy as np
import matplotlib.pyplot as plt

def simulate_yield_curve(alpha, beta, sigma, L, T, dx, dt, Y0):
    """
    Simulate yield curve dynamics using a stochastic PDE.

    Parameters:
    - alpha: Diffusion coefficient
    - beta: Advection coefficient
    - sigma: Noise strength
    - L: Maximum maturity (years)
    - T: Total time (years)
    - dx: Spatial step size
    - dt: Time step size
    - Y0: Initial yield curve (function of x)

    Returns:
    - x: Maturity grid
    - t: Time grid
    - Y: Yield curve matrix
    """
    nx = int(L / dx) + 1  
    nt = int(T / dt) + 1  
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    Y = np.zeros((nt, nx))
    
    
    Y[0, :] = Y0(x)
    
    # Stability condition
    if alpha * dt / dx**2 > 0.5:
        raise ValueError("Stability condition violated: alpha * dt / dx^2 must be <= 0.5.")
    
    
    for n in range(nt - 1):
        for i in range(1, nx - 1):
            d2Y_dx2 = (Y[n, i+1] - 2 * Y[n, i] + Y[n, i-1]) / dx**2
            dY_dx = (Y[n, i+1] - Y[n, i-1]) / (2 * dx)
            stochastic_term = sigma * np.random.normal(0, np.sqrt(dt))
            
            Y[n+1, i] = Y[n, i] + dt * (alpha * d2Y_dx2 + beta * dY_dx) + stochastic_term
        
        # Boundary conditions
        Y[n+1, 0] = Y[n+1, 1]  
        Y[n+1, -1] = Y[n+1, -2] 
    
    return x, t, Y
