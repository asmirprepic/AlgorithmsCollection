import numpy as np
import matplotlib.pyplot as plt

def simulate_wave(c, L, T, dx, dt, initial_profile, boundary_conditions):
    """
    Simulate information diffusion using the wave equation.

    Parameters:
    - c: Speed of information flow
    - L: Length of the domain (e.g., number of sectors)
    - T: Total time for simulation
    - dx: Spatial step size
    - dt: Time step size
    - initial_profile: Function defining the initial information profile u(x, 0)
    - boundary_conditions: Tuple (u_left, u_right), fixed values at boundaries

    Returns:
    - x: Spatial grid
    - t: Temporal grid
    - u: Simulated information levels
    """
    nx = int(L / dx) + 1
    nt = int(T / dt) + 1
    x = np.linspace(0, L, nx)
    t = np.linspace(0, T, nt)
    
    
    if c * dt / dx > 1:
        raise ValueError("Stability condition violated: c * dt / dx must be <= 1.")
    
    # Initialize solution grid
    u = np.zeros((nt, nx))
    
    # Set initial condition
    u[0, :] = initial_profile(x)
    
    
    for i in range(1, nx - 1):
        u[1, i] = u[0, i] + 0.5 * (c * dt / dx)**2 * (u[0, i+1] - 2*u[0, i] + u[0, i-1])
    
    
    u[:, 0] = boundary_conditions[0]
    u[:, -1] = boundary_conditions[1]
    
    
    for n in range(1, nt - 1):
        for i in range(1, nx - 1):
            u[n+1, i] = 2 * u[n, i] - u[n-1, i] + (c * dt / dx)**2 * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
        
        # Apply boundary conditions
        u[n+1, 0] = boundary_conditions[0]
        u[n+1, -1] = boundary_conditions[1]
    
    return x, t, u
