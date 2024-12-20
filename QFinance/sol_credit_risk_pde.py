import numpy as np
import matplotlib.pyplot as plt

def solve_credit_risk_pde(x_max, t_max, dx, dt, sigma, r, B):
    """
    Solve the credit risk PDE for default probabilities using finite differences.
    
    Parameters:
    - x_max: Maximum asset value
    - t_max: Time to maturity
    - dx: Spatial step size
    - dt: Time step size
    - sigma: Volatility of assets
    - r: Risk-free rate
    - B: Default threshold
    
    Returns:
    - x: Asset value grid
    - t: Time grid
    - u: Default probability matrix
    """
    x = np.arange(0, x_max + dx, dx)
    t = np.arange(0, t_max + dt, dt)
    nx = len(x)
    nt = len(t)
    
    u = np.zeros((nt, nx))
    
    # Initial condition
    u[0, :] = (x < B).astype(float)
    
    # Boundary conditions
    u[:, 0] = 1  # Default probability = 1 when x = 0
    u[:, -1] = 0  # Default probability = 0 when x is very large

    # Finite difference method
    for n in range(0, nt - 1):
        for i in range(1, nx - 1):
            dx2 = dx**2
            d2u_dx2 = (u[n, i+1] - 2*u[n, i] + u[n, i-1]) / dx2
            du_dx = (u[n, i+1] - u[n, i-1]) / (2 * dx)
            
            u[n+1, i] = u[n, i] + dt * (
                0.5 * sigma**2 * x[i]**2 * d2u_dx2 + r * x[i] * du_dx - r * u[n, i]
            )

    return x, t, u

# Parameters
x_max = 200      # Maximum asset value
t_max = 1.0      # Time to maturity (years)
dx = 1.0         # Spatial step size
dt = 0.01        # Time step size
sigma = 0.3      # Volatility
r = 0.05         # Risk-free rate
B = 50           # Default threshold

# Solve PDE
x, t, u = solve_credit_risk_pde(x_max, t_max, dx, dt, sigma, r, B)

# Visualize Default Probability
plt.figure(figsize=(10, 6))
for i in range(0, len(t), len(t)//10):
    plt.plot(x, u[i, :], label=f"t = {t[i]:.2f}")
plt.axvline(B, color="red", linestyle="--", label="Default Threshold")
plt.xlabel("Asset Value")
plt.ylabel("Default Probability")
plt.title("Default Probability as a Function of Asset Value and Time")
plt.legend()
plt.show()
