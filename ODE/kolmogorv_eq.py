import numpy as np
import matplotlib.pyplot as plt

def solve_forward_kolmogorov(drift, diffusion, x_min, x_max, dx, t_min, t_max, dt, initial_condition):
    """
    Solve the forward Kolmogorov (Fokker-Planck) equation using finite difference method.
    
    Parameters:
    - drift: function, Drift coefficient A(x)
    - diffusion: function, Diffusion coefficient B(x)
    - x_min, x_max: float, Spatial domain boundaries
    - dx: float, Spatial step size
    - t_min, t_max: float, Temporal domain boundaries
    - dt: float, Time step size
    - initial_condition: array-like, Initial probability distribution p(x, 0)
    
    Returns:
    - x: numpy array, Discretized spatial domain
    - t: numpy array, Discretized time domain
    - p_result: numpy array, Probability distribution over space and time
    """
    # Spatial domain
    x = np.arange(x_min, x_max, dx)
    nx = len(x)

    # Time domain
    t = np.arange(t_min, t_max, dt)
    nt = len(t)

    # Initial condition for p(x, 0)
    p = np.array(initial_condition)
    p /= np.sum(p * dx)  # Normalize

    # Prepare to store results
    p_result = np.zeros((nt, nx))
    p_result[0, :] = p

    # Finite difference method for the forward Kolmogorov equation
    for n in range(1, nt):
        p_new = np.zeros_like(p)
        for i in range(1, nx - 1):
            dA_dx = (drift(x[i+1]) * p[i+1] - drift(x[i-1]) * p[i-1]) / (2 * dx)
            dB_dxx = (diffusion(x[i+1]) * p[i+1] - 2 * diffusion(x[i]) * p[i] + diffusion(x[i-1]) * p[i-1]) / dx**2
            p_new[i] = p[i] + dt * (-dA_dx + dB_dxx)
        
        # Apply boundary conditions (assuming zero-flux)
        p_new[0] = p_new[1]  # Reflective at left boundary
        p_new[-1] = p_new[-2]  # Reflective at right boundary

        # Update for next time step
        p = p_new
        p_result[n, :] = p

    return x, t, p_result

# Define drift and diffusion functions
def drift_function(x):
    theta = 1.0  # Rate of mean reversion
    mu = 0.0     # Long-term mean
    return theta * (mu - x)

def diffusion_function(x):
    sigma = 0.3  # Volatility
    return (sigma ** 2) / 2

# Parameters
x_min, x_max = -2, 2
dx = 0.1
t_min, t_max = 0, 2
dt = 0.01

# Initial condition for the probability distribution
initial_condition = np.exp(-np.linspace(x_min, x_max, int((x_max-x_min)/dx))**2 / 0.1)

# Solve the forward Kolmogorov equation
x, t, p_result = solve_forward_kolmogorov(drift_function, diffusion_function, x_min, x_max, dx, t_min, t_max, dt, initial_condition)

# Plot the solution over time
plt.figure(figsize=(10, 6))
for i in range(0, len(t), int(len(t) / 5)):
    plt.plot(x, p_result[i, :], label=f't={t[i]:.2f}')
plt.xlabel('State x')
plt.ylabel('Probability Density p(x, t)')
plt.title('Forward Kolmogorov Equation (Fokker-Planck)')
plt.legend()
plt.grid(True)
plt.show()
