import numpy as np
import matplotlib.pyplot as plt

def simulate_jump_diffusion(S0, mu, sigma, lambda_, J_mean, J_std, T, dt):
    """
    Simulates a jump-diffusion process for catastrophe bond pricing.

    Parameters:
    - S0: Initial bond price
    - mu: Drift (expected return)
    - sigma: Volatility of the bond price
    - lambda_: Expected number of jump events per year
    - J_mean: Mean of jump size
    - J_std: Standard deviation of jump size
    - T: Time period (years)
    - dt: Time step

    Returns:
    - t: Time grid
    - S: Simulated bond price path
    """
    np.random.seed(42)
    N = int(T / dt)
    t = np.linspace(0, T, N)
    S = np.zeros(N)
    S[0] = S0
    
    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion
        J = np.random.normal(J_mean, J_std) if np.random.rand() < lambda_ * dt else 0  # Poisson jump
        
        S[i] = S[i-1] + mu * S[i-1] * dt + sigma * S[i-1] * dW + J

    return t, S

# Parameters
S0 = 100
mu = 0.02
sigma = 0.1
lambda_ = 0.3
J_mean = -20
J_std = 5
T = 10
dt = 0.01

t, S = simulate_jump_diffusion(S0, mu, sigma, lambda_, J_mean, J_std, T, dt)

# Plot the price evolution
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Catastrophe Bond Price")
plt.xlabel("Time (Years)")
plt.ylabel("Bond Price")
plt.title("Catastrophe Bond Price Evolution with Jump-Diffusion Model")
plt.legend()
plt.show()
