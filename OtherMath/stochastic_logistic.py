import numpy as np
import matplotlib.pyplot as plt

def simulate_stochastic_logistic(alpha, K, S0, T, dt, sigma):
    """
    Simulate stochastic logistic growth using Euler-Maruyama method.

    Parameters:
    - alpha: Growth rate
    - K: Carrying capacity
    - S0: Initial sentiment
    - T: Total time (e.g., days)
    - dt: Time step size
    - sigma: Noise strength

    Returns:
    - t: Time points
    - S: Sentiment levels
    """
    n_steps = int(T / dt)
    t = np.linspace(0, T, n_steps)
    S = np.zeros(n_steps)
    S[0] = S0

    for i in range(1, n_steps):
        dW = np.random.normal(0, np.sqrt(dt))  
        drift = alpha * S[i-1] * (1 - S[i-1] / K) * dt
        diffusion = sigma * S[i-1] * dW
        S[i] = S[i-1] + drift + diffusion

        
        S[i] = max(0, S[i])

    return t, S

# Parameters
alpha = 0.5  # Growth rate
K = 100      # Carrying capacity
S0 = 10      # Initial sentiment
T = 20       # Time horizon (days)
dt = 0.01    # Time step size
sigma = 0.2  # Noise strength


t, S = simulate_stochastic_logistic(alpha, K, S0, T, dt, sigma)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(t, S, label="Stochastic Sentiment")
plt.axhline(K, color="red", linestyle="--", label="Carrying Capacity (K)")
plt.xlabel("Time")
plt.ylabel("Sentiment")
plt.title("Stochastic Logistic Growth")
plt.legend()
plt.show()
