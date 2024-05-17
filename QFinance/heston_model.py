import numpy as np
import matplotlib.pyplot as plt

# Heston model parameters
S0 = 100         # Initial stock price
V0 = 0.04        # Initial variance
mu = 0.05        # Drift rate of the asset price
kappa = 2.0      # Speed of reversion
theta = 0.04     # Long-term mean of the variance
sigma = 0.2      # Volatility of variance
rho = -0.7       # Correlation between W1 and W2
T = 1.0          # Time to maturity
r = 0.03         # Risk-free rate
K = 100          # Strike price
M = 252          # Number of time steps (daily steps for 1 year)
dt = T / M       # Time step size
num_simulations = 10000  # Number of simulations

# Generate correlated random variables
np.random.seed(42)
Z1 = np.random.normal(size=(num_simulations, M))
Z2 = np.random.normal(size=(num_simulations, M))
W1 = Z1
W2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

# Initialize asset price and variance paths
S = np.zeros((num_simulations, M + 1))
V = np.zeros((num_simulations, M + 1))
S[:, 0] = S0
V[:, 0] = V0

# Simulate the Heston model paths
for t in range(1, M + 1):
    V[:, t] = np.maximum(V[:, t-1] + kappa * (theta - V[:, t-1]) * dt + sigma * np.sqrt(V[:, t-1] * dt) * W2[:, t-1], 0)
    S[:, t] = S[:, t-1] * np.exp((mu - 0.5 * V[:, t-1]) * dt + np.sqrt(V[:, t-1] * dt) * W1[:, t-1])

# Calculate the payoff for a European call option
payoff = np.maximum(S[:, -1] - K, 0)
option_price = np.exp(-r * T) * np.mean(payoff)

print(f"The price of the European call option using the Heston model is: ${option_price:.2f}")

# Plot some simulated paths for the asset price and variance
plt.figure(figsize=(14, 6))

# Plot asset price paths
plt.subplot(1, 2, 1)
for i in range(10):  # Plot 10 simulated paths
    plt.plot(S[i], lw=1)
plt.title('Simulated Asset Price Paths')
plt.xlabel('Time Steps')
plt.ylabel('Asset Price')
plt.grid(True)

# Plot variance paths
plt.subplot(1, 2, 2)
for i in range(10):  # Plot 10 simulated paths
    plt.plot(V[i], lw=1)
plt.title('Simulated Variance Paths')
plt.xlabel('Time Steps')
plt.ylabel('Variance')
plt.grid(True)

plt.tight_layout()
plt.show()
