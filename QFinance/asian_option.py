
import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100       # Initial stock price
K = 100        # Strike price
T = 1.0        # Time to maturity
r = 0.05       # Risk-free interest rate
sigma = 0.2    # Volatility of the underlying asset
M = 50         # Number of time steps
I = 10000      # Number of simulations

np.random.seed(0)

# Time grid
dt = T / M
t = np.linspace(0, T, M + 1)

# Simulating I/2 paths and their antithetic paths
S = np.zeros((M + 1, I))
S[0] = S0
for i in range(1, M + 1):
    Z = np.random.standard_normal(I // 2)  
    Z = np.concatenate((Z, -Z))  # Antithetic variates
    S[i] = S[i - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

# Calculating the average stock price
S_avg = np.mean(S[1:], axis=0)

# Calculating the payoff for a call option
payoff = np.maximum(S_avg - K, 0)

# Discounting the payoff back to present value
C0 = np.exp(-r * T) * np.mean(payoff)

# Plotting the average price distribution
plt.figure(figsize=(12, 7))
plt.hist(S_avg, bins=50, alpha=0.75, label='Distribution of Average Prices')
plt.axvline(x=np.mean(S_avg), color='r', linestyle='dashed', linewidth=2, label='Mean Average Price')
plt.title('Distribution of Average Prices for Simulated Paths')
plt.xlabel('Average Price')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True)
plt.show()

# Convergence of the option price estimate
convergence = []
samples = np.arange(100, I, 100)
for sample_size in samples:
    estimate = np.exp(-r * T) * np.mean(payoff[:sample_size])
    convergence.append(estimate)

plt.figure(figsize=(12, 7))
plt.plot(samples, convergence, label='Convergence of Option Price Estimate')
plt.axhline(y=C0, color='r', linestyle='dashed', linewidth=2, label=f'Converged Price: ${C0:.2f}')
plt.title('Convergence of Asian Call Option Price Estimate')
plt.xlabel('Number of Simulations')
plt.ylabel('Option Price')
plt.legend()
plt.grid(True)
plt.show()

print(f"The estimated price of the Asian call option is: ${C0:.2f}")
