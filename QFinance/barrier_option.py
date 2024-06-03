import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100           # Initial stock price
K = 100            # Strike price
B = 120            # Barrier level
T = 1.0            # Time to maturity (1 year)
r = 0.05           # Risk-free interest rate
sigma = 0.2        # Volatility
N = 10000          # Number of simulations
M = 1000           # Number of time steps
dt = T / M         # Time step size

# Initialize arrays
payoff = np.zeros(N)

# Monte Carlo simulation for barrier option
for i in range(N):
    S = S0
    hit_barrier = False
    for t in range(M):
        dW = np.sqrt(dt) * np.random.normal()
        S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * dW)
        if S >= B:
            hit_barrier = True
            break
    if hit_barrier:
        payoff[i] = np.exp(-r * T) * max(S - K, 0)

# Calculate the price of the barrier option
option_price = np.mean(payoff)

print(f"Barrier Option Price: {option_price:.2f}")
