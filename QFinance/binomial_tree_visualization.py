import matplotlib.pyplot as plt
import numpy as np 

def visualize_binomial_tree(S0, K, T, r, sigma, N):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    ST = np.zeros((N + 1, N + 1))
    ST[0, 0] = S0
    for i in range(1, N + 1):
        ST[i, 0] = ST[i - 1, 0] * u
        for j in range(1, i + 1):
            ST[i, j] = ST[i - 1, j - 1] * d

    # Initialize option values at maturity
    option = np.zeros_like(ST)
    option[N, :] = np.maximum(K - ST[N, :], 0)

    # Step backwards through the tree
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = np.exp(-r * dt) * (p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1])
            option[i, j] = np.maximum(option[i, j], K - ST[i, j])  # Early exercise condition

    return ST, option
# Parameters
S0 = 100    # Initial stock price
K = 100     # Strike price
T = 1.0     # Time to maturity (1 year)
r = 0.05    # Risk-free interest rate
sigma = 0.2 # Volatility
N = 100     # Number of time steps
# Visualize the binomial tree and option values
ST, option = visualize_binomial_tree(S0, K, T, r, sigma, N)

# Plot the stock price tree
plt.figure(figsize=(12, 6))
for i in range(N + 1):
    plt.plot(np.arange(i + 1), ST[i, :i + 1], 'b-', lw=1)
plt.xlabel('Steps')
plt.ylabel('Stock Price')
plt.title('Binomial Tree for Stock Prices')
plt.grid(True)
plt.show()

# Plot the option value tree
plt.figure(figsize=(12, 6))
for i in range(N + 1):
    plt.plot(np.arange(i + 1), option[i, :i + 1], 'g-', lw=1)
plt.xlabel('Steps')
plt.ylabel('Option Value')
plt.title('Binomial Tree for Option Values')
plt.grid(True)
plt.show()
