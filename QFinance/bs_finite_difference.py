import numpy as np

# Parameters
S_max = 100            # max stock price
K = 50                 # strike price
T = 1.0                # time to maturity
sigma = 0.2            # volatility
r = 0.05               # risk-free rate
M = 50                 # number of price steps
N = 1000               # number of time steps
dt = T/N               # time step size
dS = S_max/M           # price step size

# Create grid
S = np.linspace(0, S_max, M+1)
t = np.linspace(0, T, N+1)
V = np.zeros((M+1, N+1))

# Boundary conditions for European Call Option
V[:, -1] = np.maximum(S - K, 0)  # at maturity
V[-1, :-1] = (S_max - K) * np.exp(-r * t[:-1])  # S = S_max

# Finite Difference Method
for j in range(N-1, -1, -1):
    for i in range(1, M):
        delta = (V[i+1, j+1] - V[i-1, j+1]) / (2*dS)
        gamma = (V[i+1, j+1] - 2*V[i, j+1] + V[i-1, j+1]) / (dS**2)
        theta = -0.5 * sigma**2 * S[i]**2 * gamma - r * S[i] * delta + r * V[i, j+1]
        V[i, j] = V[i, j+1] + dt * theta

# Option price is at the second index (S = 0)
option_price = V[1, 0]
print("Option Price:", option_price)
