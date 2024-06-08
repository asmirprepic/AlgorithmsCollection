import numpy as np
import matplotlib.pyplot as plt

def crank_nicolson_american_put(S0, K, T, r, sigma, Smax, M, N):
    """
    Crank-Nicolson finite difference method for pricing an American put option.

    Parameters:
    - S0: Initial stock price
    - K: Strike price
    - T: Time to maturity
    - r: Risk-free interest rate
    - sigma: Volatility
    - Smax: Maximum stock price considered
    - M: Number of time steps
    - N: Number of stock price steps

    Returns:
    - Option price at S0
    - Grid of option prices
    - Stock prices
    """
    dt = T / M
    dS = Smax / N
    S = np.linspace(0, Smax, N+1)
    V = np.zeros((N+1, M+1))

    # Boundary conditions
    V[:, -1] = np.maximum(K - S, 0)
    V[0, :] = K * np.exp(-r * dt * np.arange(M+1))
    V[-1, :] = 0

    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * (np.arange(N+1)**2) - r * np.arange(N+1))
    beta = -dt * 0.5 * (sigma**2 * (np.arange(N+1)**2) + r)
    gamma = 0.25 * dt * (sigma**2 * (np.arange(N+1)**2) + r * np.arange(N+1))

    # Tridiagonal matrix coefficients
    A = np.zeros((N-1, N-1))
    B = np.zeros((N-1, N-1))

    for i in range(1, N):
        A[i-1, i-1] = 1 + beta[i]
        if i > 1:
            A[i-1, i-2] = alpha[i]
        if i < N-1:
            A[i-1, i] = gamma[i]

    for i in range(1, N):
        B[i-1, i-1] = 1 - beta[i]
        if i > 1:
            B[i-1, i-2] = -alpha[i]
        if i < N-1:
            B[i-1, i] = -gamma[i]

    # Time-stepping
    for j in range(M, 0, -1):
        V_inner = np.linalg.solve(A, B @ V[1:N, j])
        V[1:N, j-1] = np.maximum(K - S[1:N], V_inner)

    # Interpolate to find the option price at S0
    option_price = np.interp(S0, S, V[:, 0])

    return option_price, V, S

# Parameters
S0 = 100    # Initial stock price
K = 100     # Strike price
T = 1.0     # Time to maturity (1 year)
r = 0.05    # Risk-free interest rate
sigma = 0.2 # Volatility
Smax = 200  # Maximum stock price considered
M = 1000    # Number of time steps
N = 100     # Number of stock price steps

# Price the American put option
option_price, V, S = crank_nicolson_american_put(S0, K, T, r, sigma, Smax, M, N)

print(f"American Put Option Price: {option_price:.2f}")

# Plot the option price surface
plt.figure(figsize=(10, 6))
plt.imshow(V, extent=[0, T, 0, Smax], aspect='auto', cmap='viridis')
plt.colorbar(label='Option Price')
plt.xlabel('Time to Maturity')
plt.ylabel('Stock Price')
plt.title('American Put Option Price Surface')
plt.show()
