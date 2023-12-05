import numpy as np

def finite_difference_option_price(S, X, r, T, sigma, N, M, option_type):
    """
    Finite difference method for European option pricing.
    
    Parameters:
    S: Current price of the underlying asset
    X: Exercise price of the option
    r: Risk-free interest rate
    T: Time to expiration of the option (in years)
    sigma: Volatility of the underlying asset's price
    N: Number of time steps
    M: Number of asset price steps
    option_type: "call" or "put"
    """
    dt = T / N
    dS = S * 2 / M
    grid = np.zeros((M+1, N+1))

    # Set up the end condition
    ST = np.linspace(0, S * 2, M+1)
    if option_type == "call":
        grid[:, -1] = np.maximum(ST - X, 0)
    elif option_type == "put":
        grid[:, -1] = np.maximum(X - ST, 0)

    # Set up the coefficients
    a = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 - r * np.arange(M+1))
    b = 1 - dt * (sigma**2 * np.arange(M+1)**2 + r)
    c = 0.5 * dt * (sigma**2 * np.arange(M+1)**2 + r * np.arange(M+1))

    # Iteration over time
    for j in range(N-1, -1, -1):
        for i in range(1, M):
            grid[i, j] = a[i] * grid[i-1, j+1] + b[i] * grid[i, j+1] + c[i] * grid[i+1, j+1]

    # Find the closest grid point to S
    i = np.searchsorted(ST, S) - 1
    # Interpolate to find the option value at S
    option_value = grid[i, 0] + (S - ST[i]) * (grid[i+1, 0] - grid[i, 0]) / (dS)
    return option_value

# Example usage
S = 50  # Current stock price
X = 50  # Strike price
r = 0.05  # Annual risk-free rate
T = 1  # Time to maturity in years
sigma = 0.2  # Volatility
N = 1000  # Number of time steps
M = 100  # Number of price steps
option_type = "call"  # Option type

option_price = finite_difference_option_price(S, X, r, T, sigma, N, M, option_type)
print("Option Price:", option_price)
