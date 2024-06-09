import numpy as np
import matplotlib.pyplot as plt

def crank_nicolson_vasicek(a, b, sigma, r0, K, T, r_max, M, N):
    """
    Crank-Nicolson method for pricing a European call option on a zero-coupon bond under the Vasicek model.

    Parameters:
    - a: Speed of mean reversion
    - b: Long-term mean level
    - sigma: Volatility
    - r0: Initial short rate
    - K: Strike price of the bond option
    - T: Time to maturity of the option
    - r_max: Maximum short rate considered
    - M: Number of time steps
    - N: Number of short rate steps

    Returns:
    - Option price at r0
    - Grid of option prices
    - Short rates
    """
    dt = T / M
    dr = r_max / N
    r = np.linspace(0, r_max, N+1)
    V = np.zeros((N+1, M+1))

    # Terminal condition at maturity
    bond_prices_at_maturity = np.exp(-r * T)
    V[:, -1] = np.maximum(0, bond_prices_at_maturity - K)

    # Boundary conditions
    V[0, :] = np.maximum(0, np.exp(-0 * np.linspace(0, T, M+1)) - K)  # r = 0
    V[-1, :] = 0  # Assume bond price goes to 0 as short rate becomes very high

    # Coefficients
    alpha = 0.25 * dt * (sigma**2 * np.arange(N+1)**2 / dr**2 - a * (b - r) / dr)
    beta = 1 + dt * (sigma**2 * np.arange(N+1)**2 / dr**2 + r)
    gamma = -0.25 * dt * (sigma**2 * np.arange(N+1)**2 / dr**2 + a * (b - r) / dr)

    # Tridiagonal matrix coefficients
    A = np.zeros((N-1, N-1))
    B = np.zeros((N-1, N-1))

    for i in range(1, N):
        A[i-1, i-1] = 1 + beta[i]
        if i > 1:
            A[i-1, i-2] = -alpha[i]
        if i < N-1:
            A[i-1, i] = -gamma[i]

    for i in range(1, N):
        B[i-1, i-1] = 1 - beta[i]
        if i > 1:
            B[i-1, i-2] = alpha[i]
        if i < N-1:
            B[i-1, i] = gamma[i]

    # Time-stepping
    for j in range(M, 0, -1):
        V_inner = np.linalg.solve(A, B @ V[1:N, j] + np.exp(-r[1:N] * dt))
        V[1:N, j-1] = V_inner

    # Interpolate to find the option price at r0
    option_price = np.interp(r0, r, V[:, 0])

    return option_price, V, r

# Parameters
a = 0.1     # Speed of mean reversion
b = 0.05    # Long-term mean level
sigma = 0.02  # Volatility
r0 = 0.03   # Initial short rate
K = 0.95    # Strike price of the bond option
T = 1.0     # Time to maturity of the option (1 year)
r_max = 0.2 # Maximum short rate considered
M = 1000    # Number of time steps
N = 100     # Number of short rate steps

# Price the European call option on a zero-coupon bond
option_price, V, r = crank_nicolson_vasicek(a, b, sigma, r0, K, T, r_max, M, N)
print(f"European Call Option Price on Zero-Coupon Bond: {option_price:.4f}")

# Plot the option price surface
plt.figure(figsize=(10, 6))
plt.imshow(V, extent=[0, T, 0, r_max], aspect='auto', cmap='viridis', origin='lower')
plt.colorbar(label='Option Price')
plt.xlabel('Time to Maturity')
plt.ylabel('Short Rate')
plt.title('European Call Option Price Surface under Vasicek Model')
plt.show()
