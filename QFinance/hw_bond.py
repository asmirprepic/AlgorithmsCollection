import numpy as np
import matplotlib.pyplot as plt

# Hull-White model parameters
alpha = 0.15
sigma = 0.01
r0 = 0.03
T = 1.0  # Time to maturity

# Initial discount factors (example values, these should be derived from market data)
P0T = lambda T: np.exp(-0.02 * T)
f0t = lambda t: 0.02  # Instantaneous forward rate (example value)

def hull_white_bond_price(r, t, T, alpha, sigma, P0T, f0t):
    B = (1 - np.exp(-alpha * (T - t))) / alpha
    A = (P0T(T) / P0T(t)) * np.exp(B * f0t(t) - (sigma**2 / (4 * alpha)) * (1 - np.exp(-2 * alpha * (T - t))) * B**2)
    P = A * np.exp(-B * r)
    return P

# Calculate the bond price using the Hull-White model
bond_price = hull_white_bond_price(r0, 0, T, alpha, sigma, P0T, f0t)
print(f"The price of the zero-coupon bond using the Hull-White model's closed-form solution is: ${bond_price:.4f}")

# Plotting the bond price as a function of the initial interest rate
r_values = np.linspace(0, 0.1, 100)
bond_prices = [hull_white_bond_price(r, 0, T, alpha, sigma, P0T, f0t) for r in r_values]

plt.figure(figsize=(10, 6))
plt.plot(r_values, bond_prices)
plt.title('Zero-Coupon Bond Price as a Function of Initial Interest Rate (Hull-White Model)')
plt.xlabel('Initial Interest Rate')
plt.ylabel('Bond Price')
plt.grid(True)
plt.show()
