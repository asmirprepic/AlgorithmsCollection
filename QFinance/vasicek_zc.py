import numpy as np
import matplotlib.pyplot as plt

# Vasicek model parameters
alpha = 0.15
theta = 0.05
sigma = 0.01
r0 = 0.03
T = 1.0  # Time to maturity

def vasicek_zero_coupon_bond_price(r, T, alpha, theta, sigma):
    B = (1 - np.exp(-alpha * T)) / alpha
    A = np.exp((theta - (sigma**2) / (2 * alpha**2)) * (B - T) - (sigma**2) / (4 * alpha) * B**2)
    P = A * np.exp(-B * r)
    return P

# Calculate the bond price using the closed-form solution
bond_price = vasicek_zero_coupon_bond_price(r0, T, alpha, theta, sigma)
print(f"The price of the zero-coupon bond using the Vasicek model's closed-form solution is: ${bond_price:.4f}")

# Plotting the bond price as a function of the initial interest rate
r_values = np.linspace(0, 0.1, 100)
bond_prices = [vasicek_zero_coupon_bond_price(r, T, alpha, theta, sigma) for r in r_values]

plt.figure(figsize=(10, 6))
plt.plot(r_values, bond_prices)
plt.title('Zero-Coupon Bond Price as a Function of Initial Interest Rate')
plt.xlabel('Initial Interest Rate')
plt.ylabel('Bond Price')
plt.grid(True)
plt.show()
