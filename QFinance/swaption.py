import numpy as np
import matplotlib.pyplot as plt

# Hull-White model parameters
r0 = 0.03       # initial short rate
alpha = 0.1     # mean reversion speed
sigma = 0.01    # volatility
T = 5.0         # maturity of the swaption (in years)
S = 10.0        # swap maturity (in years)
K = 0.03        # swaption strike (fixed rate)
M = 500         # number of time steps
dt = T / M      # time step
N = 10000       # number of simulations

np.random.seed(0)

# Time grid
t = np.linspace(0, T, M+1)

# Simulate the short rate paths
r = np.zeros((M+1, N))
r[0] = r0
for i in range(1, M+1):
    Z = np.random.normal(0, 1, N)
    r[i] = r[i-1] + alpha * (theta(i*dt) - r[i-1]) * dt + sigma * np.sqrt(dt) * Z

# Calculate the bond prices
P = np.exp(-np.cumsum(r * dt, axis=0))

# Calculate the swaption payoff
payoff = np.maximum(0, P[-1] * (np.mean(P[int(T/dt):int((T+S)/dt)], axis=0) - K))

# Discount the payoff back to the present value
swaption_price = np.exp(-r0 * T) * np.mean(payoff)

print(f"The price of the European payer swaption is: ${swaption_price:.2f}")

# Visualization of short rate paths
plt.figure(figsize=(12, 6))
plt.plot(t, r[:, :10])
plt.title('Simulated Short Rate Paths under Hull-White Model')
plt.xlabel('Time (Years)')
plt.ylabel('Short Rate')
plt.grid(True)
plt.show()

# Visualization of bond prices
plt.figure(figsize=(12, 6))
plt.plot(t, P[:, :10])
plt.title('Simulated Bond Prices under Hull-White Model')
plt.xlabel('Time (Years)')
plt.ylabel('Bond Price')
plt.grid(True)
plt.show()
