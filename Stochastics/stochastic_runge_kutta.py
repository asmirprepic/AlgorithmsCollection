import numpy as np
import matplotlib.pyplot as plt

# Model parameters
mu = 0.1  # Drift coefficient
sigma = 0.2  # Diffusion coefficient
X0 = 1  # Initial condition

# Time parameters
T = 1.0  # Total time
N = 1000  # Number of time steps
dt = T/N  # Time step size
t = np.linspace(0, T, N+1)  # Time grid

# Initialize the process
X = np.zeros(N+1)
X[0] = X0

# Generate Wiener process increments
dW = np.sqrt(dt) * np.random.randn(N)

# Stochastic Runge-Kutta (Heun's method)
for n in range(N):
    K1 = mu * X[n] * dt + sigma * X[n] * dW[n]
    K2 = mu * (X[n] + K1) * dt + sigma * (X[n] + K1) * dW[n]
    X[n+1] = X[n] + 0.5 * (K1 + K2)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t, X)
plt.title('Solution of the SDE by Stochastic Runge-Kutta Method')
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.grid(True)
plt.show()
