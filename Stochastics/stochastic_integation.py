# %%

import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 1.0  # End time
N = 1000  # Number of steps
dt = T/N  # Time step
M = 5  # Number of paths to simulate

# Time array
t = np.linspace(0, T, N+1)

# Simulate Brownian motion paths
W = np.zeros((M, N+1))
for i in range(1, N+1):
    W[:, i] = W[:, i-1] + np.sqrt(dt) * np.random.randn(M)

# Perform the stochastic integration
integral = np.zeros((M, N+1))
for i in range(1, N+1):
    integral[:, i] = integral[:, i-1] + W[:, i-1] * (W[:, i] - W[:, i-1])

# Plotting
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# Plot Brownian motion paths
for i in range(M):
    axs[0].plot(t, W[i], label=f'Path {i+1}')
axs[0].set_title('Brownian Motion Paths')
axs[0].set_xlabel('Time')
axs[0].set_ylabel('W_t')
axs[0].legend()

# Plot the integral values
for i in range(M):
    axs[1].plot(t, integral[i], label=f'Integral Path {i+1}')
axs[1].set_title('Stochastic Integral Values')
axs[1].set_xlabel('Time')
axs[1].set_ylabel('Integral Value')
axs[1].legend()

plt.tight_layout()
plt.show()
