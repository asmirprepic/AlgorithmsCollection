
# %%
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
alpha, beta, delta, gamma = 1.1, 0.4, 0.1, 0.4
sigma_x, sigma_y = 0.1, 0.1  # Intensity of the noise
dt = 0.01  # Time step
T = 50  # Total time
N = int(T / dt)  # Number of time steps
times = np.linspace(0, T, N)

# Initial populations
x = np.zeros(N)
y = np.zeros(N)
x[0], y[0] = 10, 5  # Initial number of prey and predators

# Euler-Maruyama method to solve the SDE
for t in range(N - 1):
    dWx = np.random.normal(0, np.sqrt(dt))
    dWy = np.random.normal(0, np.sqrt(dt))
    
    x[t + 1] = x[t] + (alpha * x[t] - beta * x[t] * y[t]) * dt + sigma_x * x[t] * dWx
    y[t + 1] = y[t] + (delta * x[t] * y[t] - gamma * y[t]) * dt + sigma_y * y[t] * dWy
    
    # Ensure populations stay positive
    x[t + 1] = max(x[t + 1], 0)
    y[t + 1] = max(y[t + 1], 0)

# Plotting
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(times, x, label='Prey')
plt.plot(times, y, label='Predator')
plt.xlabel('Time')
plt.ylabel('Population')
plt.title('Time Series')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x, y, '-r')
plt.xlabel('Prey Population')
plt.ylabel('Predator Population')
plt.title('Phase Plane')
plt.tight_layout()

plt.show()
