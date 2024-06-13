
import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10.0          # Total time
dt = 0.01         # Time step size
N = int(T / dt)   # Number of time steps
gamma = 0.5       # Damping coefficient
omega = 1.0       # Natural frequency
sigma = 0.1       # Noise intensity

# Initialize arrays
t = np.linspace(0, T, N)
X = np.zeros(N)
Y = np.zeros(N)
X[0] = 1.0        # Initial position
Y[0] = 0.0        # Initial velocity

# Euler-Maruyama method for SDE
for i in range(1, N):
    dW = np.sqrt(dt) * np.random.normal()
    X[i] = X[i-1] + Y[i-1] * dt
    Y[i] = Y[i-1] - gamma * Y[i-1] * dt - omega**2 * X[i-1] * dt + sigma * dW

# Plot the results
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(t, X, label='Position (X)')
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Damped Harmonic Oscillator - Position')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(t, Y, label='Velocity (Y)', color='orange')
plt.xlabel('Time')
plt.ylabel('Velocity')
plt.title('Damped Harmonic Oscillator - Velocity')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
