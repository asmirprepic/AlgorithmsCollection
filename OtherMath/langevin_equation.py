mport numpy as np
import matplotlib.pyplot as plt

def langevin_equation(steps, dt, gamma, D, x0):
    """
    Simulate the Langevin equation using the Euler-Maruyama method.

    Parameters:
    - steps: Number of time steps to simulate.
    - dt: Time increment per step.
    - gamma: Drag coefficient.
    - D: Diffusion coefficient.
    - x0: Initial position of the particle.

    Returns:
    - A NumPy array of positions over time.
    """
    x = np.zeros(steps)
    x[0] = x0
    for i in range(1, steps):
        dW = np.random.normal(0, np.sqrt(dt))  # Increment of Wiener process
        dx = -gamma * x[i-1] * dt + np.sqrt(2 * D) * dW
        x[i] = x[i-1] + dx
    return x

# Parameters
steps = 10000
dt = 0.01
gamma = 1.0
D = 0.2
x0 = 0.0

# Simulate
x = langevin_equation(steps, dt, gamma, D, x0)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(np.linspace(0, steps*dt, steps), x)
plt.xlabel('Time')
plt.ylabel('Position')
plt.title('Simulation of Particle Diffusion (Langevin Equation)')
plt.show()
