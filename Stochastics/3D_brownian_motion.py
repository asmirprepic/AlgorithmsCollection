import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def brownian_motion_3D(num_steps, delta_t):
    """
    Simulate a 3D Brownian motion.

    Parameters:
    num_steps (int): Number of steps in the simulation.
    delta_t (float): Time interval between steps.

    Returns:
    numpy.array: Arrays of x, y, and z coordinates.
    """
    theta = np.random.uniform(0, np.pi, num_steps)
    phi = np.random.uniform(0, 2*np.pi, num_steps)
    step_sizes = np.sqrt(delta_t) * np.random.normal(size=num_steps)
    x = np.cumsum(step_sizes * np.sin(theta) * np.cos(phi))
    y = np.cumsum(step_sizes * np.sin(theta) * np.sin(phi))
    z = np.cumsum(step_sizes * np.cos(theta))
    return x, y, z

def simulate_multiple_paths(num_paths, num_steps, delta_t):
    """
    Simulate multiple 3D Brownian motion paths and collect final positions.

    Parameters:
    num_paths (int): Number of paths to simulate.
    num_steps (int): Number of steps in each simulation.
    delta_t (float): Time interval between steps.

    Returns:
    numpy.array: Final positions of each path.
    """
    final_positions = np.zeros((num_paths, 3))
    for i in range(num_paths):
        x, y, z = brownian_motion_3D(num_steps, delta_t)
        final_positions[i] = x[-1], y[-1], z[-1]
    return final_positions

# Simulation parameters
num_paths = 2    # Number of Brownian paths
num_steps = 1000     # Number of steps in each path
delta_t = 0.01       # Time interval

# Simulate multiple paths and collect final positions
final_positions = simulate_multiple_paths(num_paths, num_steps, delta_t)

# Plotting the paths
fig = plt.figure(figsize=(10, 5))
ax1 = fig.add_subplot(121, projection='3d')

for _ in range(num_paths):
    x, y, z = brownian_motion_3D(num_steps, delta_t)
    ax1.plot(x, y, z, alpha=0.6)

ax1.set_title('3D Brownian Motion Paths')
ax1.set_xlabel('X Position')
ax1.set_ylabel('Y Position')
ax1.set_zlabel('Z Position')
