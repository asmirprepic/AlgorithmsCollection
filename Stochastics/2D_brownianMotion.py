import numpy as np
import matplotlib.pyplot as plt

def brownian_motion_2D(num_steps, delta_t):
    """
    Simulate a 2D Brownian motion.

    Parameters:
    num_steps (int): Number of steps in the simulation.
    delta_t (float): Time interval between steps.

    Returns:
    numpy.array: Arrays of x and y coordinates.
    """
    theta = np.random.uniform(0, 2*np.pi, num_steps)
    step_sizes = np.sqrt(delta_t) * np.random.normal(size=num_steps)
    x = np.cumsum(step_sizes * np.cos(theta))
    y = np.cumsum(step_sizes * np.sin(theta))
    return x, y

def simulate_multiple_paths(num_paths, num_steps, delta_t, specific_t):
    """
    Simulate multiple 2D Brownian motion paths and calculate the expected position at a specific time.

    Parameters:
    num_paths (int): Number of paths to simulate.
    num_steps (int): Number of steps in each simulation.
    delta_t (float): Time interval between steps.
    specific_t (float): Specific time at which to calculate the expected position.

    Returns:
    tuple: Average x and y position at the specific time, and arrays of all paths.
    """
    all_x = []
    all_y = []
    for _ in range(num_paths):
        x, y = brownian_motion_2D(num_steps, delta_t)
        all_x.append(x)
        all_y.append(y)

    # Calculate the expected position at the specific time
    specific_step = int(specific_t / delta_t)
    avg_x = np.mean([path[specific_step] for path in all_x])
    avg_y = np.mean([path[specific_step] for path in all_y])

    return avg_x, avg_y, all_x, all_y

# Simulation parameters
num_paths = 1000     # Number of Brownian paths
num_steps = 1000     # Number of steps in each path
delta_t = 0.01       # Time interval
specific_t = 1       # Time at which to calculate the expected position

# Simulate and calculate the expected position
avg_x, avg_y, all_x, all_y = simulate_multiple_paths(num_paths, num_steps, delta_t, specific_t)

# Plotting
plt.figure(figsize=(10, 10))
for i in range(num_paths):
    plt.plot(all_x[i], all_y[i], '-o', markersize=2, alpha=0.1)
plt.plot(avg_x, avg_y, 'ro', markersize=5)
plt.title(f'2D Brownian Motion Paths with Expected Position at t={specific_t}')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.axis('equal')
plt.show()

print(f"Expected Position at t={specific_t}: (X: {avg_x}, Y: {avg_y})")
