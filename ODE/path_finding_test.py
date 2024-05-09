
import numpy as np
import matplotlib.pyplot as plt

# Constants
k_attr = 0.1  # Attraction constant
k_rep = 500  # Repulsion constant
max_speed = 0.5  # Maximum speed
dt = 0.1  # Time step

# Initial conditions
p = np.array([0.0, 0.0])  # Initial position of the entity
v = np.array([0.0, 0.0])  # Initial velocity of the entity
p_target = np.array([10.0, 10.0])  # Position of the target
p_obstacle = np.array([5.0, 5.0])  # Position of the obstacle

# Function to calculate derivatives
def derivatives(state, p_target, p_obstacle, k_attr, k_rep):
    p, v = state[:2], state[2:]
    force_attr = k_attr * (p_target - p)
    distance_to_obstacle = np.linalg.norm(p - p_obstacle)
    force_rep = k_rep * (p - p_obstacle) / distance_to_obstacle**3
    a = force_attr - force_rep
    return np.concatenate([v, a])

# Runge-Kutta fourth-order method
def rk4(y, h, p_target, p_obstacle, k_attr, k_rep):
    k1 = derivatives(y, p_target, p_obstacle, k_attr, k_rep)
    k2 = derivatives(y + 0.5 * h * k1, p_target, p_obstacle, k_attr, k_rep)
    k3 = derivatives(y + 0.5 * h * k2, p_target, p_obstacle, k_attr, k_rep)
    k4 = derivatives(y + h * k3, p_target, p_obstacle, k_attr, k_rep)
    return y + (h / 6) * (k1 + 2*k2 + 2*k3 + k4)

# Time simulation
time_steps = 300
trajectory = np.zeros((time_steps, 2))  # Store positions for plotting
state = np.concatenate([p, v])

for i in range(time_steps):
    state = rk4(state, dt, p_target, p_obstacle, k_attr, k_rep)
    p, v = state[:2], state[2:]
    
    # Limit the speed to max_speed
    speed = np.linalg.norm(v)
    if speed > max_speed:
        v = (v / speed) * max_speed
    state[2:] = v
    
    trajectory[i] = p

    # Stop if close enough to the target
    if np.linalg.norm(p - p_target) < 0.1:
        print("Target reached at step", i)
        trajectory = trajectory[:i+1]
        break

# Plot the result
plt.plot(trajectory[:, 0], trajectory[:, 1], label='Path')
plt.scatter(*p_target, color='green', label='Target')
plt.scatter(*p_obstacle, color='red', label='Obstacle')
plt.title('Pathfinding with Runge-Kutta Method')
plt.xlabel('X position')
plt.ylabel('Y position')
plt.legend()
plt.grid(True)
plt.show()
