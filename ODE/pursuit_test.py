# %%

import numpy as np
import matplotlib.pyplot as plt

def update_target_position_circular(t, radius, omega, x_center=0, y_center=0):
    """
    Update the target's position to move in a circular path.
    """
    x_t = x_center + radius * np.cos(omega * t)
    y_t = y_center + radius * np.sin(omega * t)
    return x_t, y_t

def update_pursuer_position(x_p, y_p, x_t, y_t, v_p, dt):
    """
    Update the pursuer's position.
    """
    dist = np.sqrt((x_t - x_p)**2 + (y_t - y_p)**2)
    x_p_new = x_p + v_p * (x_t - x_p) / dist * dt
    y_p_new = y_p + v_p * (y_t - y_p) / dist * dt
    return x_p_new, y_p_new

def solve_pursuit(v_p, radius, omega, dt, t_max, x_p0, y_p0, catch_radius=0.1):
    """
    Solve the pursuit problem using Euler's method.
    """
    t = np.arange(0, t_max, dt)
    
    # Arrays to store positions
    x_t = np.zeros_like(t)
    y_t = np.zeros_like(t)
    x_p = np.zeros_like(t)
    y_p = np.zeros_like(t)
    
    # Initial positions
    x_p[0], y_p[0] = x_p0, y_p0
    
    # Euler method to solve the differential equations
    for i in range(len(t)):
        x_t[i], y_t[i] = update_target_position_circular(t[i], radius, omega)
        if i > 0:
            x_p[i], y_p[i] = update_pursuer_position(x_p[i-1], y_p[i-1], x_t[i-1], y_t[i-1], v_p, dt)
        # Check if pursuer catches the target
        if np.sqrt((x_t[i] - x_p[i])**2 + (y_t[i] - y_p[i])**2) < catch_radius:
            x_t = x_t[:i+1]
            y_t = y_t[:i+1]
            x_p = x_p[:i+1]
            y_p = y_p[:i+1]
            t = t[:i+1]
            print(f"Target caught at t = {t[-1]:.2f} seconds")
            break
    
    return t, x_t, y_t, x_p, y_p

def plot_trajectories(x_t, y_t, x_p, y_p):
    """
    Plot the trajectories of the target and the pursuer.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x_t, y_t, label='Target', color='blue')
    plt.plot(x_p, y_p, label='Pursuer', color='red')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Pursuit Curve with Circular Target Movement')
    plt.legend()
    plt.grid(True)
    plt.show()

# Parameters
v_p = 1.2  # Speed of the pursuer
radius = 5.0  # Radius of the circular path
omega = 1.0  # Angular speed of the target
dt = 0.01  # Time step
t_max = 20.0  # Maximum time

# Initial position of the pursuer
x_p0, y_p0 = -10.0, -10.0

# Solve the pursuit problem
t, x_t, y_t, x_p, y_p = solve_pursuit(v_p, radius, omega, dt, t_max, x_p0, y_p0)

# Plot the results
plot_trajectories(x_t, y_t, x_p, y_p)
