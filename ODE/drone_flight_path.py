import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This is an example of drone flight path with wind speed application to ODE solver. 
"""

def drone_dynamics(t, z):
    x, y = z
    k = 6 / np.sqrt(x**2 + y**2)
    dxdt = -k * x
    dydt = -k * y - 4
    return [dxdt, dydt]

# Initial conditions: Drone starts at (5, 0)
z0 = [5, 0]
t_span = (0, 10)  # Simulation time

sol = solve_ivp(drone_dynamics, t_span, z0, t_eval=np.linspace(t_span[0], t_span[1], 300))

# Plotting the results
plt.figure(figsize=(8, 5))
plt.plot(sol.y[0], sol.y[1], label='Drone Path')
plt.scatter(0, 0, color='red', label='Home')
plt.title('Path of the drone affected by wind')
plt.xlabel('East-West Position (miles)')
plt.ylabel('North-South Position (miles)')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.show()
