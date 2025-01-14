"""
1 dimensional wave equation
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
c = 1  # Wave speed
nx = 1000  # Number of spatial steps
dx = 2 * np.pi / nx  # Spatial step size
dt = 0.01  # Time step size
nt = 1000  # Number of time steps

# Initial condition
x = np.linspace(0, 2 * np.pi, nx)
u = np.sin(x)
u_prev = u.copy()
u_next = np.zeros(nx)

# Time-stepping loop
for n in range(nt):
    for i in range(1, nx - 1):
        u_next[i] = 2 * u[i] - u_prev[i] + c**2 * dt**2 / dx**2 * (u[i+1] - 2 * u[i] + u[i-1])
    u_prev, u = u, u_next

# Plotting
plt.plot(x, u)
plt.show()
