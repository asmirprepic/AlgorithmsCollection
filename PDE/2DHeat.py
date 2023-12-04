"""
Solveing 2D Heat equation
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = 0.01  # Thermal diffusivity
nx, ny = 50, 50  # Number of spatial steps
dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # Spatial step sizes
dt = 0.0001  # Time step size
nt = 100  # Number of time steps

# Initial condition
u = np.zeros((ny, nx))
u[int(0.5 / dy):int(0.6 / dy), int(0.5 / dx):int(0.6 / dx)] = 100  # Initial heat source

# Time-stepping loop
for n in range(nt):
    un = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[j, i] = (un[j, i] + alpha * dt / dx**2 * (un[j, i+1] - 2 * un[j, i] + un[j, i-1]) +
                                  alpha * dt / dy**2 * (un[j+1, i] - 2 * un[j, i] + un[j-1, i]))

# Plotting
plt.imshow(u, cmap='hot')
plt.colorbar()
plt.show()
