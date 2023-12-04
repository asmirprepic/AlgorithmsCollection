import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 50, 50  # Number of spatial steps
dx, dy = 1.0 / (nx - 1), 1.0 / (ny - 1)  # Spatial step sizes

# Function f(x, y)
def f(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)

# Grid
x = np.linspace(0, 1, nx)
y = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x, y)

# Initialize u and set boundary conditions
u = np.zeros((ny, nx))
u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0  # Boundary conditions

# Iterative solver
for _ in range(1000):
    un = u.copy()
    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            u[j, i] = ((un[j, i+1] + un[j, i-1]) * dy**2 +
                       (un[j+1, i] + un[j-1, i]) * dx**2 -
                       f(x[i], y[j]) * dx**2 * dy**2) / (2 * (dx**2 + dy**2))

# Plotting
plt.imshow(u, cmap='viridis', extent=[0, 1, 0, 1])
plt.colorbar()
plt.show()
