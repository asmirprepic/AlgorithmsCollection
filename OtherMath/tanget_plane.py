# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative
from mpl_toolkits.mplot3d import Axes3D

def g(x, y):
    return x**3 - x**2 * y + y**2 - 2*x + 3*y - 2

def partial_derivative(func, var=0, point=[]):
    args = point[:]
    def wraps(x):
        args[var] = x
        return func(*args)
    return derivative(wraps, point[var], dx = 1e-6)

# Point of interest
x0, y0 = -1, 3
z0 = g(x0, y0)

# Numerically estimate partial derivatives
gx = partial_derivative(g, 0, [x0, y0])
gy = partial_derivative(g, 1, [x0, y0])

# Equation of the tangent plane
# z = z0 + gx * (x - x0) + gy * (y - y0)
def tangent_plane(x, y):
    return z0 + gx * (x - x0) + gy * (y - y0)

# Plotting
x = np.linspace(-3, 1, 400)
y = np.linspace(1, 5, 400)
x, y = np.meshgrid(x, y)
z = g(x, y)

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Surface plot
ax.plot_surface(x, y, z, alpha=0.5, rstride=10, cstride=10, color='b', edgecolors='k', label='Surface')

# Tangent plane plot
z_tangent = tangent_plane(x, y)
ax.plot_surface(x, y, z_tangent, alpha=0.5, rstride=10, cstride=10, color='r', edgecolors='k', label='Tangent Plane')

# Customize the plot
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Surface and Tangent Plane at (-1, 3, 14)')

plt.show()



