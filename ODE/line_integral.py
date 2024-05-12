
# Vector field F(x, y) = (y, x)
def vector_field(x, y):
    return y, x

# Grid of x, y points
x = np.linspace(-1.5, 1.5, 20)
y = np.linspace(-1.5, 1.5, 20)
X, Y = np.meshgrid(x, y)

U, V = vector_field(X, Y)

# Circle parameterization for the path
t = np.linspace(0, 2 * np.pi, 100)
x_path = np.cos(t)
y_path = np.sin(t)

# Plotting
plt.figure(figsize=(8, 8))
plt.quiver(X, Y, U, V, color='r', scale=10)
plt.plot(x_path, y_path, 'b-', linewidth=2, label='Path (Circle)')
plt.xlim(-1.5, 1.5)
plt.ylim(-1.5, 1.5)
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Vector Field and Integration Path')
plt.legend()
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

def F(r):
    x, y = r
    return np.array([y, x])  # Vector field F(x, y) = (y, x)

def r(t):
    return np.array([np.cos(t), np.sin(t)])  # Parameterization of the circle

def dr_dt(t):
    return np.array([-np.sin(t), np.cos(t)])  # Derivative of r(t)

def integrand(t):
    return np.dot(F(r(t)), dr_dt(t))

# Generate values of t from 0 to 2π
t_values = np.linspace(0, 2 * np.pi, 1000)
integrand_values = [integrand(t) for t in t_values]

# Plotting the integrand
plt.figure(figsize=(10, 5))
plt.plot(t_values, integrand_values, label=r'$\mathbf{F} \cdot d\mathbf{r}$')
plt.title('Integrand $\mathbf{F} \cdot d\mathbf{r}$ Along the Path')
plt.xlabel('$t$ (radians)')
plt.ylabel('Integrand Value')
plt.grid(True)
plt.legend()
plt.show()
