import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

"""
Illustrates the finite difference method to solve y''(x)+9y(x) = sin(x)
with boundary value conditions y(0),y(2) = 1
"""

# Define parameters
N = 50  # Number of intervals (try changing this for more accuracy)
x = np.linspace(0, 2, N+1)
h = 2/N

# Construct the matrix A
A = np.zeros((N-1, N-1))
np.fill_diagonal(A, 2 + 9 * h**2)
np.fill_diagonal(A[1:], -1)
np.fill_diagonal(A[:, 1:], -1)

# Construct the vector b
b = h**2 * np.sin(x[1:N])

# Apply the boundary conditions
b[0] -= 0  # y(0) = 0
b[-1] -= 1  # y(2) = 1

# Solve the linear system Ay = b
y_inner = linalg.solve(A, b)

# Add the boundary values to the solution
y = np.hstack([0, y_inner, 1])

# Plot the solution
plt.plot(x, y)
plt.title("Solution to the BVP")
plt.xlabel('x')
plt.ylabel('y(x)')
plt.grid(True)
plt.show()
