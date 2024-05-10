import numpy as np
import matplotlib.pyplot as plt

def solve_bvp(L, n, k, f):
    # L: Length of the beam
    # n: Number of intervals (n+1 points)
    # k: Stiffness constant
    # f: External force function
    
    # Setup
    h = L / n  # Step size
    x = np.linspace(0, L, n+1)  # Spatial points
    A = np.zeros((n-1, n-1))  # Coefficient matrix for interior points
    b = -h**2 * f(x[1:n])  # Right-hand side, force term

    # Build the coefficient matrix
    for i in range(n-1):
        A[i, i] = -2 - h**2 * k
        if i > 0:
            A[i, i-1] = 1
        if i < n-2:
            A[i, i+1] = 1
    
    # Solve the system of equations
    y_interior = np.linalg.solve(A, b)
    y = np.zeros(n+1)
    y[1:n] = y_interior
    
    return x, y

# Parameters
L = 10  # Length of the beam
n = 100  # Number of intervals
k = 2  # Initial guess for stiffness
f = lambda x: 1 * np.ones_like(x)  # Constant force

# Solve the BVP
x, y = solve_bvp(L, n, k, f)

# Plotting the solution
plt.plot(x, y, label='Deflection')
plt.title('Deflection of a Beam under Uniform Load')
plt.xlabel('Position along the beam')
plt.ylabel('Deflection')
plt.grid(True)
plt.legend()
plt.show()
