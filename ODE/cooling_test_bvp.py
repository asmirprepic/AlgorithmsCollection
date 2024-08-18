import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0# Length of the fin
T_0 = 100.0# Temperature at the base of the fin
T_inf = 25.0# Ambient temperature
alpha = 1.0# Alpha parameter (hP/kA)
N = 100# Number of discrete points# Discretization
dx = L / (N-1)
x = np.linspace(0, L, N)

# Matrix for the finite difference method
A = np.zeros((N, N))
b = np.zeros(N)

# Interior pointsfor i inrange(1, N-1):
    A[i, i-1] = 1
    A[i, i] = -2 - dx**2 * alpha
    A[i, i+1] = 1
    b[i] = -dx**2 * alpha * T_inf

# Boundary conditions
A[0, 0] = 1# T(0) = T_0
b[0] = T_0

A[-1, -2] = -1# dT/dx(L) = 0
A[-1, -1] = 1
b[-1] = 0# Solve the linear system
T = np.linalg.solve(A, b)

# Plotting the results
plt.plot(x, T, label='Temperature distribution $T(x)$')
plt.xlabel('Position along the fin $x$')
plt.ylabel('Temperature $T(x)$')
plt.title('Temperature Distribution Along the Fin (Finite Difference)')
plt.legend()
plt.grid(True)
plt.show()
