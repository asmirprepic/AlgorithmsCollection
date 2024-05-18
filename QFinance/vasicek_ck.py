import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded

# Vasicek model parameters
alpha = 0.15
theta = 0.05
sigma = 0.01
r_max = 0.15
r_min = 0.0
T = 1.0   # Time to maturity
r0 = 0.03  # Initial short rate
N = 100    # Number of rate steps
M = 100    # Number of time steps
dt = T / M
dr = (r_max - r_min) / N

# Grid setup
r = np.linspace(r_min, r_max, N+1)
t = np.linspace(0, T, M+1)

# Initial and boundary conditions
P = np.zeros((N+1, M+1))
P[:, -1] = 1  # Zero-coupon bond pays 1 at maturity

# Coefficients for the Crank-Nicolson scheme
a = 0.5 * dt * (sigma**2 / dr**2 - alpha * (theta - r) / dr)
b = 1 + dt * (sigma**2 / dr**2 + r)
c = 0.5 * dt * (sigma**2 / dr**2 + alpha * (theta - r) / dr)
d = 1 - dt * (sigma**2 / dr**2 + r)

# Tridiagonal matrix setup for the implicit part
A = np.zeros((3, N-1))
A[0, 1:] = -a[2:N]
A[1, :] = b[1:N]
A[2, :-1] = -c[1:N-1]

# Time stepping
for j in range(M-1, -1, -1):
    # Explicit part
    P[1:N, j] = d[1:N] * P[1:N, j+1] + a[1:N] * P[0:N-1, j+1] + c[1:N] * P[2:N+1, j+1]
    # Solve tridiagonal system
    P[1:N, j] = solve_banded((1, 1), A, P[1:N, j])

# Interpolation to get the bond price at the initial rate
P0 = np.interp(r0, r, P[:, 0])
print(f"The price of the zero-coupon bond using the Vasicek model and Crank-Nicolson method is: {P0:.4f}")

# Plotting the bond price surface
R, T = np.meshgrid(r, t)
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(R, T, P.T, cmap='viridis')
ax.set_title('Zero-Coupon Bond Price Surface')
ax.set_xlabel('Short Rate')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Bond Price')
plt.show()
