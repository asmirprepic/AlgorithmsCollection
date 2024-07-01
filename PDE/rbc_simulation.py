import numpy as np
import matplotlib.pyplot as plt

# Parameters
Lx, Ly = 1.0, 1.0  # Domain size
Nx, Ny = 100, 100  # Grid points
dx, dy = Lx / (Nx - 1), Ly / (Ny - 1)  # Grid spacing
dt = 0.001  # Time step
t_max = 5.0  # Maximum simulation time

# Physical parameters
Pr = 1.0  # Prandtl number
Ra = 1e6  # Rayleigh number

# Initialize fields
T = np.zeros((Nx, Ny))  # Temperature field
u = np.zeros((Nx, Ny))  # x-velocity field
v = np.zeros((Nx, Ny))  # y-velocity field
psi = np.zeros((Nx, Ny))  # Streamfunction for visualization

# Initial condition: linear temperature gradient
for j in range(Ny):
    T[:, j] = 1.0 - j * dy

# Boundary conditions for temperature
T[:, 0] = 1.0  # Bottom wall
T[:, -1] = 0.0  # Top wall

# Finite difference coefficients
alpha = dt / dx**2
beta = dt / dy**2

def update_velocity(u, v, T):
    un = u.copy()
    vn = v.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            u[i, j] = un[i, j] - dt * (un[i, j] * (un[i, j] - un[i-1, j]) / dx +
                                       vn[i, j] * (un[i, j] - un[i, j-1]) / dy) + \
                      alpha * (un[i+1, j] - 2*un[i, j] + un[i-1, j]) + \
                      beta * (un[i, j+1] - 2*un[i, j] + un[i, j-1]) + \
                      Ra * Pr * dt * T[i, j]

            v[i, j] = vn[i, j] - dt * (un[i, j] * (vn[i, j] - vn[i-1, j]) / dx +
                                       vn[i, j] * (vn[i, j] - vn[i, j-1]) / dy) + \
                      alpha * (vn[i+1, j] - 2*vn[i, j] + vn[i-1, j]) + \
                      beta * (vn[i, j+1] - 2*vn[i, j] + vn[i, j-1])
    return u, v

def update_temperature(T, u, v):
    Tn = T.copy()
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            T[i, j] = Tn[i, j] - dt * (u[i, j] * (Tn[i, j] - Tn[i-1, j]) / dx +
                                       v[i, j] * (Tn[i, j] - Tn[i, j-1]) / dy) + \
                      alpha * (Tn[i+1, j] - 2*Tn[i, j] + Tn[i-1, j]) + \
                      beta * (Tn[i, j+1] - 2*Tn[i, j] + Tn[i, j-1])
    return T

# Main simulation loop
t = 0.0
while t < t_max:
    u, v = update_velocity(u, v, T)
    T = update_temperature(T, u, v)
    t += dt
    if int(t/dt) % 100 == 0:
        print(f"Time: {t:.2f}s")

# Plot the results
X, Y = np.meshgrid(np.linspace(0, Lx, Nx), np.linspace(0, Ly, Ny))
plt.contourf(X, Y, T.T, cmap='inferno')
plt.colorbar(label='Temperature')
plt.quiver(X, Y, u.T, v.T)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Rayleigh-Bénard Convection')
plt.show()
