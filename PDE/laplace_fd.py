import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, kron, identity, linalg as splinalg

def solve_laplace_fd(Nx, Ny, boundary_func):
    """
    Solve Laplace's equation on a unit square using finite differences.

    Args:
        Nx (int): Number of interior grid points in x-direction
        Ny (int): Number of interior grid points in y-direction
        boundary_func (function): Function g(x, y) defining boundary values

    Returns:
        X, Y, U: Meshgrid arrays and 2D solution array U
    """
    # Step size
    hx = 1.0 / (Nx + 1)
    hy = 1.0 / (Ny + 1)

    # Interior points only
    x = np.linspace(hx, 1 - hx, Nx)
    y = np.linspace(hy, 1 - hy, Ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # 2D Laplacian operator with Dirichlet BCs using Kronecker product
    Ix = identity(Nx)
    Iy = identity(Ny)
    Tx = diags([1, -2, 1], [-1, 0, 1], shape=(Nx, Nx)) / hx**2
    Ty = diags([1, -2, 1], [-1, 0, 1], shape=(Ny, Ny)) / hy**2
    A = kron(Iy, Tx) + kron(Ty, Ix)

    # Right-hand side (zero for Laplace), adjusted for boundary conditions
    b = np.zeros(Nx * Ny)

    # Adjust b for boundary conditions
    for i in range(Nx):
        for j in range(Ny):
            xi, yj = x[i], y[j]
            idx = i * Ny + j

            if i == 0:
                b[idx] -= boundary_func(0, yj) / hx**2
            if i == Nx - 1:
                b[idx] -= boundary_func(1, yj) / hx**2
            if j == 0:
                b[idx] -= boundary_func(xi, 0) / hy**2
            if j == Ny - 1:
                b[idx] -= boundary_func(xi, 1) / hy**2

    # Solve the system
    u_vec = splinalg.spsolve(A, b)

    # Reshape to 2D
    U = u_vec.reshape((Nx, Ny))

    # Add boundary for plotting
    U_full = np.zeros((Nx + 2, Ny + 2))
    U_full[1:-1, 1:-1] = U

    # Fill boundary
    for i in range(Nx + 2):
        xi = i * hx
        U_full[i, 0] = boundary_func(xi, 0)
        U_full[i, -1] = boundary_func(xi, 1)

    for j in range(Ny + 2):
        yj = j * hy
        U_full[0, j] = boundary_func(0, yj)
        U_full[-1, j] = boundary_func(1, yj)

    # Full mesh for plotting
    x_full = np.linspace(0, 1, Nx + 2)
    y_full = np.linspace(0, 1, Ny + 2)
    Xf, Yf = np.meshgrid(x_full, y_full, indexing='ij')

    return Xf, Yf, U_full