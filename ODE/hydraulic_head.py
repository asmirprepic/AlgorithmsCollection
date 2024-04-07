# Number of points and spacing
N = 100
dx = L / (N - 1)
x_vals = np.linspace(0, L, N)

# Initialize the matrix (A) and right-hand side vector (b)
A = np.zeros((N, N))
b = np.zeros(N)

# Source term
source = -Q / (K * T)

# Fill the matrix A and vector b
for i in range(1, N-1):
    A[i, i-1] = 1 / dx**2
    A[i, i] = -2 / dx**2
    A[i, i+1] = 1 / dx**2
    b[i] = source

# Apply boundary conditions
A[0, 0] = A[-1, -1] = 1  # Dirichlet boundary conditions
b[0], b[-1] = h0, hL  # Boundary values

# Solve the system of linear equations
h_vals_numerical = np.linalg.solve(A, b)

# Plotting the numerical solution
plt.figure(figsize=(8, 6))
plt.plot(x_vals, h_vals_numerical, label='Numerical Solution', linestyle='-', color='orange')
plt.xlabel('Distance (m)')
plt.ylabel('Hydraulic Head (m)')
plt.title('Numerical Solution for Hydraulic Head Distribution')
plt.legend()
plt.grid(True)
plt.show()
