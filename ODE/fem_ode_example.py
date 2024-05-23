import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 10           # Length of the beam in meters
E = 210e9        # Young's modulus in Pa (steel)
I = 8.333e-6     # Moment of inertia in m^4
q = 1000         # Uniform load in N/m

# Number of elements
n_elements = 10
n_nodes = n_elements + 1
element_length = L / n_elements

# Generate mesh
node_positions = np.linspace(0, L, n_nodes)

# Initialize global stiffness matrix and load vector
K_global = np.zeros((n_nodes, n_nodes))
f_global = np.zeros(n_nodes)

# Element stiffness matrix and load vector for a uniform beam element
for i in range(n_elements):
    node1 = i
    node2 = i + 1
    k_local = (E * I / element_length**3) * np.array([
        [12, 6 * element_length, -12, 6 * element_length],
        [6 * element_length, 4 * element_length**2, -6 * element_length, 2 * element_length**2],
        [-12, -6 * element_length, 12, -6 * element_length],
        [6 * element_length, 2 * element_length**2, -6 * element_length, 4 * element_length**2]
    ])
    f_local = (q * element_length / 2) * np.array([1, element_length / 2, 1, element_length / 2])
    
    # Assemble into global stiffness matrix and load vector
    for local, global_ in enumerate([node1, node2]):
        f_global[global_] += f_local[local]
        for local2, global2 in enumerate([node1, node2]):
            K_global[global_, global2] += k_local[local, local2]

# Apply boundary conditions (simply supported: zero displacement at x=0 and x=L)
K_global[0, :] = K_global[-1, :] = 0
K_global[0, 0] = K_global[-1, -1] = 1
f_global[0] = f_global[-1] = 0

# Solve the system of equations
displacements = np.linalg.solve(K_global, f_global)

# Plotting the results
plt.plot(node_positions, displacements, 'bo-')
plt.xlabel('Position along the beam (m)')
plt.ylabel('Deflection (m)')
plt.title('Deflection of the Beam under Uniformly Distributed Load')
plt.grid(True)
plt.show()
