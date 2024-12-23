"""
This shows the implemenation of a constrianed optimization problem in python. 
The problem to solve is the mass of a piston with contriants on volume and stress 
for the piston. 
"""

import numpy as np
from scipy.optimize import minimize

# Constants
rho = 7850  # Density of steel in kg/m^3 (example)
sigma_max = 140e6  # Maximum allowable stress in Pascal
F = 10000  # Force experienced by the piston in Newtons
V0 = 0.001  # Target volume in m^3

# Objective function: mass of the piston
def objective(x):
    r, h = x
    return rho * np.pi * r**2 * h

# Constraint equations
def constraint_volume(x):
    r, h = x
    return np.pi * r**2 * h - V0

def constraint_stress(x):
    r, h = x
    A = np.pi * r**2
    sigma = F / A
    return sigma_max - sigma

# Initial guesses
x0 = [0.05, 0.05]  # Initial guess for r and h in meters

# Define constraints and bounds
cons = ({'type': 'eq', 'fun': constraint_volume},
        {'type': 'ineq', 'fun': constraint_stress})
bounds = [(0.01, 0.1), (0.01, 0.1)]  # Bounds for r and h

# Run optimization
solution = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=cons)
r_opt, h_opt = solution.x

print(f"Optimal radius (r): {r_opt:.4f} m")
print(f"Optimal height (h): {h_opt:.4f} m")
print(f"Minimized mass: {objective(solution.x):.4f} kg")

# Plotting
import matplotlib.pyplot as plt

# Generate data for constraints visualization
r_range = np.linspace(0.01, 0.1, 100)
h_range = V0 / (np.pi * r_range**2)
stress_range = sigma_max * (np.pi * r_range**2) / F

plt.figure(figsize=(10, 5))

# Plot volume constraint
plt.subplot(1, 2, 1)
plt.plot(r_range, h_range, label='Volume constraint')
plt.plot(r_opt, h_opt, 'ro', label='Optimal point')
plt.title('Height vs Radius (Volume constraint)')
plt.xlabel('Radius (m)')
plt.ylabel('Height (m)')
plt.legend()

# Plot stress constraint
plt.subplot(1, 2, 2)
plt.plot(r_range, stress_range, label='Stress constraint')
plt.title('Stress vs Radius')
plt.xlabel('Radius (m)')
plt.ylabel('Stress (Pa)')
plt.legend()

plt.tight_layout()
plt.show()
