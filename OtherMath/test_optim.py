import numpy as np
from scipy.optimize import minimize

# Define the drag and downforce functions
def drag(theta):
    D0, k1 = 0.5, 0.05
    return D0 * np.exp(k1 * theta)

def downforce(theta):
    L0, k2 = 1.2, 0.3
    return L0 * (1 - np.exp(-k2 * theta))

# Objective function (we aim to minimize drag)
def objective(theta):
    return drag(theta)

# Constraints
constraints = (
    {'type': 'ineq', 'fun': lambda theta: downforce(theta) - 0.8},  # downforce must be >= 0.8
    {'type': 'ineq', 'fun': lambda theta: 0.65 - drag(theta)}  # drag must be <= 0.65
)

# Initial guess
theta0 = [0.1]  # Start with a small angle in radians

# Bounds for theta (0 to 90 degrees in radians)
bounds = [(0, np.pi/2)]

# Perform the optimization
result = minimize(objective, theta0, method='SLSQP', bounds=bounds, constraints=constraints)

# Output results
theta_opt = result.x[0]
print(f"Optimal Spoiler Angle: {np.degrees(theta_opt):.2f} degrees")
print(f"Minimized Drag: {drag(theta_opt):.4f}")
print(f"Corresponding Downforce: {downforce(theta_opt):.4f}")
