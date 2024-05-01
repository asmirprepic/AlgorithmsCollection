import numpy as np
from scipy.optimize import minimize

# Define the objective function
def objective(vars):
    x, y = vars
    return x**2 + 2*x + 3*y**2 + y

# Define the constraints
cons = [
    {'type': 'ineq', 'fun': lambda vars: vars[0] - 20},  # x >= 20
    {'type': 'ineq', 'fun': lambda vars: vars[1] - 30},  # y >= 30
    {'type': 'ineq', 'fun': lambda vars: 50 - (vars[0] + 2*vars[1])},  # Machine M1
    {'type': 'ineq', 'fun': lambda vars: 100 - (2*vars[0] + vars[1])}  # Machine M2
]

# Initial guesses
x0 = [20, 30]

# Perform the optimization
result = minimize(objective, x0, method='SLSQP', constraints=cons)

print("Status:", result.message)
print("Total Cost: $", round(result.fun, 2))
print("Units of P1:", round(result.x[0], 2))
print("Units of P2:", round(result.x[1], 2))
