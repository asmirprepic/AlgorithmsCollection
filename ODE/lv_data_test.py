import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Redefine the Lotka-Volterra Equations for solve_ivp
def lotka_volterra(t, y, alpha, beta, gamma, delta):
    x, y = y
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# True parameters
alpha, beta, gamma, delta = 0.4, 0.002, 0.3, 0.001

# Initial conditions
x0, y0 = 40, 9
initial_conditions = [x0, y0]

# Time points
t = np.linspace(0, 150, 300)

# Generate synthetic data using solve_ivp for consistency
sol = solve_ivp(lotka_volterra, [t[0], t[-1]], initial_conditions, t_eval=t, args=(alpha, beta, gamma, delta))
data = sol.y.T + np.random.normal(scale=0.5, size=sol.y.T.shape)  # Adding some noise

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(t, data[:, 0], 'r-', label='Prey (data)')
plt.plot(t, data[:, 1], 'b-', label='Predator (data)')
plt.legend()
plt.show()

# Objective function to minimize (sum of squared errors)
def objective(params):
    sol = solve_ivp(lotka_volterra, [t[0], t[-1]], initial_conditions, t_eval=t, args=params)
    return np.sum((data - sol.y.T) ** 2)

# Initial guess for parameters
initial_guess = [0.3, 0.001, 0.4, 0.002]

# Bounds for parameters to ensure they are positive
bounds = [(0.1, 1.0), (0.0001, 0.01), (0.1, 1.0), (0.0001, 0.01)]

# Perform the optimization using a more appropriate algorithm, if available
result = minimize(objective, initial_guess, bounds=bounds)
estimated_params = result.x

print("Estimated Parameters: alpha={}, beta={}, gamma={}, delta={}".format(*estimated_params))

# Solve with estimated parameters
sol_estimated = solve_ivp(lotka_volterra, [t[0], t[-1]], initial_conditions, t_eval=t, args=tuple(estimated_params))

# Plot original data and fitted model
plt.figure(figsize=(10, 5))
plt.plot(t, data[:, 0], 'r-', label='Prey (data)')
plt.plot(t, data[:, 1], 'b-', label='Predator (data)')
plt.plot(t, sol_estimated.y[0], 'r--', label='Prey (model)')
plt.plot(t, sol_estimated.y[1], 'b--', label='Predator (model)')
plt.legend()
plt.show()
