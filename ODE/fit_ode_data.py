import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

"""
Example of fitting data to a ordinary differential equation. 
The steps include. 
1. Start with data and a proposed differential equation with an unknown parameter
2. Match a numerical solution to the differential equation to the data for for various values of the parameter
3. Use a optimization routine to find the value of the parameter that minimized the sum of the 
squared residuals between the data and the numerical solution. 
"""


# Logistic Growth Model
def logistic_growth(t, P, r, K):
    return r * P * (1 - P / K)

# Solve the logistic model numerically
def solve_logistic(t, r, K, P0):
    sol = solve_ivp(logistic_growth, [t.min(), t.max()], [P0], args=(r, K), t_eval=t, dense_output=True)
    return sol.sol(t).T[:, 0]

# Objective function to minimize (sum of squared residuals)
def objective(params, t, data):
    r, K = params
    model_results = solve_logistic(t, r, K, data[0])
    return np.sum((data - model_results) ** 2)

# Generate synthetic data
np.random.seed(42)
t_data = np.linspace(0, 10, 100)
true_r, true_K, P0 = 0.8, 500, 10
true_data = solve_logistic(t_data, true_r, true_K, P0)
noise = np.random.normal(0, 10, size=true_data.shape)
noisy_data = true_data + noise

# Fit model to data
initial_guess = [0.5, 300]
result = minimize(objective, initial_guess, args=(t_data, noisy_data), method='L-BFGS-B', bounds=[(0, None), (0, None)])

# Results
fitted_r, fitted_K = result.x
print("Estimated Parameters:")
print(f"r = {fitted_r:.4f}, K = {fitted_K:.4f}")

# Plotting results
plt.figure(figsize=(10, 6))
plt.scatter(t_data, noisy_data, color='red', label='Noisy Data')
plt.plot(t_data, solve_logistic(t_data, true_r, true_K, P0), 'k-', label='True Model')
plt.plot(t_data, solve_logistic(t_data, fitted_r, fitted_K, P0), 'b--', label='Fitted Model')
plt.title('Logistic Growth Model Fitting')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.show()
