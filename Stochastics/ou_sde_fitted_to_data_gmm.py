import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


def ou_process(theta, mu, sigma, X0, n, dt):
    X = np.zeros(n)
    X[0] = X0
    for i in range(1, n):
        dW = np.sqrt(dt) * np.random.randn()
        X[i] = X[i - 1] + theta * (mu - X[i - 1]) * dt + sigma * dW
    return X

# Parameters
theta = 0.5  # Speed of mean reversion
mu = 1.0  # Long-term mean
sigma = 0.1  # Volatility
X0 = 1.5  # Initial value
n = 1000  # Number of data points
dt = 1/252  # Daily steps

# Generate synthetic data, only one iteration since only one is observed generally
np.random.seed(42)
data = simulate_ou_process(theta, mu, sigma, X0, n, dt)

# Conditions for GMM
def ou_moment_conditions(params, data, dt):
    theta, mu, sigma = params
    X_t = data[:-1]
    X_t_plus_1 = data[1:]
    
    
    m1 = (X_t_plus_1 - X_t - theta * (mu - X_t) * dt)  # Mean reversion condition
    m2 = ((X_t_plus_1 - X_t)**2 - sigma**2 * dt)  # Variance condition
    
    
    moments = np.concatenate([m1, m2])
    return moments

# Objective function to minimize w.r.t initial parameters
def gmm_objective(params, data, dt):
    moments = ou_moment_conditions(params, data, dt)
    return np.mean(moments**2)

# Initial parameter guesses
initial_params = [0.1, 0.5, 0.1]

# Bounds for the parameters, ensuring some values. 
bounds = [(0.001, 2.0), (0.0, 2.0), (0.001, 1.0)]

# Optimization 
result = opt.minimize(gmm_objective, initial_params, args=(data, dt), bounds=bounds, method='L-BFGS-B')

# Extract estimated parameters
theta_est, mu_est, sigma_est = result.x

# Display the estimated parameters
print(f"Estimated theta: {theta_est}")
print(f"Estimated mu: {mu_est}")
print(f"Estimated sigma: {sigma_est}")


