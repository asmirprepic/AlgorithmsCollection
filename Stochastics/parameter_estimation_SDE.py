import numpy as np

# Example observed data (prices and time)
S = np.array([100, 102, 101, 105, 107, 106])  # Simulated stock prices
T = np.array([0, 1, 2, 3, 4, 5])  # Time points

# Time intervals (assumed to be uniform)
dt = np.diff(T)

# Log returns
log_returns = np.diff(np.log(S))

# Initial guesses for mu and sigma
mu = 0.01
sigma = 0.1

# Hyperparameters for gradient descent
learning_rate = 0.01
tolerance = 1e-6
max_iterations = 10000

# Negative log-likelihood function for GBM
def negative_log_likelihood(log_returns, dt, mu, sigma):
    nll = np.sum((log_returns - (mu - 0.5 * sigma**2) * dt)**2 / (2 * sigma**2 * dt) + np.log(sigma * np.sqrt(2 * np.pi * dt)))
    return nll

# Gradient of negative log-likelihood with respect to mu
def grad_mu(log_returns, dt, mu, sigma):
    return np.sum((log_returns - (mu - 0.5 * sigma**2) * dt) * (-dt) / (sigma**2 * dt))

# Gradient of negative log-likelihood with respect to sigma
def grad_sigma(log_returns, dt, mu, sigma):
    term1 = np.sum((log_returns - (mu - 0.5 * sigma**2) * dt) * (sigma * dt - (log_returns - (mu - 0.5 * sigma**2) * dt)) / (sigma**3 * dt))
    term2 = np.sum(dt / sigma)
    return term1 + term2

# Gradient descent loop
for i in range(max_iterations):
    # Calculate the gradients
    grad_mu_val = grad_mu(log_returns, dt, mu, sigma)
    grad_sigma_val = grad_sigma(log_returns, dt, mu, sigma)
    
    # Update parameters using gradients
    mu -= learning_rate * grad_mu_val
    sigma -= learning_rate * grad_sigma_val
    
    # Ensure sigma remains positive
    sigma = max(sigma, 1e-5)
    
    # Calculate the current negative log-likelihood
    nll = negative_log_likelihood(log_returns, dt, mu, sigma)
    
    # Print progress every 1000 iterations
    if i % 1000 == 0:
        print(f"Iteration {i}: NLL = {nll:.6f}, mu = {mu:.6f}, sigma = {sigma:.6f}")
    
    # Check for convergence
    if np.abs(learning_rate * grad_mu_val) < tolerance and np.abs(learning_rate * grad_sigma_val) < tolerance:
        print("Converged")
        break

print(f"Estimated mu: {mu}")
print(f"Estimated sigma: {sigma}")
