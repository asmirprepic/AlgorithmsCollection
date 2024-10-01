import numpy as np
import matplotlib.pyplot as plt

# Generate some synthetic data for linear regression
np.random.seed(42)

# True parameters for synthetic data
true_slope = 2.5
true_intercept = -1.0
true_sigma = 0.5

# Generate X values and noisy Y values
X = np.linspace(0, 10, 100)
Y = true_slope * X + true_intercept + np.random.normal(0, true_sigma, size=X.shape)

# Linear model
def linear_model(X, slope, intercept):
    return slope * X + intercept

# Likelihood function: assume Gaussian errors
def likelihood(Y, Y_pred, sigma):
    return np.sum(-0.5 * np.log(2 * np.pi * sigma**2) - 0.5 * ((Y - Y_pred)**2) / sigma**2)

# Prior for slope, intercept, and sigma (normal priors for slope and intercept, inverse gamma for sigma)
def log_prior(slope, intercept, sigma):
    if sigma <= 0:
        return -np.inf  # Log of invalid (negative) sigma
    # Priors: Normal(0, 10) for slope and intercept, and improper prior for sigma
    return -0.5 * (slope**2 / 10**2 + intercept**2 / 10**2) - np.log(sigma)

# Posterior distribution: prior + likelihood
def log_posterior(X, Y, slope, intercept, sigma):
    Y_pred = linear_model(X, slope, intercept)
    return log_prior(slope, intercept, sigma) + likelihood(Y, Y_pred, sigma)

# MCMC using Metropolis-Hastings
def metropolis_hastings(X, Y, n_samples=5000, burn_in=1000, step_size=0.1):
    # Starting values for slope, intercept, and sigma
    slope_current = np.random.normal(0, 1)
    intercept_current = np.random.normal(0, 1)
    sigma_current = np.abs(np.random.normal(1, 0.1))
    
    # Store the samples
    samples = np.zeros((n_samples, 3))  # 3 for slope, intercept, sigma
    
    # Current log-posterior value
    log_posterior_current = log_posterior(X, Y, slope_current, intercept_current, sigma_current)
    
    for i in range(n_samples):
        # Propose new values for slope, intercept, and sigma (random walk proposal)
        slope_proposal = np.random.normal(slope_current, step_size)
        intercept_proposal = np.random.normal(intercept_current, step_size)
        sigma_proposal = np.abs(np.random.normal(sigma_current, step_size))  # Propose positive sigma
        
        # Calculate the new log-posterior
        log_posterior_proposal = log_posterior(X, Y, slope_proposal, intercept_proposal, sigma_proposal)
        
        # Acceptance probability (log version)
        acceptance_ratio = log_posterior_proposal - log_posterior_current
        if np.log(np.random.rand()) < acceptance_ratio:
            # Accept the proposal
            slope_current = slope_proposal
            intercept_current = intercept_proposal
            sigma_current = sigma_proposal
            log_posterior_current = log_posterior_proposal
        
        # Store the samples
        samples[i] = [slope_current, intercept_current, sigma_current]
    
    # Return samples after burn-in
    return samples[burn_in:]

# Run MCMC
n_samples = 10000
samples = metropolis_hastings(X, Y, n_samples=n_samples)

# Extract samples
slope_samples = samples[:, 0]
intercept_samples = samples[:, 1]
sigma_samples = samples[:, 2]

# Plot posterior distributions for slope, intercept, and sigma
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.hist(slope_samples, bins=30, color='b', alpha=0.7)
plt.axvline(true_slope, color='r', linestyle='--', label='True value')
plt.title('Slope Samples')
plt.legend()

plt.subplot(1, 3, 2)
plt.hist(intercept_samples, bins=30, color='g', alpha=0.7)
plt.axvline(true_intercept, color='r', linestyle='--', label='True value')
plt.title('Intercept Samples')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(sigma_samples, bins=30, color='orange', alpha=0.7)
plt.axvline(true_sigma, color='r', linestyle='--', label='True value')
plt.title('Sigma Samples')
plt.legend()

plt.tight_layout()
plt.show()
