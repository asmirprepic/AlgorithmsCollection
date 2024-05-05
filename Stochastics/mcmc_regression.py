import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # for better visualization of distributions

# Generate synthetic data
np.random.seed(0)
x = 10 * np.random.rand(100)
true_intercept = 1
true_slope = 2
y = true_intercept + true_slope * x + np.random.randn(100)

# Prior distributions for the parameters (intercept and slope)

# Gaussian prior
def prior(intercept, slope):
    mean_intercept = 5
    var_intercept = 1
    mean_slope = 2.5
    var_slope = 1

    intercept_prior = np.exp(-0.5 * ((intercept - mean_intercept) ** 2) / var_intercept)
    slope_prior = np.exp(-0.5 * ((slope - mean_slope) ** 2) / var_slope)

    return intercept_prior * slope_prior


# Likelihood function
def likelihood(intercept, slope, x, y):
    predicted = intercept + slope * x
    return np.prod(np.exp(-(y - predicted) ** 2 / 20))

# Metropolis-Hastings MCMC algorithm
def metropolis_hastings(x, y, iterations=5000):
    intercept_current = 0
    slope_current = 0
    samples = []

    for i in range(iterations):
        # Propose new position
        intercept_proposed = intercept_current + np.random.normal(0, 0.5)
        slope_proposed = slope_current + np.random.normal(0, 0.5)

        # Compute likelihood ratio
        likelihood_current = likelihood(intercept_current, slope_current, x, y)
        likelihood_proposed = likelihood(intercept_proposed, slope_proposed, x, y)
        prior_current = prior(intercept_current, slope_current)
        prior_proposed = prior(intercept_proposed, slope_proposed)

        # Compute acceptance probability
        p_accept = (likelihood_proposed * prior_proposed) / (likelihood_current * prior_current)

        # Accept or reject the new state
        if np.random.rand() < p_accept:
            intercept_current = intercept_proposed
            slope_current = slope_proposed

        samples.append((intercept_current, slope_current))
    
    return np.array(samples)

# Running the MCMC algorithm
samples = metropolis_hastings(x, y, iterations=5000)
intercept_samples, slope_samples = samples[:, 0], samples[:, 1]

# Plotting the results
plt.figure(figsize=(10, 5))
plt.scatter(x, y, c='blue', label='Data')
sample_intercept, sample_slope = np.mean(samples, axis=0)
plt.plot(x, sample_intercept + sample_slope * x, c='red', label=f'Estimated line: y = {sample_intercept:.2f} + {sample_slope:.2f}x')
plt.title('Data and Fitted Line')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()


plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
sns.histplot(intercept_samples, kde=True, color='blue')
plt.title('Distribution of Intercept')
plt.xlabel('Intercept')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
sns.histplot(slope_samples, kde=True, color='green')
plt.title('Distribution of Slope')
plt.xlabel('Slope')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Optionally, show trace plots to assess convergence
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(intercept_samples, color='blue')
plt.title('Trace of Intercept')
plt.xlabel('Iteration')
plt.ylabel('Intercept Value')

plt.subplot(1, 2, 2)
plt.plot(slope_samples, color='green')
plt.title('Trace of Slope')
plt.xlabel('Iteration')
plt.ylabel('Slope Value')

plt.tight_layout()
plt.show()
