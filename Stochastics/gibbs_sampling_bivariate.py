import numpy as np
import matplotlib.pyplot as plt

def metropolis_step(current, prop_std, target_conditional_pdf):
    """A single Metropolis-Hastings step to sample from a conditional distribution."""
    candidate = current + np.random.normal(0, prop_std)
    accept_ratio = target_conditional_pdf(candidate) / target_conditional_pdf(current)
    if np.random.rand() < accept_ratio:
        return candidate
    return current

def gibbs_sampling(iterations, burn_in, initial_values, prop_std):
    x, y = initial_values
    samples = np.zeros((iterations, 2))

    for i in range(iterations):
        # Define the conditional distribution for x given y
        def cond_x_given_y(x):
            return np.exp(-0.5 * ((x - np.sin(y))**2 + (y - np.cos(x))**2))
        
        # Define the conditional distribution for y given x
        def cond_y_given_x(y):
            return np.exp(-0.5 * ((x - np.sin(y))**2 + (y - np.cos(x))**2))
        
        # Sample x given y using Metropolis-Hastings
        x = metropolis_step(x, prop_std, lambda x: cond_x_given_y(x))
        samples[i, 0] = x
        
        # Sample y given x using Metropolis-Hastings
        y = metropolis_step(y, prop_std, lambda y: cond_y_given_x(y))
        samples[i, 1] = y

    return samples[burn_in:]  # Discard burn-in period

# Parameters
iterations = 10000
burn_in = 1000
initial_values = (0, 0)
prop_std = 0.1  # Proposal standard deviation for Metropolis steps

# Run Gibbs sampling
samples = gibbs_sampling(iterations, burn_in, initial_values, prop_std)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.6, marker='o')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Samples from a Complex Bivariate Distribution using Gibbs with Metropolis Steps')
plt.grid(True)
plt.show()
