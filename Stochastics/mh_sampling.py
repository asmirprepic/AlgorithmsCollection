# %%

import numpy as np
import matplotlib.pyplot as plt

def target_pdf(x):
    """Bimodal target distribution function."""
    return 0.3 * np.exp(-(x - 1)**2) + 0.7 * np.exp(-(x - 3)**2)

def metropolis_hastings(target_pdf, iterations, initial_value, proposal_width, burn_in):
    """Metropolis-Hastings sampling with burn-in."""
    samples = []
    current = initial_value
    for i in range(iterations):
        candidate = np.random.normal(current, proposal_width)
        acceptance_probability = min(1, target_pdf(candidate) / target_pdf(current))
        if np.random.rand() < acceptance_probability:
            current = candidate
        if i >= burn_in:
            samples.append(current)
    return np.array(samples)

# Parameters
num_samples = 10000
initial_value = 2
proposal_width = 0.5
burn_in = 1000  # Example burn-in period

# Generate samples
samples = metropolis_hastings(target_pdf, num_samples + burn_in, initial_value, proposal_width, burn_in)

# Plotting
plt.hist(samples, bins=50, density=True, alpha=0.75, label='Generated Samples')
x = np.linspace(0, 4, 1000)
plt.plot(x, target_pdf(x), 'r-', lw=2, label='Target PDF')
plt.title('Random Samples Using Metropolis-Hastings')
plt.legend()
plt.show()

