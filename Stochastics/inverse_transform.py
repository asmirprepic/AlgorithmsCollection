"""
Inverse transform using binary search
"""


import numpy as np
import matplotlib.pyplot as plt

# Exponential distribution parameter
lambda_param = 1

def exponential_cdf(x, lambda_param):
    """ CDF of the exponential distribution. """
    return 1 - np.exp(-lambda_param * x)

def exponential_inverse(u, lambda_param):
    """ Known inverse (quantile function) of the exponential distribution. """
    return -np.log(1 - u) / lambda_param

def binary_search_for_inverse(u, cdf_function, lower_bound=0, upper_bound=1000, tolerance=1e-6):
    """ Binary search for finding the inverse CDF for any distribution. """
    while upper_bound - lower_bound > tolerance:
        mid = (lower_bound + upper_bound) / 2
        if cdf_function(mid, lambda_param) < u:
            lower_bound = mid
        else:
            upper_bound = mid
    return (lower_bound + upper_bound) / 2

# Generate samples using the known inverse and binary search
np.random.seed(0)
sample_count = 10000
known_inverse_samples = [exponential_inverse(np.random.uniform(), lambda_param) for _ in range(sample_count)]
binary_search_samples = [binary_search_for_inverse(np.random.uniform(), exponential_cdf) for _ in range(sample_count)]

# Plot histograms for comparison
plt.figure(figsize=(12, 6))
plt.hist(known_inverse_samples, bins=50, alpha=0.6, label='Known Inverse', color='blue')
plt.hist(binary_search_samples, bins=50, alpha=0.6, label='Binary Search', color='green')
plt.title("Comparison of Sampling Methods")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()
