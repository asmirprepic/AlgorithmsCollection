import numpy as np
from scipy.integrate import quad
from scipy.optimize import bisect

# Define the PDF of your distribution
def pdf(x):
    # Example: Standard normal distribution PDF
    return np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)

# Numerically integrate PDF to get CDF
def cdf(x):
    return quad(pdf, -np.inf, x)[0]

# Inverse CDF using bisection method
def inverse_cdf(u, lower_bound, upper_bound):
    # 'bisect' finds the root of the function cdf(x) - u = 0
    return bisect(lambda x: cdf(x) - u, lower_bound, upper_bound)

# Generate a random number from uniform distribution
u = np.random.uniform(0, 1)

# Apply the bisection method
# Note: Choose bounds (lower_bound and upper_bound) according to the expected support of your distribution
random_variate = inverse_cdf(u, -10, 10)  # for a normal distribution, -10 to 10 covers most practical cases

print("Random variate from the distribution:", random_variate)
