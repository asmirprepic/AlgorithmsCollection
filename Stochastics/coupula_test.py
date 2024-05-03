import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, expon, beta

# Set the seed for reproducibility
np.random.seed(42)

# Parameters
n_samples = 1000  # Number of samples to generate
rho = 0.6         # Correlation coefficient between the two variables
a, b = 2, 5       # Parameters for the Beta distribution

# Correlation matrix for two variables
correlation_matrix = np.array([
    [1, rho],
    [rho, 1]
])

# Generate independent standard normal random variables
standard_normals = np.random.normal(0, 1, (n_samples, 2))

# Cholesky decomposition of the correlation matrix
L = np.linalg.cholesky(correlation_matrix)

# Generate correlated normal variables
correlated_normals = standard_normals.dot(L.T)

# Transform to uniform using the CDF of the standard normal
uniforms = norm.cdf(correlated_normals)

# Transform the first variable to Exponential distribution using inverse CDF
exponential_samples = expon.ppf(uniforms[:, 0])

# Transform the second variable to Beta distribution using inverse CDF
beta_samples = beta.ppf(uniforms[:, 1], a, b)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(exponential_samples, beta_samples, alpha=0.5)
plt.title('Scatter Plot of Exponential and Beta Variables Using Gaussian Copula')
plt.xlabel('Exponential Samples')
plt.ylabel('Beta Samples')
plt.grid(True)
plt.show()
