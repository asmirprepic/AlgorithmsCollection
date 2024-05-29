import numpy as np
from scipy.stats import norm, poisson

def gaussian_copula_poisson(lambda1, lambda2, rho, size):
    # Step 1: Generate independent standard normal random variables
    Z = np.random.normal(size=(2, size))

    # Step 2: Introduce correlation using Cholesky decomposition
    C = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(C)
    correlated_Z = L @ Z

    # Step 3: Convert the standard normal variables to uniform variables using CDF
    U = norm.cdf(correlated_Z)

    # Step 4: Transform the uniform variables to Poisson random variables
    X1 = poisson.ppf(U[0], mu=lambda1)
    X2 = poisson.ppf(U[1], mu=lambda2)

    return X1, X2

# Parameters
lambda1 = 5  # Mean of the first Poisson random variable
lambda2 = 10  # Mean of the second Poisson random variable
rho = 0.9  # Correlation coefficient
size = 1000  # Number of samples

# Generate dependent Poisson random variables using Gaussian copula
X1, X2 = gaussian_copula_poisson(lambda1, lambda2, rho, size)

# Visualize the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X1, X2, alpha=0.5, marker='o')
plt.title('Dependent Poisson Random Variables Using Gaussian Copula')
plt.xlabel('Poisson Random Variable X1')
plt.ylabel('Poisson Random Variable X2')
plt.grid(True)
plt.show()
