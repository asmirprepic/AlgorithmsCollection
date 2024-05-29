import numpy as np
from scipy.stats import poisson

def clayton_copula_poisson(lambda1, lambda2, theta, size):
    # Step 1: Generate independent uniform random variables
    U1 = np.random.uniform(size=size)
    U2 = np.random.uniform(size=size)

    # Step 2: Apply the Clayton copula to introduce dependence
    V = (U2**(-theta) - 1) * (U1**(-theta) - 1) + 1
    U2_dependent = V**(-1/theta)

    # Step 3: Transform the uniform variables to Poisson random variables
    X1 = poisson.ppf(U1, mu=lambda1)
    X2 = poisson.ppf(U2_dependent, mu=lambda2)

    return X1, X2

# Parameters
lambda1 = 5  # Mean of the first Poisson random variable
lambda2 = 10  # Mean of the second Poisson random variable
theta = 0.1  # Parameter of the Clayton copula (theta > 0)
size = 1000  # Number of samples

# Generate dependent Poisson random variables using Clayton copula
X1, X2 = clayton_copula_poisson(lambda1, lambda2, theta, size)

# Visualize the result
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(X1, X2, alpha=0.5, marker='o')
plt.title('Dependent Poisson Random Variables Using Clayton Copula')
plt.xlabel('Poisson Random Variable X1')
plt.ylabel('Poisson Random Variable X2')
plt.grid(True)
plt.show()
