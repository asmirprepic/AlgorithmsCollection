import numpy as np
import matplotlib.pyplot as plt

def clayton_copula_sample(size, theta):
    """Generate samples from a Clayton copula."""
    # Step 1: Generate uniform random variables
    u = np.random.uniform(low=0, high=1, size=size)
    
    # Step 2: Generate a dependent uniform variable using Clayton's inverse CDF
    # Generate a random variable from exponential distribution
    w = np.random.exponential(scale=1/theta, size=size)
    
    # Calculate the inverse conditional distribution of v given u
    v = (1 + w * (u ** (-theta))) ** (-1/theta)
    
    return u, v

# Example of generating and plotting Clayton copula samples
theta = 2  # Higher theta implies stronger tail dependence
samples = clayton_copula_sample(1000, theta)

plt.figure(figsize=(6, 6))
plt.scatter(samples[0], samples[1], alpha=0.5)
plt.xlabel('U')
plt.ylabel('V')
plt.title('Samples from a Clayton Copula')
plt.show()
