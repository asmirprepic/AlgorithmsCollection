import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def clayton_copula_3d_sample(theta, size):
    """Generate samples from a three-dimensional Clayton copula."""
    # Sample the first variable uniformly
    u = np.random.uniform(low=0, high=1, size=size)
    
    # Sample the second variable conditionally on the first
    conditional_w1 = (np.random.exponential(scale=1.0, size=size) / (u ** (-theta))) + 1
    v = (conditional_w1) ** (-1 / theta)
    
    # Sample the third variable conditionally on the first and second
    conditional_w2 = (np.random.exponential(scale=1.0, size=size) / ((u ** (-theta) + v ** (-theta)) ** (1 / theta))) + 1
    w = (conditional_w2) ** (-1 / theta)

    return u, v, w

# Sample from the 3D Clayton copula
theta = 2  # Dependency parameter
samples = clayton_copula_3d_sample(theta, 1000)
u_samples, v_samples, w_samples = samples

# Plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(u_samples, v_samples, w_samples, alpha=0.6)
ax.set_xlabel('U')
ax.set_ylabel('V')
ax.set_zlabel('W')
ax.set_title('3D Scatter Plot of Clayton Copula Samples')
plt.show()
