# %%


import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def empirical_cdf(data):
    sorted_data = np.sort(data)
    ranks = np.searchsorted(sorted_data, data, side='right') / len(data)
    return ranks

def clayton_log_likelihood(theta, u, v):
    """Calculate the log likelihood for the Clayton copula."""
    if theta <= 0:
        return -np.inf  # log likelihood is undefined for non-positive theta
    return np.sum(np.log((theta + 1) * (u ** (-theta - 1) + v ** (-theta - 1)) ** (-2 - 1/theta) * 
                         (u * v) ** (-theta - 1)))

def clayton_log_likelihood_derivative(theta, u, v):
    """First derivative of the log likelihood function."""
    if theta <= 0:
        return 0  # Avoid invalid operations
    term1 = np.sum((u ** (-theta - 1) + v ** (-theta - 1)) ** (-1 - 1/theta))
    term2 = np.sum(np.log(u * v) * (u ** (-theta - 1) + v ** (-theta - 1)) ** (-1 - 1/theta))
    return len(u) / theta ** 2 - term1 + term2


def clayton_log_likelihood_second_derivative(theta, u, v):
    if theta <= 0:
        return 0
    term1 = 2 * len(u) / theta ** 3
    term2 = np.sum((np.log(u) + np.log(v))**2 * (u**(-theta) + v**(-theta))**(-1/theta))
    term3 = 2 * np.sum((np.log(u) + np.log(v)) * (u**(-theta) + v**(-theta))**(-1 - 1/theta) * (u**(-theta - 1) + v**(-theta - 1)))
    term4 = np.sum((u**(-theta) + v**(-theta))**(-2 - 1/theta) * (u**(-theta - 1) + v**(-theta - 1))**2)
    return term1 - term2 - term3 - term4

def newton_raphson(u, v, initial_guess=2, max_iter=1000, tol=1e-9):
    theta = initial_guess
    for i in range(max_iter):
        derivative = clayton_log_likelihood_derivative(theta, u, v)
        second_derivative = clayton_log_likelihood_second_derivative(theta, u, v)
        if abs(derivative) < tol:
            break
        if second_derivative == 0:  # Prevent division by zero
            break
        theta -= derivative / second_derivative
        if theta <= 0:
            theta = tol  # Maintain a positive theta
    return theta


def gradient_descent(u, v, initial_guess=1, learning_rate=0.001, max_iter=10000, tol=1e-6):
    """Optimize theta using gradient descent."""
    theta = initial_guess
    for _ in range(max_iter):
        grad = clayton_log_likelihood_derivative(theta, u, v)
        if abs(grad) < tol:
            break
        theta += learning_rate * grad  # Update step
        if theta <= 0:
            theta = tol  # Keep theta positive
    return theta


# Example data
np.random.seed(0)
data1 = np.random.gamma(shape=2, scale=2, size=1000)
data2 = data1 * 0.5 + np.random.gamma(shape=2, scale=2, size=1000) * 0.5
data = np.column_stack((data1, data2))

# Transform to uniform
u = empirical_cdf(data[:, 0])
v = empirical_cdf(data[:, 1])

# Fit the copula
theta_estimated = gradient_descent(u, v)
print("Estimated Theta:", theta_estimated)


def clayton_copula_sample(theta, size):
    """Generate samples from a Clayton copula."""
    u = np.random.uniform(low=0, high=1, size=size)
    # Generate a dependent variable v from u using the inverse conditional method
    v = (1 + ((np.random.exponential(scale=1.0, size=size) / (u ** (-theta))) + 1) ** (-1 / theta))
    return u, v

# Example using the estimated theta
theta = 2  # Replace with your estimated theta
u_samples, v_samples = clayton_copula_sample(theta, 1000)

# Plotting
plt.figure(figsize=(12, 6))

# Scatter plot of the samples
plt.subplot(1, 2, 1)
plt.scatter(u_samples, v_samples, alpha=0.6)
plt.xlabel('U')
plt.ylabel('V')
plt.title('Scatter Plot of Clayton Copula Samples')

# Contour plot to show density
plt.subplot(1, 2, 2)
plt.hist2d(u_samples, v_samples, bins=30, cmap='Blues')
plt.colorbar()
plt.xlabel('U')
plt.ylabel('V')
plt.title('Density Plot of Clayton Copula Samples')

plt.tight_layout()
plt.show()
