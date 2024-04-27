import numpy as np

# Parameters
lambda_param = 0.5  # Rate parameter of the exponential distribution
u = np.random.uniform(0, 1)  # Uniform random number

# CDF of the exponential distribution
def F(x):
    return 1 - np.exp(-lambda_param * x)

# Analytical inverse of the CDF
def F_inv(u):
    return -np.log(1 - u) / lambda_param

# Bisection method to find the inverse CDF numerically
def bisection_method(F, u, a, b, max_iter=20):
    for _ in range(max_iter):
        mid = (a + b) / 2
        if F(mid) < u:
            a = mid
        else:
            b = mid
    return (a + b) / 2

# Apply the bisection method
a = 0
b = 10 / lambda_param  # A reasonable upper bound
random_variate = bisection_method(F, u, a, b)

print("Random variate (Bisection):", random_variate)
print("Random variate (Analytical):", F_inv(u))
