import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(20, 2)
y = np.array([1] * 10 + [-1] * 10)
X[:10] += 1
X[10:] -= 1

# Define the kernel function
def linear_kernel(x1, x2):
    return np.dot(x1, x2)

# Calculate the Gram matrix
m, n = X.shape
K = np.zeros((m, m))
for i in range(m):
    for j in range(m):
        K[i, j] = linear_kernel(X[i], X[j])

# Define the quadratic programming problem
P = np.outer(y, y) * K
q = -np.ones(m)
A = y.reshape(1, -1)
b = np.zeros(1)
G = -np.eye(m)
h = np.zeros(m)

# Solve the quadratic programming problem using gradient descent
def objective(alpha):
    return 0.5 * np.dot(alpha, np.dot(P, alpha)) - np.sum(alpha)

def gradient(alpha):
    return np.dot(P, alpha) - np.ones(m)

def gradient_descent(alpha_init, learning_rate=0.001, max_iter=10000):
    alpha = alpha_init
    for _ in range(max_iter):
        grad = gradient(alpha)
        alpha -= learning_rate * grad
        alpha = np.clip(alpha, 0, None)
        if np.linalg.norm(grad) < 1e-5:
            break
    return alpha

# Initialize alphas and solve
alpha_init = np.zeros(m)
alphas = gradient_descent(alpha_init)

# Support vectors have non-zero Lagrange multipliers
support_vectors = alphas > 1e-5
w = np.sum(alphas[support_vectors, np.newaxis] * y[support_vectors, np.newaxis] * X[support_vectors], axis=0)
b = np.mean(y[support_vectors] - np.dot(X[support_vectors], w))

# Plot the data and decision boundary
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.7)
plt.scatter(X[support_vectors, 0], X[support_vectors, 1], edgecolors='k', facecolors='none', label='Support Vectors')
x_plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
y_plot = -(w[0] * x_plot + b) / w[1]
plt.plot(x_plot, y_plot, 'k-', label='Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Support Vector Machine using NumPy')
plt.legend()
plt.show()

# Print final model parameters
print(f'Weights: {w}')
print(f'Bias: {b}')
