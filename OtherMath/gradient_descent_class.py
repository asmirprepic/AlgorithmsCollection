import numpy as np

class GradientDescent:
    def __init__(self, func, learning_rate=0.01, max_iter=1000, tolerance=1e-6, epsilon=1e-8):
        """
        Initialize the GradientDescent optimizer.

        :param func: The function to minimize.
        :param learning_rate: The learning rate for the gradient descent updates.
        :param max_iter: The maximum number of iterations to perform.
        :param tolerance: The tolerance for stopping the algorithm.
        :param epsilon: A small value for numerical gradient estimation.
        """
        self.func = func
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.epsilon = epsilon

    def numerical_gradient(self, x):
        """
        Estimate the gradient of the function at x using finite differences.

        :param x: The point at which to estimate the gradient.
        :return: The estimated gradient.
        """
        grad = np.zeros_like(x)
        fx = self.func(*x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += self.epsilon
            grad[i] = (self.func(*x_eps) - fx) / self.epsilon
        return grad

    def optimize(self, start):
        """
        Perform gradient descent optimization.

        :param start: The starting point for the optimization.
        :return: The found minimum point and the history of points.
        """
        x = np.array(start, dtype=float)
        history = [x.copy()]
        for _ in range(self.max_iter):
            grad = self.numerical_gradient(x)
            x = x - self.learning_rate * grad
            history.append(x.copy())
            if np.linalg.norm(grad) < self.tolerance:
                break
        return x, np.array(history)

# Example usage
def func(x, y):
    return (x - 2)**2 + (y - 3)**2

# Parameters
start = [0.0, 0.0]
learning_rate = 0.1
max_iter = 1000
tolerance = 1e-6

# Create GradientDescent instance
gd = GradientDescent(func, learning_rate, max_iter, tolerance)

# Run optimization
min_point, history = gd.optimize(start)

# Results
print(f"Minimum point: {min_point}")
print(f"Minimum value: {func(*min_point)}")

# Plotting the function and the descent path
import matplotlib.pyplot as plt

X, Y = np.meshgrid(np.linspace(-1, 4, 400), np.linspace(0, 6, 400))
Z = func(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=50)
plt.plot(history[:, 0], history[:, 1], 'ro-', label='Gradient Descent Path')
plt.scatter([min_point[0]], [min_point[1]], color='blue', label='Minimum Point')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Descent Optimization')
plt.legend()
plt.show()
