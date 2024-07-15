import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class GradientDescent:
    def __init__(self, func, learning_rate=0.01, max_iter=1000, tolerance=1e-6, epsilon=1e-8, constraints=None):
        """
        Initialize the GradientDescent optimizer with optional constraints.

        :param func: The function to minimize.
        :param learning_rate: The learning rate for the gradient descent updates.
        :param max_iter: The maximum number of iterations to perform.
        :param tolerance: The tolerance for stopping the algorithm.
        :param epsilon: A small value for numerical gradient estimation.
        :param constraints: A list of constraint functions.
        """
        self.func = func
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.epsilon = epsilon
        self.constraints = constraints if constraints else []

    def numerical_gradient(self, x, *args):
        grad = np.zeros_like(x)
        fx = self.func(x, *args)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += self.epsilon
            grad[i] = (self.func(x_eps, *args) - fx) / self.epsilon
        return grad

    def project(self, x):
        """
        Project the point x onto the feasible region defined by the constraints.

        :param x: The point to project.
        :return: The projected point.
        """
        for constraint in self.constraints:
            x = constraint(x)
        return x

    def optimize(self, start, *args):
        x = np.array(start, dtype=float)
        history = [x.copy()]
        for _ in range(self.max_iter):
            grad = self.numerical_gradient(x, *args)
            x = x - self.learning_rate * grad
            history.append(x.copy())
            if np.linalg.norm(grad) < self.tolerance:
                break
        return x, np.array(history)

# Constraint functions
def constraint1(x):
    x[0] = max(x[0], 0)  # x >= 0
    return x

def constraint2(x):
    x[1] = min(x[1], 2)  # y <= 4
    return x

def constraint3(x):
    x[2] = np.clip(x[2], 0, 4)  # 0 <= z <= 2
    return x

# Example usage with constraints
def func(x, y, z):
    return (x - 2)**2 + (y - 3)**2 + z

def testing():
# Parameters
    start = [0.0, 0.0, 0.0]
    learning_rate = 0.1
    max_iter = 1000
    tolerance = 1e-6

    # Create GradientDescent instance with constraints
    gd = GradientDescent(func, learning_rate, max_iter, tolerance, constraints=[constraint1, constraint2, constraint3])

    # Run optimization
    min_point, history = gd.optimize(start)

    # Results
    print(f"Minimum point: {min_point}")
    print(f"Minimum value: {func(*min_point)}")

    # Plotting the function and the descent path in 3D
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    X, Y, Z = np.meshgrid(np.linspace(-1, 4, 100),
                        np.linspace(0, 6, 100),
                        np.linspace(-1, 2, 100))
    values = np.array([func(x, y, z) for x, y, z in zip(X.ravel(), Y.ravel(), Z.ravel())])
    values = values.reshape(X.shape)

    ax.plot_surface(X[:, :, 0], Y[:, :, 0], values[:, :, 0], alpha=0.5, cmap='viridis')
    ax.plot(history[:, 0], history[:, 1], history[:, 2], 'ro-', label='Gradient Descent Path')
    ax.scatter(min_point[0], min_point[1], min_point[2], color='b', label='Minimum Point')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Gradient Descent Optimization with Constraints in 3D')
    ax.legend()
    plt.show()
