
import numpy as np
import matplotlib.pyplot as plt

# Define the Rosenbrock function
def rosenbrock(x, y, a=1, b=100):
    return (a - x)**2 + b * (y - x**2)**2

# Compute the gradient using finite differences
def finite_difference_gradient(f, x, y, h=1e-5):
    grad_x = (f(x + h, y) - f(x, y)) / h
    grad_y = (f(x, y + h) - f(x, y)) / h
    return np.array([grad_x, grad_y])

# Apply box constraints
def apply_box_constraints(x, y, x_min, x_max, y_min, y_max):
    x = np.clip(x, x_min, x_max)
    y = np.clip(y, y_min, y_max)
    return x, y

# Gradient descent algorithm with box constraints
def gradient_descent(starting_point, learning_rate, num_iterations, x_min, x_max, y_min, y_max):
    x, y = starting_point
    path = [(x, y)]
    for _ in range(num_iterations):
        grad = finite_difference_gradient(rosenbrock, x, y)
        x, y = x - learning_rate * grad[0], y - learning_rate * grad[1]
        x, y = apply_box_constraints(x, y, x_min, x_max, y_min, y_max)
        path.append((x, y))
    return np.array(path)

# Parameters for the gradient descent
starting_point = np.array([-1.5, 1.5])
learning_rate = 0.001
num_iterations = 20000

# Box constraints
x_min, x_max = -1.5, 1.5
y_min, y_max = 0.5, 2.5

# Run the gradient descent algorithm
path = gradient_descent(starting_point, learning_rate, num_iterations, x_min, x_max, y_min, y_max)

# Plot the Rosenbrock function
x = np.linspace(-2, 2, 400)
y = np.linspace(-1, 3, 400)
X, Y = np.meshgrid(x, y)
Z = rosenbrock(X, Y)

plt.figure(figsize=(10, 6))
plt.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis')
plt.plot(path[:, 0], path[:, 1], 'r-', label='Optimization Path')
plt.scatter(path[:, 0], path[:, 1], c='red', s=10)
plt.title('Rosenbrock Function Optimization with Box Constraints')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.colorbar()
plt.show()
