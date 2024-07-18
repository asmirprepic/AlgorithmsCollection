import numpy as np
import matplotlib.pyplot as plt

# Define the objective function (total distance traveled)
def total_distance(path):
    distance = 0
    for i in range(len(path) - 1):
        distance += np.linalg.norm(path[i] - path[i + 1])
    return distance

# Compute the gradient using finite differences
def finite_difference_gradient(f, path, h=1e-5):
    grad = np.zeros_like(path)
    for i in range(len(path)):
        for j in range(2):  # x and y coordinates
            path_step = np.copy(path)
            path_step[i, j] += h
            grad[i, j] = (f(path_step) - f(path)) / h
    return grad

# Apply obstacle constraints
def apply_obstacle_constraints(path, obstacles, radius):
    for i in range(len(path)):
        for obs in obstacles:
            if np.linalg.norm(path[i] - obs) < radius:
                direction = (path[i] - obs) / np.linalg.norm(path[i] - obs)
                path[i] = obs + direction * radius
    return path

# Gradient descent algorithm with obstacle constraints
def gradient_descent(starting_path, learning_rate, num_iterations, obstacles, radius):
    path = np.copy(starting_path)
    for _ in range(num_iterations):
        grad = finite_difference_gradient(total_distance, path)
        path -= learning_rate * grad
        path = apply_obstacle_constraints(path, obstacles, radius)
    return path

# Define the starting point, destination, and obstacles
start = np.array([0, 0])
end = np.array([10, 10])
obstacles = [np.array([3, 3]), np.array([7, 7])]
radius = 1.0

# Initial path (straight line)
num_points = 100
initial_path = np.linspace(start, end, num_points)

# Parameters for the gradient descent
learning_rate = 0.01
num_iterations = 1000

# Run the gradient descent algorithm
optimal_path = gradient_descent(initial_path, learning_rate, num_iterations, obstacles, radius)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(initial_path[:, 0], initial_path[:, 1], 'g--', label='Initial Path')
plt.plot(optimal_path[:, 0], optimal_path[:, 1], 'r-', label='Optimal Path')
plt.scatter(start[0], start[1], c='blue', label='Start')
plt.scatter(end[0], end[1], c='black', label='End')
for obs in obstacles:
    circle = plt.Circle(obs, radius, color='gray', alpha=0.5)
    plt.gca().add_artist(circle)
plt.title('Robot Path Optimization with Obstacle Avoidance')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid(True)
plt.show()
