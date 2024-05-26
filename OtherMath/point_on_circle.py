
import numpy as np
import matplotlib.pyplot as plt

def nearest_point_on_circle(points, query_point, radius=1):
    # Transform points to polar coordinates
    angles = np.arctan2(points[:, 1], points[:, 0])
    circle_points = np.array([[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles])
    
    # Calculate distances to the query point
    distances = np.linalg.norm(circle_points - query_point, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index, circle_points[nearest_index]

# Generate random points around the origin
np.random.seed(42)
points = np.random.rand(100, 2) * 2 - 1  # Points in the range [-1, 1]
query_point = np.array([0.5, 0.5])

nearest_index, nearest_circle_point = nearest_point_on_circle(points, query_point)
print(f"Nearest index: {nearest_index}, Nearest point on circle: {nearest_circle_point}")

def plot_nearest_point_on_circle(points, query_point, nearest_circle_point):
    plt.figure(figsize=(10, 8))
    plt.scatter(points[:, 0], points[:, 1], label='Data Points', color='blue')
    plt.scatter(query_point[0], query_point[1], label='Query Point', color='red', marker='x', s=100)
    plt.scatter(nearest_circle_point[0], nearest_circle_point[1], label='Nearest Point on Circle', color='green', marker='o', s=100)
    
    # Plot the circle
    circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--')
    plt.gca().add_artist(circle)
    
    # Draw a line between the query point and the nearest circle point
    plt.plot([query_point[0], nearest_circle_point[0]], [query_point[1], nearest_circle_point[1]], color='black', linestyle='--')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nearest Point on Circle')
    plt.legend()
    plt.axis('equal')
    plt.show()

plot_nearest_point_on_circle(points, query_point, nearest_circle_point)

# %%
