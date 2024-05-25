import numpy as np
import matplotlib.pyplot as plt

def nearest_neighbor_search(data, query_point):
    distances = np.linalg.norm(data - query_point, axis=1)
    nearest_index = np.argmin(distances)
    return nearest_index, data[nearest_index]

def plot_nearest_neighbor(data, query_point, nearest_point):
    plt.figure(figsize=(10, 8))
    plt.scatter(data[:, 0], data[:, 1], label='Data Points', color='blue')
    plt.scatter(query_point[0], query_point[1], label='Query Point', color='red', marker='x', s=100)
    plt.scatter(nearest_point[0], nearest_point[1], label='Nearest Neighbor', color='green', marker='o', s=100)
    
    # Draw a line between the query point and the nearest neighbor
    plt.plot([query_point[0], nearest_point[0]], [query_point[1], nearest_point[1]], color='black', linestyle='--')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Nearest Neighbor Search')
    plt.legend()
    plt.show()

# Example usage
np.random.seed(42)
data = np.random.rand(100, 2)  # Random 2D points
query_point = np.array([0.5, 0.5])

nearest_index, nearest_point = nearest_neighbor_search(data, query_point)
print(f"Nearest index: {nearest_index}, Nearest point: {nearest_point}")

# Visualize the nearest neighbor search
plot_nearest_neighbor(data, query_point, nearest_point)
