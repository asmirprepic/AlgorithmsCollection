import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Function to load and preprocess data
def load_and_preprocess_data(dataset, target_classes, features):
    X = dataset.data[target_classes, :][:, features]
    y = dataset.target[target_classes]
    y = y.reshape(-1, 1)
    X_b = np.c_[np.ones((X.shape[0], 1)), X]  # Add bias term
    return X_b, y

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(theta, X, y):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    epsilon = 1e-5
    cost = (-1/m) * (np.dot(y.T, np.log(h + epsilon)) + np.dot((1 - y).T, np.log(1 - h + epsilon)))
    return cost

# Gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    
    for i in range(iterations):
        theta -= (learning_rate / m) * np.dot(X.T, (sigmoid(np.dot(X, theta)) - y))
        cost_history[i] = compute_cost(theta, X, y)
    
    return theta, cost_history

# Function to plot the cost function
def plot_cost_function(cost_history, iterations):
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), cost_history, 'b-')
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Function using Gradient Descent')
    plt.show()

# Function to plot the decision boundary
def plot_decision_boundary(X, y, theta):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[y.flatten() == 0][:, 1], X[y.flatten() == 0][:, 2], label='Class 0')
    plt.scatter(X[y.flatten() == 1][:, 1], X[y.flatten() == 1][:, 2], label='Class 1')
    x_value = np.array([np.min(X[:, 1]), np.max(X[:, 1])])
    y_value = -(theta[0] + theta[1] * x_value) / theta[2]
    plt.plot(x_value, y_value, 'r-')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Logistic Regression Decision Boundary')
    plt.legend()
    plt.show()

# Main function
def main():
    # Load dataset
    dataset = load_iris()
    
    # Define target classes and features to use
    target_classes = np.arange(100)  # Use only the first 100 samples
    features = [0, 1]  # Use the first two features
    
    # Preprocess data
    X_b, y = load_and_preprocess_data(dataset, target_classes, features)
    
    # Initialize parameters
    theta = np.zeros((X_b.shape[1], 1))
    learning_rate = 0.01
    iterations = 1000
    
    # Perform gradient descent
    theta, cost_history = gradient_descent(X_b, y, theta, learning_rate, iterations)
    
    # Plot the cost function
    plot_cost_function(cost_history, iterations)
    
    # Plot the decision boundary
    plot_decision_boundary(X_b, y, theta)

# Run the main function
if __name__ == '__main__':
    main()
