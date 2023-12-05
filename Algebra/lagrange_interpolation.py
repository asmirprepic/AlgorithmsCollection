import numpy as np

def lagrange_interpolation(x_points, y_points, x):
    """
    Interpolate using the Lagrange polynomial.

    Parameters:
    x_points (list): x-coordinates of data points.
    y_points (list): y-coordinates of data points.
    x (float): The x-value to interpolate.

    Returns:
    float: Interpolated y-value at x.
    """
    sum = 0
    n = len(x_points)
    for i in range(n):
        term = y_points[i]
        for j in range(n):
            if i != j:
                term = term * (x - x_points[j]) / (x_points[i] - x_points[j])
        sum += term

    return sum

# Example usage
x_points = [0, 1, 2]
y_points = [1, 3, 2]
x = 1.5
print(f"Interpolated value at x={x}: {lagrange_interpolation(x_points, y_points, x)}")
