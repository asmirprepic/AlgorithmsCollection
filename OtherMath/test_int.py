import numpy as np

# Define the velocity function
def velocity(t):
    return 3*t**2 + 2*t + 1

# Define the time interval
t_start = 0
t_end = 10
num_points = 1000  # Number of points for numerical integration

# Create an array of time points
t = np.linspace(t_start, t_end, num_points)

# Compute the corresponding velocity values
v = velocity(t)

# Use the trapezoidal rule to estimate the area under the curve
distance = np.trapz(v, t)

print(f"The estimated distance traveled is {distance:.2f} meters.")
