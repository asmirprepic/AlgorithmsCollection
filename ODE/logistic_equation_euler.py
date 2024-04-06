import numpy as np
import matplotlib.pyplot as plt

def euler_logistic(r, K, P0, T, dt):
    """
    Solve the logistic equation using the Euler method.
    
    Parameters:
    - r: Rate of maximum population growth.
    - K: Carrying capacity of the environment.
    - P0: Initial population size.
    - T: Total time.
    - dt: Time step size.
    
    Returns:
    - t: Array of time points.
    - P: Array of population sizes at each time point.
    """
    # Time array
    t = np.arange(0, T+dt, dt)
    
    # Initialize array to hold population sizes
    P = np.zeros(len(t))
    
    # Set initial population size
    P[0] = P0
    
    # Euler integration
    for i in range(1, len(t)):
        P[i] = P[i-1] + dt * r * P[i-1] * (1 - P[i-1] / K)
    
    return t, P

# Parameters
r = 0.1  # Rate of maximum population growth
K = 1000  # Carrying capacity
P0 = 10  # Initial population size
T = 100  # Total time
dt = 0.1  # Time step size

# Solve the logistic equation
t, P = euler_logistic(r, K, P0, T, dt)

# Plot the solution
plt.figure(figsize=(10, 6))
plt.plot(t, P, label='Population size')
plt.title('Logistic Equation Solution')
plt.xlabel('Time')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()
