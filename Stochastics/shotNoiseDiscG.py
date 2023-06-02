
import numpy as np
import matplotlib.pyplot as plt

# Set parameters
lambda_param = 1
T = 10


# Define the g(t) function
def g(t):
    if 0 <= t < 0.5:
        return 1
    elif t >= 0.5:
        return 0
    else:
        return 0
    
processes = []
nr_processes = 100
# Generate time values
t_values = np.linspace(0, T, num=1000)

# Define the X(t) function

for i in range(1,nr_processes):
    # Generate exponential random variables
    epsilon = np.random.exponential(scale=1/lambda_param, size=1000)
    eta = np.random.exponential(scale=1/lambda_param, size=1000)

    # Find K_1_epsilon
    cumulative_sum_epsilon = np.cumsum(epsilon)
    K_1 = np.searchsorted(cumulative_sum_epsilon, 10, side='right')

    cumulative_sum_eta = np.cumsum(epsilon)
    K_2 = np.searchsorted(cumulative_sum_eta, 0.5, side='right')

    def X(t):
        sum1 = 0
        sum2 = 0
        for k1 in range(1, K_1):
            sum1 += g(t - np.sum(epsilon[:k1]))
    
        for k2 in range(1,K_2):
            sum2 += g(t + np.sum(eta[:k2]))
        return sum1 + sum2
    
    # Compute X(t) values
    X_values = [X(t) for t in t_values]
    
    processes.append(X_values)

# Plot the sample path of X(t)
plt.plot(t_values,processes[0])
 
plt.xlabel('Time')
plt.ylabel('X(t)')
plt.title('Sample Path of X(t)')
plt.show()
