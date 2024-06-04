import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parameters for the NHPP
lambda_0 = 0.5  # Scale parameter
beta = 1.5      # Shape parameter
T = 10          # Total time duration for the simulation

# Function to generate inter-arrival times for NHPP using the inverse method
def generate_nhpp_event_times(lambda_0, beta, T):
    t = 0
    event_times = []
    while t < T:
        u = np.random.uniform()
        t += (-np.log(u) / (lambda_0 * (t / 1)**(beta - 1)))**(1 / beta)
        if t < T:
            event_times.append(t)
    return np.array(event_times)

# Simulate NHPP event times
event_times = generate_nhpp_event_times(lambda_0, beta, T)

print(f"Number of events: {len(event_times)}")
print(f"Event times: {event_times}")

# Generate the number of events up to each time point
event_counts = np.arange(1, len(event_times) + 1)

# Create time axis for plotting (including time 0)
times = np.concatenate(([0], event_times))
counts = np.concatenate(([0], event_counts))

# Plot the NHPP
plt.figure(figsize=(10, 6))
plt.step(times, counts, where='post')
plt.xlabel('Time')
plt.ylabel('Number of Events')
plt.title('Non-Homogeneous Poisson Process (NHPP) Simulation')
plt.grid(True)
plt.show()


# Hypothetical failure data (event times)
failure_data = event_times

# Log-likelihood function for NHPP (Power Law Process)
def nhpp_log_likelihood(params, data, T):
    lambda_0, beta = params
    n = len(data)
    term1 = n * np.log(lambda_0) + (beta - 1) * np.sum(np.log(data))
    term2 = np.sum((data / 1)**beta) * lambda_0
    term3 = -(lambda_0 / beta) * T**beta
    return -(term1 - term2 + term3)

# Initial parameter guesses
initial_params = [0.5, 1.5]

# MLE estimation
result = minimize(nhpp_log_likelihood, initial_params, args=(failure_data, T), bounds=((1e-6, None), (1e-6, None)))
lambda_0_est, beta_est = result.x

print(f"Estimated lambda_0: {lambda_0_est:.4f}")
print(f"Estimated beta: {beta_est:.4f}")


# Reliability function R(t) for NHPP (Power Law Process)
def reliability_function(t, lambda_0, beta):
    return np.exp(-(lambda_0 / beta) * t**beta)

# Hazard rate function lambda(t) for NHPP (Power Law Process)
def hazard_rate_function(t, lambda_0, beta):
    return lambda_0 * (t / 1)**(beta - 1)

# Generate time points
time_points = np.linspace(0, T, 1000)

# Calculate reliability and hazard rate
reliability = reliability_function(time_points, lambda_0_est, beta_est)
hazard_rate = hazard_rate_function(time_points, lambda_0_est, beta_est)

# Plot reliability function
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(time_points, reliability, label='Reliability Function')
plt.xlabel('Time')
plt.ylabel('Reliability')
plt.title('Reliability Function over Time')
plt.legend()
plt.grid(True)

# Plot hazard rate function
plt.subplot(2, 1, 2)
plt.plot(time_points, hazard_rate, label='Hazard Rate Function', color='orange')
plt.xlabel('Time')
plt.ylabel('Hazard Rate')
plt.title('Hazard Rate Function over Time')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()



