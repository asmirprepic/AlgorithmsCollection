import numpy as np
import matplotlib.pyplot as plt

def simulate_birth_death(n_initial, lambda_0, mu_0, K, T, dt):
    """
    Simulate a birth-death process with dependency on population size.
    
    :param n_initial: Initial population size
    :param lambda_0: Base birth rate per individual
    :param mu_0: Base death rate per individual
    :param K: Carrying capacity
    :param T: Total time to simulate
    :param dt: Time step
    :return: Time series of population sizes
    """
    times = np.arange(0, T, dt)
    population_sizes = [n_initial]
    current_population = n_initial

    for t in times[1:]:
        current_lambda = lambda_0 * (1 - current_population / K)
        current_mu = mu_0 * (1 + current_population / K)
        if current_population > 0:
            # Calculate number of births and deaths
            births = np.random.poisson(current_lambda * current_population * dt)
            deaths = np.random.poisson(current_mu * current_population * dt)
            current_population += births - deaths
            current_population = max(current_population, 0)  # Ensure no negative population
        population_sizes.append(current_population)
    
    return times, population_sizes

# Simulation parameters
n_initial = 100
lambda_0 = 0.1
mu_0 = 0.05
K = 500
T = 50
dt = 0.1

# Run simulation
times, populations = simulate_birth_death(n_initial, lambda_0, mu_0, K, T, dt)

# Plotting results
plt.figure(figsize=(10, 6))
plt.plot(times, populations, label='Population over Time')
plt.xlabel('Time')
plt.ylabel('Population Size')
plt.title('Complex Birth-Death Process Simulation')
plt.legend()
plt.grid(True)
plt.show()
