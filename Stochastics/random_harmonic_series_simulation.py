import numpy as np
import matplotlib.pyplot as plt

def random_harmonic_series(num_terms: int) -> float:
    """Simulate the random harmonic series up to num_terms."""
    series_sum = 0.0
    
    for n in range(1, num_terms + 1):
        X_n = np.random.choice([0, 1])  # Randomly choose 0 or 1 with equal probability
        series_sum += X_n / n
    
    return series_sum

# Simulate the series multiple times
def simulate_random_harmonic(num_simulations: int, num_terms: int):
    results = [random_harmonic_series(num_terms) for _ in range(num_simulations)]
    return results

# Parameters
num_simulations = 10000
num_terms = 1000

# Run the simulation
results = simulate_random_harmonic(num_simulations, num_terms)

# Plot the results
plt.hist(results, bins=50, density=True)
plt.title('Histogram of the Random Harmonic Series Sum')
plt.xlabel('Sum of Series')
plt.ylabel('Density')
plt.show()
