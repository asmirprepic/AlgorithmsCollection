import numpy as np
from scipy.stats import norm

def generate_uniform_samples(mean, std, rho, n_samples=1000):
    """Generate uniform samples from a bivariate normal distribution."""
    correlation_matrix = np.array([[1, rho], [rho, 1]])
    normal_samples = np.random.multivariate_normal([0, 0], correlation_matrix, size=n_samples)
    uniform_samples = norm.cdf(normal_samples)
    return uniform_samples

def transform_to_marginals(uniform_samples, params1, params2):
    """Transform uniform samples to target marginals using inverse CDFs."""
    marginal1 = norm.ppf(uniform_samples[:, 0], loc=params1['mean'], scale=params1['std'])
    marginal2 = norm.ppf(uniform_samples[:, 1], loc=params2['mean'], scale=params2['std'])
    return marginal1, marginal2

def simulate_temperature_electricity(mean_temp, std_temp, mean_elec, std_elec, rho, n_samples=1000):
    """Simulate temperature and electricity usage."""
    uniform_samples = generate_uniform_samples(0, 0, rho, n_samples)
    temperatures, electricities = transform_to_marginals(
        uniform_samples,
        {'mean': mean_temp, 'std': std_temp},
        {'mean': mean_elec, 'std': std_elec}
    )
    return temperatures, electricities

# Example usage:
np.random.seed(42)
mean_temperature = 70  # Mean temperature (F)
std_temperature = 15   # Standard deviation of temperature
mean_electricity = 200 # Mean electricity usage (kWh)
std_electricity = 50   # Standard deviation of electricity usage
correlation = 0.8      # Assumed correlation between temperature and electricity usage

temperatures, electricities = simulate_temperature_electricity(
    mean_temperature, std_temperature, mean_electricity, std_electricity, correlation
)

# Output the first 10 samples to see the results
print("Sampled Temperatures and Electricity Usage:")
print(list(zip(temperatures[:10], electricities[:10])))
