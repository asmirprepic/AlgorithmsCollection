import numpy as np

def sample_empirical_distribution(empirical_data: np.array, num_samples: int =1):
    """
    Samples from an empirical distribution using inverse transform sampling.

    Parameters:
    empirical_data (array-like): Observed data from the empirical distribution.
    num_samples (int): The number of samples to draw from the empirical distribution.

    Returns:
    numpy.ndarray: Array of sampled values from the empirical distribution.
    """
    
    sorted_data = np.sort(empirical_data)

    
    n = len(empirical_data)
    empirical_cdf = np.arange(1, n + 1) / n

    
    uniform_samples = np.random.uniform(0, 1, size=num_samples)

    
    empirical_samples = np.interp(uniform_samples, empirical_cdf, sorted_data)

    return empirical_samples
