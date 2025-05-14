import numpy as np

def importance_sampling_rare_events(n_samples: int, n_variables: int, threshold: float, mu: float = 0, sigma: float = 0, shift: float = 3) -> float: 

  """
  Estmate the probability of rare events using importance sampling. 

  Args: 
  n_samples: Number of samples to simulate
  n_variables: Number of independent variables in the sum
  threshold: Treshold of the rare event
  mu: Mean of the original normal distribution 
  sigma: Standard deviation of the original normal distribution
  shift: Shift to mean for importance sampling (higher mean to simualate rare events) 

  Returns: 
  float: Estimated probability of rare event 
  """
  mu_shifted = mu + shift
  likelihood_ratio = np.zeros(n_samples)

  for i in range(n_samples): 
    samples = np.random.normal(mu_shifted, sigma,n_variables)
    sum_samples = np.sum(samples)

    original_density = np.exp(-((samples-mu)**2)/(2*sigma**2)).prod()
    shifted_density = np.exp(-((samples-mu_shifted)**2)/(2*sigma**2)).prod()
    likelihood_ratios = original_density/shifted_density

    if sum_samples > threshold: 
      likelihood_ratios[i] = likelihood_ratio

    probability_estimate = np.mean(likelihood_ratios[likelihood_ratios > 0])

    return probability_estimate
                              
