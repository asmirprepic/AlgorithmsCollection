import numpy as np

def abc_metropolis_hastings(simulated_data,observed_data, prior_sampler,proposal_sampler,distance_metric, tolerance, init,num_samples: int) -> np.ndarray:
  """
  ABC for parameter estimation when the likelihood is not known 

  Parameters: 
  --------------
  simulate_data (function): Fuction to simulate data
  observed_data (np.ndarray): The observed data
  prior_sampler(function): Functino to sample from prior distribution
  posterior_sampler(function): Function to sample from proposal distribution
  distance_metric(function): Functin to calculatte the distance between simulated and observed
  tolerance (float): Toleroance for accepting a proposal
  init (float): Initial starting value of the parameter
  num_samples(int): Number of samples to generate

  Returns: 
  np.ndarray: Array of accepted samples
  
  """

  samples = np.zeros(num_samples)
  samples[0] = init

  for t in range(1,num_samples):
    current_sample = samples[t-1]
    proposed_sample = proposal_sampler(current_sample)

    distance = distance_metric(simulated_data,observed_data)

    if distance < tolerance:
      samples[t] = proposed_sample
    else: 
      samples[t] = current_sample

  return samples
