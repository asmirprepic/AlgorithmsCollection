import numpy as np

class RejectionSampler:
  def __init__(self,target_pdf,proposal_sampler,proposal_pdf,c):
    """
    Initialize the rejection sampler class

    Parameters:
    - target_pdf: function
      The target pdf that we want to sample from
    - proposal sampler: function
      Generating samplers from proposal distribution
    - proposal_pdf: function
      Probability density function for the proposal pdf
    - c: float
      The constant such that c * proposal_pdf(x) >= target_pdf(x) for all x
    """
    self.target_pdf = target_pdf
    self.proposal_sampler = proposal_sampler
    self.proposal_pdf = proposal_pdf
    self.c = c

  def sample(self,n_samples):
    """
    Generates samples from target distribution. 

    Parameters: 
      -n_sample: int
      number of smples to generate from target

    Returns:
      - samples: list
        A list of samples drawn from the target distribution
    """

    samples = []
    while len(samples) < n_samples:
      # Generate samples from the proposal distribution
      x = self.proposal_sampler()
      # Calculate probability of accepting
      u = np.random.uniform(0,self.c * self.proposal_pdf(x))
      # accept or reject
      if u <= self.target_pdf(x):
        samples.append(x)
    return samples

    
