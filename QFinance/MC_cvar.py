import numpy as np

def simulate_credit_var(default_prob: float, exposure:float, recovery_rate: float,num_simulations: int = 10000) -> float: 
  """
  Simulate credit value at risk using Monte Carlo

  Parameters: 
  ------------
  default_prob: Probability of default
  exposure: Exposure at default
  recovery_rate: recovery rate in case of default
  num_simulations: Number of monte carlo simulations

  Returns: credit VaR at 99% confidence
  """
  losses = np.zeros(num_simulations)

  for i in range(num_simulations):
    if np.random.rand() < default_prob:
      loss = exposure*(1-recovery_rate)
    else:
      loss = 0
    losses[i] = loss

  credit_var = np.percentile(losses,99)
  
