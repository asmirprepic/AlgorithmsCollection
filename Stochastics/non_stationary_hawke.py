import numpy as np


def time_varying_mu(t: float) -> float:
  """
  Definition of a time-varying baseline intensity-function. 

  Parameters: 
  --------------
  t: Current time
  
  Returns: 
  Baseline intensity at time t
  
  """

  return 0.2*(1+np.sin(0.2*t))

def exponential_kernel(t: float, alpha: float, beta:float) -> float:

  """
  Parameteres:
  --------------
  t: time_difference
  alpha: self-excitationf factor
  beta: decay rate

  Returns: 
  ----------
  Kernel value for the given time difference
  """
  return alpha*np.exp(-beta*t)

def simulate_nonstationary_hawkes(alpha: float, beta: float, T: float, baseline_mu) -> list:


  events = []
  t = 0
  while t < T: 
    lambda_t = baseline_mu(t) * sum(exponential_kernel(t-ti,alpha,beta) for ti in events)
    u = np.random.uniform()
    w = -np.log(u)/lambda_t
    t = t+w
    lambda_star = baseline_mu(t)* sum(exponential_kernel(t-ti,alpha,beta) for ti in events)

    d = np.random.uniform()

    if d <= lambda_star/lambda_t: 
      events.append(t)
    return events
