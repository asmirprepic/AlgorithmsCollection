import numpy as np
import pandas as pd

class RegimeSwitchingSimulator:
  def __init__(self,n_periods = 1000, transition_matrix = None, regime_params = None, seed = None):
    """
    Parameters: 
    - n_periods: number of time steps to simulate
    - transition_matrix: 2D NumPy array of shape (n_regimes, n_regimes) with regime transition
    - regime_params: list of dicts, each with mean and std for return distributions in the regime
    - seed: seed

    """
  self.n_periods = n_periods
  self.transition_matrix = transition_matrix
  self.regime_params = regime_params
  self.n_regimes = len(regime_params)
  self.seed = seed
  if seed is not None:
      np.random.seed(seed)

def _simulate_regime_path(self):
  """Simulate the sequence of regimes using the transition matrix."""
  regimes = np.zeros(self.n_periods, dtype=int)
  regimes[0] = np.random.choice(self.n_regimes)  
  for t in range(1, self.n_periods):
      regimes[t] = np.random.choice(
          self.n_regimes,
          p=self.transition_matrix[regimes[t - 1]]
      )
  return regimes

def _simulate_returns(self, regimes):
  """Simulate returns based on the regime sequence."""
  returns = np.zeros(self.n_periods)
  for t in range(self.n_periods):
      mu = self.regime_params[regimes[t]]['mean']
      sigma = self.regime_params[regimes[t]]['std']
      returns[t] = np.random.normal(mu, sigma)
  return returns

def simulate(self):
    """Run full simulation and return a DataFrame."""
    regimes = self._simulate_regime_path()
    returns = self._simulate_returns(regimes)
    prices = 100 * np.exp(np.cumsum(returns))  # Log-returns to price path
    df = pd.DataFrame({
        'Regime': regimes,
        'Return': returns,
        'Price': prices
    })
    return df
