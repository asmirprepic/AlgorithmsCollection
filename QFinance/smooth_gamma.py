import pandas as pd
import numpy as np

def smooth_gamma(self,gamma: float, spot_price: float,barrier_level: float,smoothing_factor: float) -> float: 
  """
  Smooth gamma adjustment by promixty to the barrier

  Parameters: 
  --------------
  gamma: The raw gamma value
  spot_price: The current spot price

  Returns:
  ----------
  float: smoothed gamma value
  """

  distance_to_barrier = abs(spot_price - barrier_level)/ self.barrier_level
  adjustment_factor = np.exp(-smoothing_factor/distance_to_barrier) if distance_to_barrier > 0 else 0
  smoothed_gamma = gamma*adjustment_factor

  return smoothed_gamma

  
