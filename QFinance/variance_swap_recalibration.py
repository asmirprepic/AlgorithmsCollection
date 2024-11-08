import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def ewma_variance(data. pd.Series, decay_factor: float) -> pd.Series: 
  """
  Calculate exponential moving average for a series of returns. 

  Parameters: 
  ------------
  data: Daily returns series
  decay_factor: Decay factor for EWMA

  Returns: 
  -------------
  EMWA variance estimate

  """

  return data.ewm(alpha = 1- decay_factor).var()

def filter_jumps(data: pd.Series, threshold: float =2.5) -> pd.Series: 
  """
  Filter out extreme jumps based on  a Z-score

  Parameters: 
  -------------
  data: Daily returns 
  threshold: Z-score for the filter

  Returns: 
  -----------
  Series with jumps filtered

  """
  z_scores = (data-data.mean())/data.std()
  filtered_data = data.copy()
  filtered_data[np.abs(z_scores)> threshold] = np.nan
  return filtered_data.ffill().bfill()

def backtest_variance_forecast(returns: pd.Series, decay_factor: float, threshold:float) -> pd.DataFrame: 
  """  
  Backtest variance forecast with forcast vs realized

  Parameters:
  -------------
  returns: returns data
  decay_factor: decay for emwa
  threshold: threshold for filtering
  """
  filtered_data = filter_jumps(returns, threshold= threshold)
  forecasted_var =emwa_variance(filtered_data ,decay_factor)

  realized_var = returns.rolling(window = 21).var()
  tracking_error = np.sqrt(mean_squared_error(forecasted_var.dropna(),realizede_var.dropna()))

  return pd.DataFrame({
    'ForecastedVariance': forecasted_var,
    'RealizedVariance': realized_var,
    'Trackingerror': tracking_error
  })


def price_variance_swaps(forecasted_variance: float, strike: float, notional: float) -> float:
  """
  Caclulate payoff for variance swap

  Parameters: 
  ------------
  forecasted_variance: Forecasted variance for the model
  strike: Strike variance for the swap
  notional: Notional amount

  Returns: 
  ------------
  The payoff of variance swap
  """

  return notional*(forecasted_variance-strike)
  
  






