import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def breeden_litzenberger(calls: list , strikes: list, T: float, r: float): 
  """  
  Breeden litzenberger method to estimate risk-neutral density from call option prices

  Parameters: 
  --------------
  Calls: List or array of call prices
  Strikes: List or array of strike prices
  T: Time to expiration
  r: Risk- free rate

  Returns: 
  -------------
  Risk neutral probability density function and corresponding strike prices
  """

  strikes=np.array(strikes)
  calls = np.array(calls)

  f_call = interpr1d(strikes, calls, kind = 'cubic', fill_value = 'extrapolate')
  fine_strikes = np.linspace(min(strikes),max(strikes),500)
  fine_calls = f_call(fine_strikes)

  dC_dK = np.gradient(fine_strikes)
  d2C_d2K = np.gradient(dC_dK,fine_strikes)

  pdf = np.exp(r*T)*d2C_d2K

  return fine_strikes, pdf

  
