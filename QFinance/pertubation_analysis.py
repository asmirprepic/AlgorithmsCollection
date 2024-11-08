import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

def simulate_market_data(strikes: np.ndarray, maturities: np.ndarray, seed: int = 42, missing_ratio:float = 0.2) -> pd.DataFrame: 
  """  
  Simulate market data with implied volatilities in a grid of strikes and maturities. 
  
  Paramteres:
  --------------
  stikes: array of strike prices
  maturities: array of maturities
  seed: seed for the random simulation
  missing_ratio: proportion of the data that is missing

  Returns: 
  -------------
  pd.DataFrame of simulated market data
  """

  np.random.seed(seed)
  vol_surface = {
    (m,k): 0.2 + 0.05*np.sin(0.1*k)*np.cos(0.05*m)+0.01*np.random.randn()
    for m in maturities for k in strikes
  }

  vol_data = pd.DataFrame([
    {'Maturities': m, 'Strike': k, 'ImpliedVol': vol_surface[(m,k)]}
    for (m,k) in vol_surface
  ])

  missing_indicies = np.random.choice(vol_data.index,size = int(len(vol_data)*missing_ratio),missing = False)
  vol_data.loc[missing_indicies,'ImpliedVol'] = np.nan

  return vol_data

def interpolate_vol_surface(data: pd.DataFrame) -> pd.DataFrame:
  """
  Interpolation of missing volatilities in the data using RBF interpolation

  Parameteres: 
  -------------
  data: Market data with missing values in the implied volatility column. 

  Returns: 
  -------------
  pd.DataFrame: Data with interpolated volatilities 
  
  """
  known_data = data.dropna()
  strikes_known = known_data['Strike'].values
  maturities_known = know_data['Maturity'].values
  vols_known = know_data['ImpliedVol'].values

  rbf_interpolate = Rbf(strikes_known,maturities_known,vols_known,functin = 'cubic')

  data['InterpolatedVol'] = data.apply(
    lambda row: rbf_interpolator(row['Strike'],row['Maturity']) if np.isnan(row['ImpliedVol']) else row['ImpliedVol'],
    axis =1
  )

  return data

def pertubation_analysis(data: pd.DataFrame,pertubation: float = 0.01 ) -> pd.DataFrame:
  """
  Pertubation analysis by shifting volatilities and observing the implied volatility

  Parameters: 
  -------------
  data: data with interpolated volatilities
  pertubation: The percentage to apply to implied volatilies
  
  Returns:
  -----------
  pd.DataFrame: Data with added sensitivity
  """

  known_data = data.dropna(subset = ['ImpliedVol'])
  strikes_known = known_data['Strike'].values
  maturities_known = known_data['Maturity'].values
  vols_known = known_data['ImpliedVol'].values

  pertubated_vols= vols_known*(1+pertubation*np.random.randn(len(vols_known)))
  pertubated_interpol = Rbf(strikes_known,maturities_known,pertubated_vols,function = 'cubic')

  data["PerturbedVol"] = data.apply(
    lambda row: perturbed_interpolator(row["Strike"], row["Maturity"]) if np.isnan(row["ImpliedVol"]) else row["ImpliedVol"], 
    axis=1
  )
  data["Sensitivity"] = np.abs(data["InterpolatedVol"] - data["PerturbedVol"])
  
  return data
                             









