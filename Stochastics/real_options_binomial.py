import numpy as np
import matplotlib.pyplot as plt

def real_options_binomial_tree(S: float,K: float, r: float, N: int, option_type: str = 'call',descision_type: str = 'extend') -> float: 
  """
  Real option valuation using a binomial tree

  Parameters: 
  -------------
  S: Current project value
  K: Strike price investment cost or expansion cost
  T: Time to maturity
  r: Risk free rate
  sigma: volatility of the project value
  N: number of time steps
  option_type: call or put
  descision_type: expand, abandon or something else
  """

  dt = T/N
  u = np.exp(sigma*np.sqrt(dt))
  d = 1/u
  p = (np.exp(r*dt)-d)/(u-d)

  asset_price = np.zeros((N+1),(N+1))
  asset_price[0,0] = S

  for i in range(1,N+1):
    for j in range(1,N+1):
      asset_prices[i,j] = S*(u**(i-j))*(d**j)

  option_values = np.zeros((N+1),(N+1))
  for j in range(N+1):
    if option_type == 'call':
      option_values[j,N] = max(0,asset_prices[j,N] - K)
    elif option_type == 'put':
      option_values[j,N] = max(0,K-asset_prices[j,N])

  for i in range(N-1,-1,-1):
    for j in range(j+1):
      option_values[j,i] = np.exp(-r*dt)*(p*option_values[j,i+1]+(1-p)*option_values[j+1,i+1])

  plot_binomial_tree(asset_prices,option_values,N)


def plot_binomial_tree(asset_prices: np.ndarray,option_values: np.ndarray,N: int) -> None:
  """
  Plot the binomial tree showing asset prices and option values

  Parameters: 
  -----------
  asset_prices: The binomial tree of asset prices
  option_values: The binomial tree of option values
  N: Number of steps in the binomial tree
  """

  fig,ax = plt.subplots(2,1,figsize=(10,8))

  ax[0].set_title("Asset price tree")
  for i in range(N+1):
    for j in range(i+1):
      ax[0].plot(i,asset_prices[j,i],'bo')
      ax[0].plot(i,asset_prices[j,i],f"{option_values[j,i]:.2f}",fontsize = 8)

  ax[0].set_xlabel("Time steps")
  ax[0].set_ylabel("Asset prices")

  ax[1].set_title("Option value tree")
  for i in range(N+1):
    for j in range(i+1):
      ax[1].plot(i,option_values[j,i],'ro')
      ax[1].plot(i,option_values[j,i],f"{option_values[j,i]:.2f}",fontsize = 8)
  ax[1].set_xlabel("Time steps")
  ax[1].set_ylabel("Option values")

  plt.tight_layout()
  plt.show()

  
      
