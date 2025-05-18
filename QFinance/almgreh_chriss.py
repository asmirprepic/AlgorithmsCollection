import numpy as np
def almgren_chriss(T:float,X:float,N:int,sigma:float,eta:float,gamma:float,alpha:float) -> dict:
  """
  Almgren Chriss for optional trading execution strategy. 
  Parameters: 
  T: Time horizon for the trade
  X: Initial number of shares
  N: Number of trading intervals
  sigma: Volatility of the stockprice
  eta: Market impact parameter
  gamma: Risk aversion coefficient
  alpha: Temporary impact parameter
  """

  dt = T/N
  kappa = np.sqrt(gamma*sigma**2/eta)
  t = np.linspace(0,T,N)
  X_t = X*(np.cosh(kappa*(T-t))/np.cosh(kappa*T))

  execution_cost = np.sum(alpha*X_t[:-1]**2*dr)
  risk_cost = 0.5*gamma*np.sum((X_t[:-1]-X_t[1:])**2)

  return {
    'execetion_trajectory': X_t,
    'execution_cost': execution_cost,
    'risk_cost': risk_cost,
    'total_cost': execution_cost + risk_cost
  }
