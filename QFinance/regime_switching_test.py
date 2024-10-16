import numpy as np
import matplotlib.pyplot as plt

def simulate_regime_switching_model(T = 1, N = 252,mu1=0.05,mu2 = 0.02,sigma1=0.1,sigma2=0.3,p11=0.9,p22=0.85,S0=100) -> Tuple(np.ndarray,np.nd.array,np.ndarray):
  """
  Simulation of regime switching model with two regimes

  Parameters:

  Returns:
  """  

  # Time
  dt = T/N
  times = np.linspace(0,T,dt)

  prices = np.zeros(N)
  regimes = np.zeros(N)

  prices[0] = S0
  current_regime = 1
  regimes[0] = current_regime

  # Transition probability matrix
  P = np.array([[p11,1-p11],[1-p22,p22]])

  for t in range(1,N):
    if current_regime == 1: 
      current_regime = np.random.choice([1,2],p = P[0])
    else:
      current_regime = np.random.choice([1,2],p = P[1])
    regimes[t] = current_regime

    if current_regime ==1: 
      mu,sigma = mu1,sigma1  
    else:
      mu,sigma = mu2,sigma2

    dW = np.random.normal(0, np.sqrt(dt))
    prices[t] = prices[t-1]*np.exp((mu-0.5*sigma**2)*dt+sigma*dW)

    return time, prices, regimes
