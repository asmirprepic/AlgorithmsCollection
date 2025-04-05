import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def simulate_diffusion(T,dt,kind = 'gbm',**kwargs):

  t_grid = np.arange(0, T+dt, dt)  
  N = len(grid)
  X = np.zeros(N)

  if kind == 'gbm':
    mu = kwargs.get('mu',0.1)
    sigma = kwargs.get('sigma',0.2)
    X[0] = kwargs.get('x0',1.0)

    for i in range(1,N):
      dw = np.random.normal(0,np.sqrt(dt))
      X[i] = X[i-1]*np.exp((mu-0.5*sigma**2)*dt + sigma*dw)

  elif kind == 'ou':
    theta = kwargs.get('theta', 0.7)
    mu = kwargs.get('mu', 1.0)
    sigma = kwargs.get('sigma', 0.3)
    X[0] = kwargs.get('x0', mu)

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        X[i] = X[i - 1] + theta * (mu - X[i - 1]) * dt + sigma * dW

    X = np.maximum(X, 0.0)  # Keep intensity non-negativ


  A = np.cumsum(X)*dt

return t_grid, X, A

def simulate_poisson_in_random_time(A,rate):
  T_star = A[-1]
  num_events = np.random.poisson(rate*T_star)
  event_times_A = np.sort(np.random.uniform(0,T_star,num_events))
  inverse_A = interp1d(A,np.arange(len(A)),bounds_error = False, fill_values = 'extrapolate')
  indices = inverse_A(event_times_A)
  
    
  
