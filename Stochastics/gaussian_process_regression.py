import numpy as np
import matplotlib.pyplot as plt

def rbf_kernel(X1: np.ndarray,X2: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0) -> np.ndarray:
  """
  Radial basis function kernel definition

  Parameters: 
  X1, X2: Arrays of input points (n_samples x n_features)
  length_scale: Length scale parameter
  sigma: signal variance parameter

  Returns: 
  Covariance matrix

  """

  sqdist = np.sum(X1**2,axis =1).reshape(-1,1)+ np.sum(X2**2,axis = 1)-2*np.dot(X1,X2.T)
  return sigma_f**2*np.exp(-0.5/length_scale**2*sqdist)
  
