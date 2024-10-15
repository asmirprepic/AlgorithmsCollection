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

def gaussian_process(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, length_scale: float = 1.0, sigma_f: float = 1.0, sigma_y:float =1e-8) -> Tuple[np.ndarray, np.ndarray]:
    """  
    Perform gaussian process regressinon 
    Parameters: 
    X_train: training_data (n_samples x n_features)
    y_train: Training data outputs (n_samples x 1)
    X_test: Test data inputs (n_test_samples x n_features)
    length_scale: length scale
    sigma_f: Signal variance
    sigma_y: noise variance

    Returns: 
    Tuple:
      mean of the posterior distribution
      covariance of posterior distribution
      
    """
    K = rbf_kernel(X_train,X_train,length_scale, sigma_f)  + sigma_y**2 * np.eye(len(X_train))
    K_s = rbf_kernel(X_train,X_test, length_scale,sigma_f)
    K_ss = rbf_kernel(X_test,X_test, length_scale,sigma_f) + 1e-8*np.eye(len(test))

    K_inv = np.linalg.inv(K)
    mu_s = K_s.T.dot(K_inv).dot(y_train)

    cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

    return mu_s, cov_s

def f(x:np.ndarray) -> np.ndarray:
  """
  Generating synthetic data  for sin function


  """

  return np.sin(x)
