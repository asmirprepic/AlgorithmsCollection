import numpy as np
from numpy.linalg import inv, cholesky
from scipy.stats import invwishart,multivariate_normal
from numpy.typing import NDArray
from typing import Tuple, List

class BVAR:
  def __init__(self,Y,lags = 1):
    self.Y = Y
    self.lags = lags
    self.T,self.K = Y.shape
    self.X,self.Y = self.build_lagged_matrix(Y,lags)
    self.KX = self.X.shape[1]

  def _build_lagged_matrices(self,Y: NDArray[np.float64], lags:int ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    "Construct lagged regressor matrix X and target matrix Y"
    X = [
      np.hstack(Y[t-l] for l in range(1,lags +1))
      for t in range(lags,len(Y))
    ]
    return np.array(X,Y[lags:])

  def fit(self,n_iter: int = 100, burn_in: int = 200) -> None:
    """
    Fit VAR using gibbs sampling
    """

    # Prios
    A0 = np.zeros((self.K, self.KX))
    V0 = np.eye(self.KX)*10.0
    nu0 = self.K +2 
    S0 = np.eye(self.K)

    A_samples: List[NDArray[np.float64]] = []
    Sigma_samples: List[NDArray[np.float64]] = []

    # Initial values
    A = np.zeros((self.K,self.KX))
    Sigma = np.eye(self.K)

    for it in range(n_iter):
      # Posterior
      XTX = self.X.T @ self.X
      XTY = self.X.T @ self.Y_target

      V_post_inv = inv(V0) + np.kron(inv(Sigma),XTX)
      V_post = inv(V_post_inv)
      A_post_mean = V_post @ (
        inv(V0).flatten() @ A0.flatten() + np.kron(inv(Sigma),self.X.T) @ self.Y_target.flatten()
      )
      A_flat= multivariate_normal.rvs(mean = A_post_mean, cov = V_post)
      A = A_flat.reshape(self.K,self.KX)

      residuals = self.Y_target - self.X @ A.T
      S_post = S0 + residuals.T @ residuals
      Sigma = invwishart.rvs(df = nu0 + len(self.Y_target), scale = S_post)

      if  it >= burn_in: 
        A_samples.append(A.copy())
        Sigma_samples.append(Sigma.copy())

    self.A_samples = np.array(A_samples)
    self.Sigma_samples = np.array(Sigma_samples)

      
