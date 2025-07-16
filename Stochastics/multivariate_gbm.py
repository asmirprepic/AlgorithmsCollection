import numpy as np
import pandas as pd

class MVarGBM:
    def __init__(self,mu,cov,S0,T=1.0,dt = 0.01,seed = None):

        self.mu = np.array(mu)
        self.cov = np.array(cov)
        self.S0 = np.array(S0)
        self.T = T
        self.dt = dt
        self.N = int(T / dt)
        self.dim = len(S0)
        self.seed = seed

        if self.seed is not None:
            np.random.seed(self.seed)

        self.L = np.linalg.cholesky(self.cov)

    def simulate(self, n_paths=5):
        """
        Simulate multiple paths.

        Returns:
        - t : time grid
        - paths : shape (n_paths, d, N+1)
        """
        t = np.linspace(0, self.T, self.N + 1)
        paths = np.zeros((n_paths, self.dim, self.N + 1))
        paths[:, :, 0] = self.S0

        for i in range(self.N):
            z = np.random.randn(n_paths, self.dim)  # standard normal
            dW = z @ self.L.T * np.sqrt(self.dt)    # correlated Brownian motion
            drift = (self.mu - 0.5 * np.diag(self.cov)) * self.dt
            paths[:, :, i + 1] = paths[:, :, i] * np.exp(drift + dW)

        return t, paths
