import numpy as np

class OjaPCA1:
    """
    Online top eigenvector w of Cov[x]; update: w += eta * x (x^T w), then normalize.
    Pass mean-centered x_t (or let it EW-center first).
    """
    def __init__(self, d, eta=0.01, ew_mu=0.01):
        self.w = np.random.randn(d); self.w /= np.linalg.norm(self.w) + 1e-12
        self.eta = float(eta)
        self.mu = np.zeros(d); self.alpha = float(ew_mu)

    def update(self, x: np.ndarray) -> np.ndarray:
        self.mu += self.alpha * (x - self.mu)               # EW mean
        xc = x - self.mu
        self.w += self.eta * xc * (xc @ self.w)
        nrm = np.linalg.norm(self.w) + 1e-12
        self.w /= nrm
        return self.w
