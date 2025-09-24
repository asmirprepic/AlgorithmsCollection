class EWMACov:
    """
    RiskMetrics EWMA covariance: S_t = λ S_{t-1} + (1-λ) x_t x_t^T
    On means: use EW mean to center. Returns S_t each update.
    """
    def __init__(self, d, lam=0.97):
        self.lam = float(lam)
        self.mu = np.zeros(d)
        self.S = np.zeros((d,d))
        self.alpha = 1 - lam

    def update(self, x: np.ndarray) -> np.ndarray:
        xc = x - self.mu
        self.mu = (1 - self.alpha) * self.mu + self.alpha * x
        self.S = self.lam * self.S + self.alpha * np.outer(xc, xc)
        return self.S
