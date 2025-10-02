import numpy as np

class OnlineMMDRFF:
    """
    Online two-sample drift detector (reference vs. live) via MMD with RBF kernel.
    - Fit RFF on reference sample once (set sigma).
    - Maintain EW means of φ(x) for reference and live.
    - Alarm when ||μ_live - μ_ref||^2 exceeds a threshold.
    """
    def __init__(self, ref_sample: np.ndarray, D: int = 256, sigma: float = None, lam=0.99, seed=0):
        X = np.asarray(ref_sample, float)
        self.d = X.shape[1] if X.ndim == 2 else 1
        X = X.reshape(-1, self.d)
        if sigma is None:
            rng = np.random.default_rng(seed)
            idx = rng.choice(len(X), size=min(500, len(X)), replace=False)
            Z = X[idx]
            pd = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=2)
            med = np.median(pd[pd>0])
            sigma = med if med > 0 else 1.0
        self.sigma = float(sigma)

        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 1/self.sigma, size=(self.d, D))
        self.b = rng.uniform(0, 2*np.pi, size=D)
        self.D = D
        self.lam = float(lam)
        self.alpha = 1 - self.lam

        self.mu_ref = np.zeros(D)
        self.mu_live = np.zeros(D)
        for x in X:
            self.mu_ref = self.lam * self.mu_ref + self.alpha * self._phi(x)

    def _phi(self, x):
        x = np.asarray(x, float).ravel()
        z = x @ self.W + self.b
        return np.sqrt(2.0/self.D) * np.cos(z)

    def update(self, x, threshold: float):
        """
        Feed a live point x; return MMD^2 estimate and alarm bool.
        threshold: pick via held-out calibration; start around 3–5 * baseline value.
        """
        self.mu_live = self.lam * self.mu_live + self.alpha * self._phi(x)
        mmd2 = float(np.sum((self.mu_live - self.mu_ref)**2))
        return mmd2, (mmd2 > threshold)
