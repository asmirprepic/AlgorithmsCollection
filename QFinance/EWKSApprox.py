import numpy as np

class EWKSApprox:
    """
    EW histogram KS (approx). Choose bins over a fixed range.
    Maintain EW hist for baseline and live; KS ≈ sup|CDF_live - CDF_ref|.
    """
    def __init__(self, bins=np.linspace(-0.1, 0.1, 51), lam=0.995):
        self.edges = np.asarray(bins, float)
        self.K = len(self.edges) - 1
        self.lam = float(lam); self.a = 1 - self.lam
        self.h_ref = np.ones(self.K) / self.K
        self.h_live = np.ones(self.K) / self.K

    def _bin(self, x):
        i = np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, self.K-1)
        return i

    def update_ref(self, x: float):
        i = self._bin(x)
        self.h_ref *= self.lam; self.h_ref[i] += self.a
        self.h_ref /= self.h_ref.sum()

    def update_live(self, x: float):
        i = self._bin(x)
        self.h_live *= self.lam; self.h_live[i] += self.a
        self.h_live /= self.h_live.sum()
        # KS distance on CDFs
        F_ref = np.cumsum(self.h_ref); F_live = np.cumsum(self.h_live)
        ks = float(np.max(np.abs(F_live - F_ref)))
        return {"KS": ks}
