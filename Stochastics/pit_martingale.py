import numpy as np

class PITCalibrationMartingale:
    """
    Online calibration test for PIT values u_t in (0,1).
    Uses a mixture of 'power' test martingales:
        M_λ(t) = ∏_{s≤t} λ * u_s^{λ-1},  for λ in Λ ⊂ (0,1)
    Combined wealth M(t) = max_λ M_λ(t); anytime-valid p-value p_t = 1 / sup_{s≤t} M(s).
    If p_t gets small (e.g., < 0.01), reject calibration.

    Args:
        lambdas: grid in (0,1); smaller λ is sensitive to small PITs (left-tail miscal).
    """
    def __init__(self, lambdas=None):
        self.lams = np.array(lambdas if lambdas is not None else np.linspace(0.05, 0.95, 19))
        self.W = np.ones_like(self.lams, dtype=float)
        self.W_sup = 1.0
        self.t = 0

    def update(self, u: float):
        """
        Feed a PIT u in (0,1). Returns dict with current stats.
        """
        if not (0.0 < u < 1.0):
            raise ValueError("u must be strictly between 0 and 1")
        self.t += 1
        # One-step growth for each λ: g_λ(u) = λ * u^{λ-1}
        growth = self.lams * (u ** (self.lams - 1.0))
        self.W *= growth
        Wmax = float(self.W.max())
        self.W_sup = max(self.W_sup, Wmax)
        p_anytime = 1.0 / self.W_sup
        return {"t": self.t, "W_max": Wmax, "p_anytime": p_anytime}

    def reset(self):
        self.W[:] = 1.0
        self.W_sup = 1.0
        self.t = 0
