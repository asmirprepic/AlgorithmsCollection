import numpy as np

class EGPortfolio:
    """
    Exponentiated-Gradient online portfolio (Helmbold et al., 1998).
    Inputs each step: price relatives x_t (e.g., [S1_t/S1_{t-1}, ..., SN_t/SN_{t-1}]).
    Update:
        w_i <- w_i * exp(eta * x_i / (w·x))
    then renormalize. eta>0 controls aggressiveness.
    """
    def __init__(self, n_assets: int, eta: float = 0.1):
        self.n = int(n_assets)
        self.eta = float(eta)
        self.w = np.ones(self.n) / self.n
        self.growth = 1.0  # cumulative wealth

    def update(self, x: np.ndarray):
        x = np.asarray(x, float).ravel()
        if x.size != self.n:
            raise ValueError("x must have length n_assets")
        rx = float(self.w @ x)                  # portfolio gross return this step
        self.growth *= rx
        # EG update (mirror descent on simplex with entropic mirror map)
        mult = np.exp(self.eta * x / max(rx, 1e-12))
        self.w *= mult
        self.w /= self.w.sum() + 1e-18
        return self.w, rx, self.growth

    def weights(self):
        return self.w.copy()
