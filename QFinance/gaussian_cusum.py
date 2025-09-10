import numpy as np

class GaussianCUSUM:
    """
    Sequential change detector for a mean shift in Gaussian data.
    H0: X_t ~ N(mu0, sigma2)   vs   H1: X_t ~ N(mu0 + delta, sigma2)
    Uses one-sided CUSUM (increase in mean). For two-sided, keep two CUSUMs.
    """
    def __init__(self, mu0=0.0, sigma=1.0, delta=0.5, h=5.0):
        """
        mu0   : pre-change mean under H0
        sigma : known std dev
        delta : detectable shift in mean (H1 mean = mu0 + delta)
        h     : decision threshold (≈ controls false-alarm rate)
        """
        self.mu0 = float(mu0)
        self.s2 = float(sigma)**2
        self.delta = float(delta)
        self.h = float(h)

        self.k = self.delta / self.s2
        self.c = 0.5 * (self.delta**2) / self.s2
        self.S = 0.0  # CUSUM stat
        self.t_alarm = None

    def update(self, x):
        """
        Feed one new observation x. Returns (alarm: bool, S: float).
        """
        # l_t = (delta / sigma^2) * (x_t - mu0 - delta/2)
        lt = self.k * (x - self.mu0) - self.c
        self.S = max(0.0, self.S + lt)
        alarm = self.S > self.h
        if alarm and self.t_alarm is None:
            self.t_alarm = True
        return alarm, self.S

    def reset(self):
        self.S = 0.0
        self.t_alarm = None
