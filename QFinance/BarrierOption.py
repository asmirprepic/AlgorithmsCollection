import numpy as np
from scipy.stats import norm

class BarrierOption:
    def __init__(self, spot, strike, barrier, maturity, vol, rate, is_call=True, is_knock_out=True):
        self.S = spot
        self.K = strike
        self.H = barrier
        self.T = maturity
        self.sigma = vol
        self.r = rate
        self.is_call = is_call
        self.is_knock_out = is_knock_out

    def price(self):
        """Simplified pricing using analytical approximation for up-and-out call."""
        if self.S >= self.H:
            return 0.0 if self.is_knock_out else self._vanilla_price()
        return self._vanilla_price() * (1 - np.exp(-self.r * self.T) * (self.S / self.H) ** (2 * self.r / (self.sigma ** 2)))

    def _vanilla_price(self):
        d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        d2 = d1 - self.sigma * np.sqrt(self.T)
        if self.is_call:
            return self.S * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S * norm.cdf(-d1
