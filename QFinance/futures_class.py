from scipy.optimize import brentq
import numpy as np

class FuturesAnalysis(Derivative):
    def __init__(self, S, K, T, r, sigma):
        super().__init__(S, K, T, r, sigma)

    def futures_price(self):
        return self.S * np.exp(self.r * self.T)

    def implied_volatility(self, market_price, tol=1e-5, max_iterations=100):
        def objective_function(vol):
            self.sigma = vol
            return self.futures_price() - market_price
        
        implied_vol = brentq(objective_function, 1e-5, 5.0, xtol=tol, maxiter=max_iterations)
        return implied_vol
