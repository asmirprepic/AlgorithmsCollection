"""
Caculating option price with BS
"""


import numpy as np
from scipy.stats import norm

def black_scholes(S, K, T, r, sigma):
    """
    S: stock price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price
