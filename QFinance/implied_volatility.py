import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def black_scholes_call(S, K, T, r, sigma):
    """Price a European option using the Black-Scholes model.
    S: stock price
    K: strike price
    T: time to maturity
    r: risk-free interest rate
    sigma: volatility
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def implied_volatility(target_value, S, K, T, r, *args):
    """Calculate the implied volatility given a target price.
    target_value: the observed price of the option
    """
    def objective_function(sigma):
        return black_scholes_call(S, K, T, r, sigma) - target_value
    
    return brentq(objective_function, 1e-12, 1, *args)

# Example usage:
S = 100  # Underlying asset price
K = 100  # Option strike price
T = 1    # Time to maturity in years
r = 0.05 # Risk-free rate
observed_option_price = 10  # Observed market price of the option

iv = implied_volatility(observed_option_price, S, K, T, r)
print(f"Implied Volatility: {iv * 100:.2f}%")
