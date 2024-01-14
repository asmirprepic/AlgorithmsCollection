import numpy as np
from scipy.stats import norm

def black_caplet_price(notional, maturity, strike_rate, forward_rate, sigma, yield_curve_rate):
    d1 = (np.log(forward_rate / strike_rate) + 0.5 * sigma**2 * maturity) / (sigma * np.sqrt(maturity))
    d2 = d1 - sigma * np.sqrt(maturity)

    caplet_price = notional * (forward_rate * np.exp(-yield_curve_rate * maturity) * norm.cdf(d1) - 
                               strike_rate * np.exp(-yield_curve_rate * maturity) * norm.cdf(d2))
    return caplet_price

# Parameters
notional = 1000000  # $1,000,000
maturity = 1  # 1 year
strike_rate = 0.03  # 3%
forward_rate = 0.02  # 2%
sigma = 0.01  # Volatility of 1%
yield_curve_rate = 0.02  # Flat yield curve at 2%

# Calculate caplet price
caplet_price = black_caplet_price(notional, maturity, strike_rate, forward_rate, sigma, yield_curve_rate)
print(f"Caplet Price: ${caplet_price:.2f}")
