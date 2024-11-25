import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Simulated data for implied volatilities across strikes
strike_levels = np.linspace(80, 120, 10)  # Strike prices as % of spot
implied_vols = 0.2 + 0.05 * (strike_levels / 100 - 1)**2  # Simulated skewed volatilities

# Barrier option data testing and simulation. 
barrier_data = pd.DataFrame({
    "Spot": [100] * 5,
    "Strike": [95, 100, 105, 100, 90],
    "Barrier": [90, 110, 115, 85, 95],
    "Type": ["Knock-In", "Knock-Out", "Knock-Out", "Knock-In", "Knock-Out"],
    "TimeToMaturity": [1, 0.5, 1.5, 1, 2],  # Years
    "Notional": [1000000] * 5
})


def fit_skew_model(strike_levels, implied_vols):
    """
    Fit a cubic spline to model the implied volatility skew.
    """
    return interp1d(strike_levels, implied_vols, kind="cubic", fill_value="extrapolate")

skew_model = fit_skew_model(strike_levels, implied_vols)


def barrier_option_price(S, K, B, r, q, T, vol, option_type, notional):
    """
    Price a barrier option using the Black-Scholes formula with skew adjustments.
    """
    from scipy.stats import norm

    d1 = (np.log(S / K) + (r - q + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    
    # Standard Black-Scholes call/put value
    bs_price = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    # Apply barrier adjustment
    barrier_adj = np.exp(-2 * (r - q) * np.log(B / S) / vol**2)
    if option_type == "Knock-In":
        price = barrier_adj * bs_price
    elif option_type == "Knock-Out":
        price = bs_price * (1 - barrier_adj)
    else:
        raise ValueError("Invalid option type")
    
    return notional * price

def price_barrier_options(data, skew_model, r=0.01, q=0.0):
    """
    Price barrier options in the dataset using skew-adjusted volatilities.
    """
    prices = []
    for _, row in data.iterrows():
        vol = skew_model(row["Strike"] / row["Spot"] * 100)  # Adjusted volatility
        price = barrier_option_price(
            S=row["Spot"], K=row["Strike"], B=row["Barrier"],
            r=r, q=q, T=row["TimeToMaturity"], vol=vol,
            option_type=row["Type"], notional=row["Notional"]
        )
        prices.append(price)
    data["AdjustedPrice"] = prices
    return data

barrier_data = price_barrier_options(barrier_data, skew_model)
