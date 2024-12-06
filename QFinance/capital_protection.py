def price_capital_protection_product(P, r, T, S0, K, sigma):
    """
    Price a capital protection product with participation in upside.
    
    Parameters:
    - P: Principal (investment amount).
    - r: Risk-free rate.
    - T: Time to maturity (years).
    - S0: Spot price of the underlying asset.
    - K: Strike price of the call option.
    - sigma: Volatility of the underlying asset.
    
    Returns:
    - Zero-Coupon Bond Cost, Call Option Cost, Participation Rate, and Total Cost.
    """
    
    Z = P * np.exp(-r * T)  # Cost of the zero-coupon bond
    
    
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_option_cost = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    
    remaining_funds = P - Z
    participation_rate = remaining_funds / call_option_cost
    return Z, call_option_cost, participation_rate
