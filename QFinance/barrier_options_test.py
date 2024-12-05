def price_barrier_option(S0, K, B, r, sigma, T, n_simulations=100000, option_type="Knock-Out"):
    """
    Price a barrier option using Monte Carlo simulation.
    
    Parameters:
    - S0: Initial spot price.
    - K: Strike price.
    - B: Barrier level (relative to S0).
    - r: Risk-free rate.
    - sigma: Volatility of the underlying.
    - T: Time to maturity in years.
    - n_simulations: Number of Monte Carlo paths.
    - option_type: Type of barrier option ("Knock-Out" or "Knock-In").
    
    Returns:
    - Option price.
    """
    dt = 1 / 252  
    n_steps = int(T / dt)
    barrier = S0 * B

    
    paths = np.zeros((n_simulations, n_steps))
    paths[:, 0] = S0
    for t in range(1, n_steps):
        z = np.random.normal(size=n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    # Check barrier condition
    if option_type == "Knock-Out":
        knocked_out = (paths.min(axis=1) < barrier)
        payoffs = np.maximum(paths[:, -1] - K, 0) * ~knocked_out
    elif option_type == "Knock-In":
        knocked_in = (paths.min(axis=1) < barrier)
        payoffs = np.maximum(paths[:, -1] - K, 0) * knocked_in
    else:
        raise ValueError("Invalid option type.")

    
    option_price = np.exp(-r * T) * np.mean(payoffs)
    return option_price
