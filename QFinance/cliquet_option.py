def price_cliquet(S0, T, r, sigma, periods, cap, floor, global_floor, nominal, simulations=10000):
    """
    Price a Cliquet option using Monte Carlo simulation.
    """
    dt = T / periods
    total_payoffs = []
    
    for _ in range(simulations):
        S = S0
        periodic_returns = []
        for _ in range(periods):
            z = np.random.normal()
            S = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
            r_i = (S / S0) - 1  # Periodic return
            capped_floor_return = min(max(r_i, floor), cap)
            periodic_returns.append(capped_floor_return)
        
        payoff = nominal * max(sum(periodic_returns), global_floor)
        total_payoffs.append(payoff)
    
    discounted_payoff = np.mean(total_payoffs) * np.exp(-r * T)
    return discounted_payoff
