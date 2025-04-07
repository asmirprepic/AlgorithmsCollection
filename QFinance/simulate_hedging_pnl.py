def simulate_hedging_pnl(S0, K, B, T, r, sigma, hedge_freq, vol_scenario):
    """
    Simulate P&L variance for hedging a barrier option.
    
    """
    dt = T / hedge_freq
    n_steps = int(T / dt)
    paths = np.zeros(n_steps)
    paths[0] = S0
    pnl = []

    for t in range(1, n_steps):
        z = np.random.normal(0, 1)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

        # Hedge adjustments
        delta = barrier_option_delta(paths[t - 1], K, B, T - t * dt, r, sigma)
        hedge_pnl = delta * (paths[t] - paths[t - 1]) - 0.01 * np.abs(delta)  # Transaction cost
        pnl.append(hedge_pnl)

    return np.var(pnl)
