
def price_skew_adjusted_barrier(S0, K, B, T, r, sigma_skew, barrier_type="Knock-Out"):
    """
    Price a barrier option with skew-adjusted volatility.
    """
    dt = 1 / 252
    n_steps = int(T / dt)
    n_simulations = 100000
    paths = np.zeros((n_simulations, n_steps))
    paths[:, 0] = S0

    # Simulate paths with local vol
    for t in range(1, n_steps):
        z = np.random.normal(size=n_simulations)
        local_vol = local_volatility(sigma_skew.keys(), sigma_skew.values(), S0)
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * local_vol**2) * dt + local_vol * np.sqrt(dt) * z
        )

    # Barrier logic
    if barrier_type == "Knock-Out":
        knocked_out = (paths.min(axis=1) < B)
        payoffs = np.maximum(paths[:, -1] - K, 0) * ~knocked_out
    elif barrier_type == "Knock-In":
        knocked_in = (paths.min(axis=1) < B)
        payoffs = np.maximum(paths[:, -1] - K, 0) * knocked_in
    else:
        raise ValueError("Invalid barrier type")

    return np.exp(-r * T) * np.mean(payoffs)
