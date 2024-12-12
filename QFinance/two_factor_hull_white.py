def two_factor_hull_white_simulation(a1, a2, sigma1, sigma2, r0, T, dt, n_simulations):
    """
    Simulate interest rate paths using a two-factor Hull-White model.
    """
    n_steps = int(T / dt)
    rates = np.zeros((n_simulations, n_steps))
    rates[:, 0] = r0

    for t in range(1, n_steps):
        z1 = np.random.normal(size=n_simulations)
        z2 = np.random.normal(size=n_simulations)
        dr1 = -a1 * rates[:, t - 1] * dt + sigma1 * np.sqrt(dt) * z1
        dr2 = -a2 * rates[:, t - 1] * dt + sigma2 * np.sqrt(dt) * z2
        rates[:, t] = rates[:, t - 1] + dr1 + dr2

    return rates

