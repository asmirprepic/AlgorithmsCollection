def path_integral_pricing(S0, T, r, sigma, payoff_function, n_steps=1000, n_simulations=10000):
    """
    Price an exotic option using path integrals.
    """
    dt = T / n_steps
    paths = np.zeros((n_simulations, n_steps))
    paths[:, 0] = S0

    for t in range(1, n_steps):
        z = np.random.normal(size=n_simulations)
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)

    payoffs = np.array([payoff_function(path) for path in paths])
    return np.exp(-r * T) * np.mean(payoffs)
