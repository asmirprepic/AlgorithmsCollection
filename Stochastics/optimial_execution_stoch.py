def optimal_execution_stochastic(X0, T, sigma, gamma, eta, n_steps=100, market_impact_vol=0.01):
    """
    Optimal execution with stochastic market impact.
    """
    dt = T / n_steps
    time_grid = np.linspace(0, T, n_steps)
    X = np.zeros(n_steps)
    X[0] = X0

    for t in range(1, n_steps):
        market_impact = np.random.normal(0, market_impact_vol)
        dX = -gamma * (X[t - 1] / T) * dt - eta * sigma * np.sqrt(dt) + market_impact
        X[t] = max(X[t - 1] + dX, 0)

    return time_grid, X
