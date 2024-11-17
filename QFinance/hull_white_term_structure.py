def simulate_hull_white(a, sigma, r0, T, dt, num_paths):
    """
    Simulate short rate paths using the Hull-White model.
    
    dr(t) = a * (theta(t) - r(t)) * dt + sigma * dW(t)
    """
    num_steps = int(T / dt)
    rates = np.zeros((num_steps + 1, num_paths))
    rates[0, :] = r0

    for t in range(1, num_steps + 1):
        dW = np.random.normal(0, np.sqrt(dt), num_paths)
        rates[t, :] = rates[t - 1, :] + a * (0 - rates[t - 1, :]) * dt + sigma * dW

    return rates


def calibrate_hull_white(yield_curve, maturities, r0):
    """
    Calibrate the Hull-White parameters to fit the initial yield curve.
    """
    dt = maturities[1] - maturities[0]

    def objective(params):
        a, sigma = params
        simulated_rates = simulate_hull_white(a, sigma, r0, max(maturities), dt, 1000)
        simulated_yield_curve = np.mean(simulated_rates, axis=1)
        return np.sum((simulated_yield_curve[:len(yield_curve)] - yield_curve) ** 2)

    result = minimize(objective, x0=[0.1, 0.01], bounds=[(0.01, 1), (0.001, 0.1)])
    return result.x
