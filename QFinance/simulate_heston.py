
def simulate_heston_jump(S0, T, r, kappa, theta, sigma, rho, v0, lambda_jump, mu_jump, sigma_jump, dt, num_paths):
    """
    Simulate asset paths using the Heston model with jump diffusion.
    """
    num_steps = int(T / dt)
    asset_paths = np.zeros((num_steps + 1, num_paths))
    vol_paths = np.zeros((num_steps + 1, num_paths))
    asset_paths[0, :] = S0
    vol_paths[0, :] = v0

    for t in range(1, num_steps + 1):
        # Correlated Brownian motions
        z1 = np.random.normal(0, 1, num_paths)
        z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, num_paths)

        # Volatility dynamics (Heston model)
        vol_paths[t, :] = np.maximum(vol_paths[t - 1, :] + kappa * (theta - vol_paths[t - 1, :]) * dt +
                                     sigma * np.sqrt(vol_paths[t - 1, :] * dt) * z2, 0)

        # Jump diffusion dynamics
        jumps = (np.random.poisson(lambda_jump * dt, num_paths) > 0) * np.random.normal(mu_jump, sigma_jump, num_paths)

        # Asset dynamics
        asset_paths[t, :] = asset_paths[t - 1, :] * np.exp(
            (r - 0.5 * vol_paths[t - 1, :]) * dt +
            np.sqrt(vol_paths[t - 1, :] * dt) * z1 + jumps
        )

    return asset_paths, vol_paths
