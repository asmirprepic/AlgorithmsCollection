def analytic_theta_linear(mu: np.ndarray, Sigma: np.ndarray, a: np.ndarray, b: float) -> np.ndarray:
    """
    Cramér (saddlepoint) tilt for linear event {a^T X >= b}, X~N(mu, Σ):
      choose θ* so that E_θ[a^T X] = b  =>  a^T (mu + Σ θ*) = b
      => θ* = Σ^{-1} a * ( (b - a^T mu) / (a^T Σ a) )
    This makes the rare event typical under the tilted law.
    """
    mu = np.asarray(mu, dtype=float).ravel()
    a = np.asarray(a, dtype=float).ravel()
    Sigma = np.asarray(Sigma, dtype=float)
    denom = float(a @ Sigma @ a)
    if denom <= 0:
        raise ValueError("a^T Σ a must be positive.")
    scale = (b - float(a @ mu)) / denom
    # Solve Σ θ = scale * a  ->  θ = Σ^{-1} (scale a)
    theta = np.linalg.solve(Sigma, scale * a)
    return theta
