def generate_fbm(n, H, T):
    """
    Generates a fractional Brownian motion (fBM) path using the Cholesky method.

    Parameters:
    - n: Number of time steps
    - H: Hurst exponent
    - T: Total time horizon

    Returns:
    - t: Time grid
    - B_H: Simulated fractional Brownian motion path
    """
    dt = T / n
    t = np.linspace(0, T, n)

    
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = 0.5 * (abs(i*dt) ** (2 * H) + abs(j*dt) ** (2 * H) - abs(i*dt - j*dt) ** (2 * H))

    
    L = np.linalg.cholesky(R + 1e-10 * np.eye(n))  # Adding small noise for numerical stability

    
    W = np.random.normal(size=n)

    
    B_H = np.dot(L, W)

    return t, B_H
