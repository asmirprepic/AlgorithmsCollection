def euler_maruyama_vectorized(self, T, N, M=1, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    dt = T / N
    sqrt_dt = np.sqrt(dt)
    t = np.linspace(0, T, N + 1)
    
    # Precompute the random increments
    dw = sqrt_dt * np.random.randn(N, M)
    
    # Compute the increments
    increments = (self.mu - 0.5 * self.sigma**2) * dt + self.sigma * dw
    
    # Cumulative sum to get log(X)
    log_X = np.log(self.x0) + np.cumsum(increments, axis=0)
    log_X = np.vstack([np.log(self.x0) * np.ones(M), log_X])  # Add initial condition
    
    # Exponentiate to get X
    X = np.exp(log_X)
    
    return t, X
