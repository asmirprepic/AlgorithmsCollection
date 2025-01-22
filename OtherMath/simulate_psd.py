def simulate_psd(n_steps, initial_psd, target_psd, volatility):
    """
    Simulates the evolution of the particle size distribution (PSD) over time.
    
    Args:
        n_steps: Number of simulation steps.
        initial_psd: Initial particle size distribution (as a vector).
        target_psd: Target particle size distribution (as a vector).
        volatility: Stochastic volatility of the PSD process.
    
    Returns:
        Simulated PSDs over time (2D array of shape [n_steps, len(initial_psd)]).
    """
    psd_dim = len(initial_psd)
    psds = np.zeros((n_steps, psd_dim))
    psds[0] = initial_psd
    
    for t in range(1, n_steps):
        # Simulate PSD evolution with some noise
        noise = np.random.normal(0, volatility, psd_dim)
        psds[t] = psds[t-1] + 0.1 * (target_psd - psds[t-1]) + noise
        psds[t] = np.maximum(psds[t], 0)  # Ensure PSD values are non-negative
    
    return psds
