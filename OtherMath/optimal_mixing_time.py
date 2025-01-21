def optimal_mixing_time(H_max, k, t0, threshold, max_time):
    """
    Finds the optimal mixing time for powder homogeneity.
    
    Args:
        H_max: Maximum achievable homogeneity.
        k: Rate of mixing.
        t0: Time of fastest mixing.
        threshold: Marginal improvement threshold.
        max_time: Maximum mixing time.
    
    Returns:
        Optimal time to stop mixing.
    """
    times = np.linspace(0, max_time, 100)
    homogeneity = simulate_homogeneity(times, H_max, k, t0)
    marginal_improvements = np.diff(homogeneity)

    for i, improvement in enumerate(marginal_improvements):
        if improvement < threshold:
            return times[i]
    return max_time
