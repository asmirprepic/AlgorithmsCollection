def st_garch(params, returns, z):
    """
    Computes conditional variances under an ST-GARCH(1,1) model.
    
    Model:
        sigma_t^2 = omega + (alpha + delta * G(z[t-1])) * epsilon[t-1]^2 +
                    (beta + theta * G(z[t-1])) * sigma[t-1]^2
    
    Args:
        params (np.array): Parameter vector [omega, alpha, delta, beta, theta, gamma, c].
        returns (np.array): Array of returns (innovations).
        z (np.array): Transition variable series (can be the same as returns or another variable).
    
    Returns:
        np.array: Conditional variances sigma^2.
    """
    omega, alpha, delta, beta, theta, gamma, c = params
    T = len(returns)
    sigma2 = np.zeros(T)
    
    # Initialize the first conditional variance (e.g., with the sample variance)
    sigma2[0] = np.var(returns)
    
    for t in range(1, T):
        # Compute the smooth transition factor using the lagged transition variable.
        G = logistic_function(z[t-1], gamma, c)
        sigma2[t] = (omega + 
                     (alpha + delta * G) * (returns[t-1]**2) +
                     (beta + theta * G) * sigma2[t-1])
    return sigma2
