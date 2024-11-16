def simulate_counterparty_defaults(counterparties, chol_matrix, num_simulations=10000):
    """
    Simulate correlated counterparty defaults using Gaussian copula.
    """
    num_cps = len(counterparties)
    losses = np.zeros(num_simulations)

    for sim in range(num_simulations):
        # Generate correlated normal variables
        normal_rvs = np.random.normal(size=num_cps)
        correlated_normals = chol_matrix @ normal_rvs
        
        # Convert to uniform using CDF
        uniform_rvs = norm.cdf(correlated_normals)
        
        # Simulate defaults
        defaults = uniform_rvs < counterparties["Default_Prob"].values
        
        # Calculate losses considering recovery rates
        loss = np.sum(counterparties["Exposure"].values * (1 - counterparties["Recovery_Rate"].values) * defaults)
        losses[sim] = loss

    return losses
