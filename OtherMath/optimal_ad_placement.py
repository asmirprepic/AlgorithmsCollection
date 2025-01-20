def optimal_ad_placement(true_ctr, max_attempts, cost_per_view):
    """
    Decides when to stop showing an ad to a user using Thompson Sampling.
    
    Args:
        true_ctr: True click-through rate of the user.
        max_attempts: Maximum number of ad impressions.
        cost_per_view: Cost of showing one ad.

    Returns:
        Total cost and whether the user clicked.
    """
    alpha, beta = 1, 1  # Priors for beta
    total_cost = 0
    user_behavior = simulate_user_behavior(true_ctr, max_attempts)
    
    for t, clicked in enumerate(user_behavior):
        total_cost += cost_per_view
        
        # Sample from the posterior
        sampled_ctr = np.random.beta(alpha, beta)
        
        # Stopping descision
        expected_gain = sampled_ctr - cost_per_view
        if expected_gain < 0:
            break
        
        # Posterior update
        if clicked:
            alpha += 1
            break
        else:
            beta += 1
    
    return total_cost, clicked
