def simulate_portfolio_losses(contracts, historical_data, num_simulations=10000):
    """
    Simulate portfolio losses using Monte Carlo.
    """
    portfolio_losses = []

    for _ in range(num_simulations):
        simulated_years = historical_data.sample(frac=1, replace=True)  # Bootstrap sampling
        total_loss = 0

        for _, contract in contracts.iterrows():
            # Filter events exceeding retention but within coverage limit
            covered_events = simulated_years[
                (simulated_years["Severity"] > contract["Retention"]) &
                (simulated_years["Severity"] <= contract["Retention"] + contract["CoverageLimit"])
            ]
            covered_losses = covered_events["Severity"] - contract["Retention"]
            total_loss += covered_losses.sum()

        portfolio_losses.append(total_loss)

    return portfolio_losses
