from scipy.optimize import minimize

def optimize_portfolio(contracts, historical_data):
    """
    Optimize portfolio allocation to maximize risk-adjusted returns.
    """
    def objective(weights):
        contracts["Allocation"] = weights
        contracts["AllocatedPremium"] = contracts["Allocation"] * contracts["ExpectedPremium"]
        allocated_premium = contracts["AllocatedPremium"].sum()

        
        contracts["WeightedRetention"] = contracts["Retention"] * contracts["Allocation"]
        contracts["WeightedCoverage"] = contracts["CoverageLimit"] * contracts["Allocation"]
        losses = simulate_portfolio_losses(contracts, historical_data)
        expected_loss = np.mean(losses)

        
        return -allocated_premium / expected_loss  

    
    initial_weights = [1 / len(contracts)] * len(contracts)

    
    bounds = [(0, 1)] * len(contracts)
    constraints = [{"type": "eq", "fun": lambda w: sum(w) - 1}]  

    
    result = minimize(objective, initial_weights, bounds=bounds, constraints=constraints)
    optimized_weights = result.x

    contracts["OptimizedAllocation"] = optimized_weights
    return contracts

optimized_contracts = optimize_portfolio(contracts, historical_data)
print(optimized_contracts[["ContractID", "OptimizedAllocation"]])
