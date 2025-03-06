class CommodityShockSimulator:
    def __init__(self, initial_prices: dict, volatilities: dict, num_simulations: int = 1000, time_horizon: int = 1):
        """
        Initialize simulation parameters.
        - initial_prices: dict of initial prices for commodities.
        - volatilities: dict of daily volatilities for each commodity.
        - num_simulations: Number of Monte Carlo simulations.
        - time_horizon: Time horizon in days for each simulation.
        """
        self.initial_prices = initial_prices
        self.volatilities = volatilities
        self.num_simulations = num_simulations
        self.time_horizon = time_horizon
        self.shock_scenarios = {"mild_shock": 1.5, "moderate_shock": 2.0, "severe_shock": 3.0}  # Example shock multipliers

    def simulate_paths(self):
        """
        Simulate price paths for each commodity.
        """
        simulated_paths = {commodity: np.zeros((self.time_horizon, self.num_simulations)) for commodity in self.initial_prices}
        
        for commodity, price in self.initial_prices.items():
            daily_volatility = self.volatilities[commodity]
            for t in range(1, self.time_horizon):
                shocks = np.random.normal(0, daily_volatility, self.num_simulations)
                simulated_paths[commodity][t] = simulated_paths[commodity][t - 1] * np.exp(shocks)
            logging.info("Simulated paths for %s completed.", commodity)
        
        return simulated_paths

    def apply_shocks(self, paths, shock_level):
        """
        Apply extreme shock scenarios to each commodity.
        - shock_level: Multiplier indicating the severity of the price shock.
        """
        
        shocked_paths = {}
        for commodity, path in paths.items():
            shock_multiplier = self.shock_scenarios[shock_level]
            shocked_paths[commodity] = path * shock_multiplier
            logging.info("Applied %s shock to %s.", shock_level, commodity)
        
        
        
        
        return shocked_paths
