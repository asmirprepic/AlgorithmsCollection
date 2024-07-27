class MonteCarloBarrierOptionAnalysis(Derivative):
    def __init__(self, S, K, T, r, sigma, barrier, option_type='call', barrier_type='up-and-out', simulations=10000):
        super().__init__(S, K, T, r, sigma)
        self.barrier = barrier
        self.option_type = option_type.lower()
        self.barrier_type = barrier_type.lower()
        self.simulations = simulations

    def simulate_price_paths(self):
        dt = self.T / self.simulations
        price_paths = np.zeros((self.simulations, int(self.T / dt) + 1))
        price_paths[:, 0] = self.S
        
        for t in range(1, price_paths.shape[1]):
            z = np.random.standard_normal(self.simulations)
            price_paths[:, t] = price_paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        
        return price_paths

    def barrier_option_price(self):
        price_paths = self.simulate_price_paths()
        if self.option_type == 'call' and self.barrier_type == 'up-and-out':
            payoff = np.maximum(price_paths[:, -1] - self.K, 0)
            payoff[price_paths.max(axis=1) >= self.barrier] = 0
        elif self.option_type == 'put' and self.barrier_type == 'down-and-out':
            payoff = np.maximum(self.K - price_paths[:, -1], 0)
            payoff[price_paths.min(axis=1) <= self.barrier] = 0
        else:
            raise ValueError("Invalid option type or barrier type.")
        
        price = np.exp(-self.r * self.T) * np.mean(payoff)
        return price