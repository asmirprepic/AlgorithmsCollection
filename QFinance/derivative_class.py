class Derivative:
    def __init__(self, S, K, T, r, sigma):
        self.S = S  # Current stock price
        self.K = K  # Strike price
        self.T = T  # Time to maturity (in years)
        self.r = r  # Risk-free interest rate (annual)
        self.sigma = sigma  # Volatility of the underlying stock (annual)
    
    def get_params(self):
        return {
            'Stock Price': self.S,
            'Strike Price': self.K,
            'Time to Maturity': self.T,
            'Risk-free Rate': self.r,
            'Volatility': self.sigma
        }
