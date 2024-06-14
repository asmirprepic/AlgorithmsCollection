class InterestRateSwapMonteCarlo:
    def __init__(self, notional, fixed_rate, initial_floating_rate, years_to_maturity, payments_per_year, discount_curve, sigma, num_simulations):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.initial_floating_rate = initial_floating_rate
        self.years_to_maturity = years_to_maturity
        self.payments_per_year = payments_per_year
        self.discount_curve = discount_curve
        self.sigma = sigma
        self.num_simulations = num_simulations

    def simulate_floating_rates(self):
        dt = 1 / self.payments_per_year
        num_steps = self.years_to_maturity * self.payments_per_year
        floating_rates = np.zeros((self.num_simulations, num_steps + 1))
        floating_rates[:, 0] = self.initial_floating_rate

        for t in range(1, num_steps + 1):
            z = np.random.normal(size=self.num_simulations)
            floating_rates[:, t] = floating_rates[:, t - 1] * np.exp(-0.5 * self.sigma**2 * dt + self.sigma * np.sqrt(dt) * z)

        return floating_rates

    def floating_leg_pv(self):
        floating_rates = self.simulate_floating_rates()
        dt = 1 / self.payments_per_year
        num_steps = self.years_to_maturity * self.payments_per_year
        floating_leg_pvs = np.zeros(self.num_simulations)

        for t in range(1, num_steps + 1):
            cash_flow_time = t * dt
            discount_factor = self.discount_curve(cash_flow_time)
            cash_flows = floating_rates[:, t] * self.notional * dt
            floating_leg_pvs += cash_flows * discount_factor

        floating_leg_pvs += self.notional * self.discount_curve(self.years_to_maturity)
        return np.mean(floating_leg_pvs)

    def fixed_leg_pv(self):
        fixed_leg_pv = 0
        for t in range(1, self.years_to_maturity * self.payments_per_year + 1):
            cash_flow_time = t / self.payments_per_year
            discount_factor = self.discount_curve(cash_flow_time)
            cash_flow = self.fixed_rate * self.notional / self.payments_per_year
            fixed_leg_pv += cash_flow * discount_factor
        fixed_leg_pv += self.notional * self.discount_curve(self.years_to_maturity)
        return fixed_leg_pv

    def swap_value(self):
        return self.floating_leg_pv() - self.fixed_leg_pv()

# Example usage
sigma = 0.02  # Volatility of the floating rate
num_simulations = 10000

swap_mc = InterestRateSwapMonteCarlo(notional, fixed_rate, floating_rate, years_to_maturity, payments_per_year, discount_curve, sigma, num_simulations)
swap_value_mc = swap_mc.swap_value()

print(f"Swap Value (Monte Carlo): {swap_value_mc:.2f}")
