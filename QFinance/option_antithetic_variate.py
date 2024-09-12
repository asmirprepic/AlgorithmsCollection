import numpy as np

# Define option parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1.0   # Time to maturity (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
n_simulations = 100000  # Number of Monte Carlo simulations

# Monte Carlo simulation with Antithetic Variates
def european_call_option_antithetic(S0, K, T, r, sigma, n_simulations):
    # Generate random standard normal variables
    Z = np.random.normal(0, 1, n_simulations // 2)
    Z_antithetic = -Z  # Antithetic variates

    # Combine both sets of variates
    Z_combined = np.concatenate([Z, Z_antithetic])

    # Simulate stock price at maturity
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z_combined)

    # Calculate the payoff for the call option
    payoff = np.maximum(ST - K, 0)

    # Discount the payoff back to present value
    discounted_payoff = np.exp(-r * T) * payoff

    # Return the average of the payoffs
    return np.mean(discounted_payoff)

# Calculate the option price
option_price = european_call_option_antithetic(S0, K, T, r, sigma, n_simulations)
print(f"European Call Option Price (with antithetic variates): {option_price:.4f}")
