import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.interpolate import interp1d
from scipy.integrate import simps  # Simpson's rule for numerical integration

def estimate_risk_neutral_density(options, S, T, r):
    """
    Estimates the risk-neutral probability density of options.

    Parameters:
    - options: DataFrame with columns ['strike', 'price', 'iv']
    - S: Current price of the underlying asset
    - T: Time to expiration in years
    - r: Risk-free interest rate

    Returns:
    - A DataFrame with estimated risk-neutral densities.
    """
    # Calculate the second derivative numerically
    options['density'] = np.nan  # Initialize density column
    
    for i in range(1, len(options)-1):
        K = options.loc[i, 'strike']
        delta_K = options.loc[i+1, 'strike'] - K
        C_minus = options.loc[i-1, 'price']
        C = options.loc[i, 'price']
        C_plus = options.loc[i+1, 'price']
        
        second_derivative = (C_minus - 2*C + C_plus) / (delta_K**2)
        options.loc[i, 'density'] = second_derivative

    # Interpolate densities for a smoother curve if necessary
    density_interpolation = interp1d(options['strike'], options['density'], fill_value="extrapolate")
    
    return options

# Example usage
# options = pd.DataFrame({
#     'strike': [...],  # Strike prices
#     'price': [...],   # Option prices
#     'iv': [...]       # Implied volatilities
# })
# result = estimate_risk_neutral_density(options, S=100, T=0.5, r=0.01)


# Generate a synthetic dataset
np.random.seed(42)  # For reproducibility
strike_prices = np.linspace(80, 120, 20)
true_iv = 0.2  # True implied volatility for the synthetic data
S = 100  # Current stock price
T = 1.0  # Time to expiration in years
r = 0.05  # Risk-free rate

# Calculate option prices using Black-Scholes formula for calls
# Note: This is a simplified approach and skips some steps for brevity
option_prices = S * norm.cdf((np.log(S / strike_prices) + (r + true_iv**2 / 2) * T) / (true_iv * np.sqrt(T))) - \
                strike_prices * np.exp(-r * T) * norm.cdf((np.log(S / strike_prices) + (r - true_iv**2 / 2) * T) / (true_iv * np.sqrt(T)))

# Create DataFrame
options_df = pd.DataFrame({
    'strike': strike_prices,
    'price': option_prices,
    'iv': np.full_like(strike_prices, true_iv)  # Assume IV is known and constant for simplicity
})

# Apply the function
estimated_density = estimate_risk_neutral_density(options_df, S, T, r)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(estimated_density['strike'][1:-1], estimated_density['density'][1:-1], label='Estimated Risk-Neutral Density')
plt.fill_between(estimated_density['strike'][1:-1], 0, estimated_density['density'][1:-1], alpha=0.2)
plt.title('Estimated Risk-Neutral Density')
plt.xlabel('Strike Price')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()
