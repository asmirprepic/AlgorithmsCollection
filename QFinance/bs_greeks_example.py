import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt

def black_scholes_greeks(S, K, T, r, sigma, option_type='call'):
    """
    Compute the Black-Scholes price and Greeks for a European option.

    Parameters:
    - S: Current stock price
    - K: Option strike price
    - T: Time to expiration in years
    - r: Risk-free interest rate
    - sigma: Volatility
    - option_type: 'call' or 'put'

    Returns:
    - price, delta, gamma, vega, theta, rho
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    theta = - (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r * T) * norm.cdf(d2)
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) if option_type == 'call' else -K * T * np.exp(-r * T) * norm.cdf(-d2)
    
    return price, delta, gamma, vega, theta, rho



# Create a sample options portfolio
portfolio = pd.DataFrame({
    'type': ['call', 'call', 'put', 'put'],
    'strike': [95, 100, 105, 110],
    'price': [10, 8, 6, 4],  # These prices are placeholders
    'T': [0.5, 0.5, 0.5, 0.5],  # Time to expiration in years
    'sigma': [0.2, 0.2, 0.2, 0.2],  # Implied volatilities
    'quantity': [10, 20, -15, -5]  # Number of contracts
})

S = 100  # Current stock price
r = 0.05  # Risk-free interest rate

# Calculate Greeks for each option in the portfolio
for index, row in portfolio.iterrows():
    price, delta, gamma, vega, theta, rho = black_scholes_greeks(S, row['strike'], row['T'], r, row['sigma'], row['type'])
    portfolio.at[index, 'delta'] = delta
    portfolio.at[index, 'gamma'] = gamma
    portfolio.at[index, 'vega'] = vega
    portfolio.at[index, 'theta'] = theta
    portfolio.at[index, 'rho'] = rho

# Aggregate Greeks for the entire portfolio
portfolio['total_delta'] = portfolio['delta'] * portfolio['quantity']
portfolio['total_gamma'] = portfolio['gamma'] * portfolio['quantity']
portfolio['total_vega'] = portfolio['vega'] * portfolio['quantity']
portfolio['total_theta'] = portfolio['theta'] * portfolio['quantity']
portfolio['total_rho'] = portfolio['rho'] * portfolio['quantity']

total_greeks = portfolio[['total_delta', 'total_gamma', 'total_vega', 'total_theta', 'total_rho']].sum()
print("Total Portfolio Greeks:")
print(total_greeks)


# Calculate Greeks for a range of underlying asset prices
S_range = np.linspace(80, 120, 100)
greeks_over_S = {'S': S_range, 'delta': [], 'gamma': [], 'vega': [], 'theta': [], 'rho': []}

for S_ in S_range:
    total_delta, total_gamma, total_vega, total_theta, total_rho = 0, 0, 0, 0, 0
    for index, row in portfolio.iterrows():
        _, delta, gamma, vega, theta, rho = black_scholes_greeks(S_, row['strike'], row['T'], r, row['sigma'], row['type'])
        total_delta += delta * row['quantity']
        total_gamma += gamma * row['quantity']
        total_vega += vega * row['quantity']
        total_theta += theta * row['quantity']
        total_rho += rho * row['quantity']
    greeks_over_S['delta'].append(total_delta)
    greeks_over_S['gamma'].append(total_gamma)
    greeks_over_S['vega'].append(total_vega)
    greeks_over_S['theta'].append(total_theta)
    greeks_over_S['rho'].append(total_rho)

# Convert to DataFrame
greeks_df = pd.DataFrame(greeks_over_S)

# Plot the Greeks
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(greeks_df['S'], greeks_df['delta'], label='Delta')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Delta')
plt.title('Portfolio Delta')
plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(greeks_df['S'], greeks_df['gamma'], label='Gamma')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Gamma')
plt.title('Portfolio Gamma')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(greeks_df['S'], greeks_df['vega'], label='Vega')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Vega')
plt.title('Portfolio Vega')
plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(greeks_df['S'], greeks_df['theta'], label='Theta')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Theta')
plt.title('Portfolio Theta')
plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(greeks_df['S'], greeks_df['rho'], label='Rho')
plt.xlabel('Underlying Asset Price')
plt.ylabel('Rho')
plt.title('Portfolio Rho')
plt.grid(True)

plt.tight_layout()
plt.show()


# Calculate the current portfolio delta
current_portfolio_delta = total_greeks['total_delta']

# To make the portfolio delta-neutral, calculate the number of shares needed
# Negative sign indicates a short position to offset the positive delta
shares_needed = -current_portfolio_delta

# Print the number of shares needed to make the portfolio delta-neutral
print(f"Shares needed to make the portfolio delta-neutral: {shares_needed:.2f}")

# Recalculate the portfolio Greeks after making it delta-neutral
# Assume we short the necessary shares
delta_neutral_greeks = total_greeks.copy()
delta_neutral_greeks['total_delta'] += shares_needed

print("Delta-Neutral Portfolio Greeks:")
print(delta_neutral_greeks)
