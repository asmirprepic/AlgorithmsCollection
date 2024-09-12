import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as si

# Define option parameters
S0 = 100  # Initial stock price
K = 100   # Strike price
T = 1.0   # Time to maturity (in years)
r = 0.05  # Risk-free interest rate
sigma = 0.2  # Volatility of the underlying asset
dt = 1/252  # Time step in years (assuming daily steps)

# Black-Scholes formula to calculate delta
def bs_delta(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = si.norm.cdf(d1)  # Delta for a call option
    return delta

# Simulate a stock price path using geometric Brownian motion (GBM)
def simulate_stock_path(S0, T, r, sigma, dt):
    n_steps = int(T / dt)
    time = np.linspace(0, T, n_steps)
    stock_path = np.zeros(n_steps)
    stock_path[0] = S0
    
    for t in range(1, n_steps):
        z = np.random.normal()
        stock_path[t] = stock_path[t-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z)
    
    return stock_path, time

# Simulate delta hedging strategy
def delta_hedging(S0, K, T, r, sigma, dt):
    # Simulate stock price path
    stock_path, time = simulate_stock_path(S0, T, r, sigma, dt)
    
    # Initialize variables
    portfolio_value = 0
    num_shares = 0
    cash_position = 0
    option_value = []
    stock_positions = []
    
    # Loop through each time step and hedge using delta
    for t in range(len(time)):
        remaining_time = T - time[t]
        
        # Calculate the option delta at this time step
        delta = bs_delta(stock_path[t], K, remaining_time, r, sigma)
        
        # Adjust the number of shares based on delta
        hedge_shares = delta - num_shares
        
        # Update the portfolio: Buy/sell shares to adjust delta
        portfolio_value += hedge_shares * stock_path[t]
        num_shares = delta
        
        # Keep track of the option value and stock positions
        option_value.append(np.maximum(stock_path[t] - K, 0))  # Option intrinsic value (for call option)
        stock_positions.append(hedge_shares)
    
    return stock_path, option_value, stock_positions, time

# Run the delta hedging simulation
stock_path, option_value, stock_positions, time = delta_hedging(S0, K, T, r, sigma, dt)

# Plot the stock price, option value, and hedging positions over time
plt.figure(figsize=(12, 8))

# Plot stock price
plt.subplot(3, 1, 1)
plt.plot(time, stock_path, label="Stock Price")
plt.title("Stock Price Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Stock Price")
plt.legend()

# Plot option value
plt.subplot(3, 1, 2)
plt.plot(time, option_value, label="Option Value", color="green")
plt.title("Option Value Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Option Value")
plt.legend()

# Plot hedging positions
plt.subplot(3, 1, 3)
plt.plot(time, stock_positions, label="Hedging Position", color="orange")
plt.title("Hedging Positions Over Time")
plt.xlabel("Time (Years)")
plt.ylabel("Number of Shares")
plt.legend()

plt.tight_layout()
plt.show()
