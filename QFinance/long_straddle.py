import numpy as np
import matplotlib.pyplot as plt

def long_straddle_payoff(stock_prices, strike_price, call_premium, put_premium):
    """
    Calculate the payoff from a long straddle strategy.

    stock_prices: Array of stock prices at expiration
    strike_price: Strike price for both the call and the put option
    call_premium: Premium paid for the call option
    put_premium: Premium paid for the put option
    """
    # Calculate the payoff for the call option
    call_payoff = np.maximum(stock_prices - strike_price, 0) - call_premium
    
    # Calculate the payoff for the put option
    put_payoff = np.maximum(strike_price - stock_prices, 0) - put_premium
    
    # The total payoff is the sum of the call payoff and the put payoff
    total_payoff = call_payoff + put_payoff
    
    return total_payoff

# Example usage
strike_price = 50  # Strike price for both options
call_premium = 5  # Premium paid for the call option
put_premium = 5  # Premium paid for the put option

# Generate a range of stock prices at expiration
stock_prices = np.linspace(30, 70, 100)
payoff = long_straddle_payoff(stock_prices, strike_price, call_premium, put_premium)

# Plot the results
plt.figure(figsize=(10,6))
plt.plot(stock_prices, payoff, label="Long Straddle Payoff")
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit/Loss')
plt.title('Long Straddle Strategy Payoff')
plt.axhline(0, color='black', lw=1)
plt.axvline(strike_price, color='red', linestyle='--', label='Strike Price')
plt.legend()
plt.grid(True)
plt.show()
