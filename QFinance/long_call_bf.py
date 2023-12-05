import numpy as np
import matplotlib.pyplot as plt

def long_call_butterfly_payoff(stock_prices, K1, K2, K3, premium):
    """
    Calculate the payoff from a long call butterfly spread.

    stock_prices: Array of stock prices at expiration
    K1, K2, K3: Strike prices for the three calls (K1 < K2 < K3)
    premium: Premium paid for each option (assumed the same for simplicity)
    """
    payoff = np.zeros_like(stock_prices)
    for i, S in enumerate(stock_prices):
        if S < K1:
            payoff[i] = -2 * premium
        elif K1 <= S < K2:
            payoff[i] = (S - K1) - 2 * premium
        elif K2 <= S <= K3:
            payoff[i] = (K2 - K1) - (S - K2) - 2 * premium
        else:  # S > K3
            payoff[i] = (K2 - K1) - (K3 - K2) - 2 * premium
    
    return payoff

# Example usage
K1 = 45  # Lower strike price
K2 = 50  # Middle strike price
K3 = 55  # Higher strike price
premium = 2  # Premium paid for each option

# Generate a range of stock prices at expiration
stock_prices = np.linspace(30, 70, 100)
payoff = long_call_butterfly_payoff(stock_prices, K1, K2, K3, premium)

# Plot the results
plt.plot(stock_prices, payoff)
plt.xlabel('Stock Price at Expiration')
plt.ylabel('Profit/Loss')
plt.title('Long Call Butterfly Spread Payoff')
plt.axhline(0, color='black', lw=1)
plt.grid(True)
plt.show()
