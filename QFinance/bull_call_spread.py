import numpy as np
import matplotlib.pyplot as plt

def call_option_payoff(S, K, price):
    return np.maximum(0, S - K) - price

def bull_call_spread_payoff(S, K1, K2, price1, price2):
    payoff_long_call = call_option_payoff(S, K1, price1)
    payoff_short_call = -call_option_payoff(S, K2, -price2)
    return payoff_long_call + payoff_short_call

# Option parameters
S = np.linspace(0, 200, 500)  # Underlying asset price range at expiration
K1 = 100  # Strike price of the long call
K2 = 120  # Strike price of the short call
price1 = 10  # Price of the long call
price2 = 5   # Price of the short call

# Calculate payoffs
payoff_long_call = call_option_payoff(S, K1, price1)
payoff_short_call = -call_option_payoff(S, K2, -price2)
total_payoff = bull_call_spread_payoff(S, K1, K2, price1, price2)

# Plotting the payoff diagrams
plt.figure(figsize=(10, 6))

# Long call option
plt.plot(S, payoff_long_call, label='Long Call', linestyle='--')

# Short call option
plt.plot(S, payoff_short_call, label='Short Call', linestyle='--')

# Total payoff
plt.plot(S, total_payoff, label='Bull Call Spread', color='black')

# Annotations and labels
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(K1, color='red', linewidth=0.5, linestyle='--', label=f'Strike Price K1 = {K1}')
plt.axvline(K2, color='blue', linewidth=0.5, linestyle='--', label=f'Strike Price K2 = {K2}')
plt.title('Bull Call Spread Payoff Diagram')
plt.xlabel('Underlying Asset Price at Expiration')
plt.ylabel('Profit / Loss')
plt.legend()
plt.grid(True)
plt.show()
