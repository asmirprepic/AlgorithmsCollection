import numpy as np

def price_forward(S, r, T):
    """
    Calculate the theoretical price of a forward contract.

    Parameters:
    S (float): Current price of the underlying asset.
    r (float): Risk-free interest rate (annualized).
    T (float): Time to maturity of the contract (in years).

    Returns:
    float: The theoretical price of the forward contract.
    """
    forward_price = S * np.exp(r * T)
    return forward_price

# Example usage
S = 100  # Current price of the underlying asset
r = 0.05  # Annual risk-free rate
T = 1  # Time to maturity in years
forward_price = price_forward(S, r, T)
print(f"Theoretical Forward Price: {forward_price}")
