import numpy as np

def binomial_option_pricing(S, K, T, r, sigma, N, option_type='call'):
    """
    Calculate the price of a European option using the binomial option pricing model.

    Parameters:
    S (float): Current price of the underlying asset.
    K (float): Strike price of the option.
    T (float): Time to expiration in years.
    r (float): Risk-free interest rate (annual).
    sigma (float): Volatility of the underlying asset.
    N (int): Number of steps in the binomial tree.
    option_type (str): Type of the option ('call' or 'put').

    Returns:
    float: Calculated price of the option.
    """

    # Time step per period
    dt = T / N

    # Calculate up and down factors
    u = np.exp(sigma * np.sqrt(dt))  # Up factor
    d = 1 / u                         # Down factor

    # Risk-neutral probability
    p = (np.exp(r * dt) - d) / (u - d)

    # Initialize the binomial tree for underlying asset prices
    binomial_tree = np.zeros([N + 1, N + 1])
    for i in range(N + 1):
        for j in range(i + 1):
            binomial_tree[j, i] = S * (u ** j) * (d ** (i - j))

    # Initialize the option value tree
    option = np.zeros([N + 1, N + 1])

    # Determine the option value at the final nodes
    if option_type == 'call':
        option[:, N] = np.maximum(np.zeros(N + 1), binomial_tree[:, N] - K)
    elif option_type == 'put':
        option[:, N] = np.maximum(np.zeros(N + 1), K - binomial_tree[:, N])

    # Backward induction to calculate option price
    for i in range(N - 1, -1, -1):
        for j in range(i + 1):
            option[j, i] = np.exp(-r * dt) * (p * option[j + 1, i + 1] + (1 - p) * option[j, i + 1])

    return option[0, 0]
