def variance_swap_price(option_data, F, T):
    """
    Compute the fair value of a variance swap using log-strike interpolation.
    
    Parameters:
    - option_data: List of tuples [(K, price, type)] where type is "call" or "put".
    - F: Forward price of the underlying.
    - T: Time to maturity in years.
    
    Returns:
    - Fair variance.
    """
    strikes = np.array([data[0] for data in option_data])
    prices = np.array([data[1] for data in option_data])
    types = np.array([data[2] for data in option_data])

    
    puts = [(K, price) for K, price, t in zip(strikes, prices, types) if t == "put" and K < F]
    calls = [(K, price) for K, price, t in zip(strikes, prices, types) if t == "call" and K >= F]

    
    put_integral = sum((price / K**2) * (K2 - K1) for (K1, price), (K2, _) in zip(puts[:-1], puts[1:]))
    call_integral = sum((price / K**2) * (K2 - K1) for (K1, price), (K2, _) in zip(calls[:-1], calls[1:]))

    
    fair_variance = (2 / T) * (put_integral + call_integral)
    return fair_variance
