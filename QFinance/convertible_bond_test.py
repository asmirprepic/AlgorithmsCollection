def price_convertible_bond(S0, K, r, q, T, sigma, conversion_ratio, call_price=None, put_price=None, n_steps=100):
    """
    Price a convertible bond using a binomial tree.
    
    Parameters:
    - S0: Initial stock price
    - K: Face value of the bond
    - r: Risk-free rate
    - q: Dividend yield
    - T: Time to maturity (years)
    - sigma: Stock volatility
    - conversion_ratio: Number of shares per bond
    - call_price: Call provision (optional)
    - put_price: Put provision (optional)
    - n_steps: Number of steps in the binomial tree
    
    Returns:
    - Convertible bond price.
    """
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))      
    d = 1 / u                            
    p = (np.exp((r - q) * dt) - d) / (u - d)  # Risk-neutral probability
    discount = np.exp(-r * dt)
    
    
    stock_tree = np.zeros((n_steps + 1, n_steps + 1))
    for i in range(n_steps + 1):
        for j in range(i + 1):
            stock_tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
    
    
    bond_value = np.zeros((n_steps + 1, n_steps + 1))
    for j in range(n_steps + 1):
        conversion_value = conversion_ratio * stock_tree[j, n_steps]
        bond_value[j, n_steps] = max(K, conversion_value)
    
    
    for i in range(n_steps - 1, -1, -1):
        for j in range(i + 1):
            hold_value = discount * (p * bond_value[j, i + 1] + (1 - p) * bond_value[j + 1, i + 1])
            conversion_value = conversion_ratio * stock_tree[j, i]
            bond_value[j, i] = max(hold_value, conversion_value)  
            
            
            if call_price:
                bond_value[j, i] = min(bond_value[j, i], call_price)
            if put_price:
                bond_value[j, i] = max(bond_value[j, i], put_price)
    
    return bond_value[0, 0]
