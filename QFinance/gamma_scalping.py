def gamma_scalping(data, option_price=5, delta_hedge_ratio=0.5, notional=100):
    """
    Simulate a Gamma Scalping strategy.
    - Long gamma: Buy ATM options.
    - Hedge delta dynamically based on price changes.
    
    Parameters:
    - option_price: Cost of the option.
    - delta_hedge_ratio: Initial delta hedge ratio.
    - notional: Notional exposure per option.

    Returns:
    - DataFrame with P&L.
    """
    pnl = []
    hedging_costs = []
    option_position = notional / option_price  # Number of options bought
    delta = delta_hedge_ratio * option_position  # Initial delta hedge
    cash = 0

    for i in range(1, len(data)):
        price_change = data["Price"].iloc[i] - data["Price"].iloc[i - 1]

        
        gamma_pnl = option_position * (price_change**2) / 2

        
        hedge_cost = -delta * price_change
        cash += hedge_cost

        # Rebalance delta
        delta = delta_hedge_ratio * option_position * (data["Price"].iloc[i] / data["Price"].iloc[0])

        # Net P&L
        net_pnl = gamma_pnl + cash
        pnl.append(net_pnl)
        hedging_costs.append(hedge_cost)

    data["GammaP&L"] = pnl + [0]
    data["HedgingCosts"] = hedging_costs + [0]
    data["CumulativeP&L"] = data["GammaP&L"].cumsum()
    return data
