def delta_neutral_strategy(data, option_price=2, hedge_ratio=0.5, notional=10000):
    """
    Simulate a delta-neutral strategy by buying/selling options and hedging dynamically.
    """
    pnl = []
    hedging_costs = []
    delta_positions = []

    for i in range(len(data) - 1):
        row = data.iloc[i]
        next_row = data.iloc[i + 1]

        
        delta = row["Signal"] * hedge_ratio
        delta_positions.append(delta)

        
        hedge_cost = delta * (next_row["Price"] - row["Price"])

        
        option_pnl = row["Signal"] * notional * (row["ImpliedVol"] - row["RealizedVol"]) * option_price

        
        net_pnl = option_pnl - hedge_cost
        pnl.append(net_pnl)
        hedging_costs.append(hedge_cost)

    data["P&L"] = pnl + [0]
    data["HedgingCosts"] = hedging_costs + [0]
    data["Delta"] = delta_positions + [0]

    return data
