def iron_condor(data, lower_put_strike, higher_put_strike, lower_call_strike, higher_call_strike, premium_put, premium_call):
    """
    Simulate the Iron Condor strategy.
    
    Parameters:
    - data: DataFrame with asset prices.
    - lower_put_strike: Strike price of the bought put.
    - higher_put_strike: Strike price of the sold put.
    - lower_call_strike: Strike price of the sold call.
    - higher_call_strike: Strike price of the bought call.
    - premium_put: Net premium from the put spread.
    - premium_call: Net premium from the call spread.
    """
    pnl = []
    for price in data["Price"]:
        if price < lower_put_strike:
            # Loss on put spread
            pnl.append(-premium_put + (lower_put_strike - price))
        elif price < higher_put_strike:
            # Profit on put spread
            pnl.append(premium_put)
        elif price < lower_call_strike:
            # Profit on entire Iron Condor
            pnl.append(premium_put + premium_call)
        elif price < higher_call_strike:
            # Profit on call spread
            pnl.append(premium_call)
        else:
            # Loss on call spread
            pnl.append(-premium_call + (price - higher_call_strike))
    
    data["P&L"] = pnl
    data["CumulativeP&L"] = data["P&L"].cumsum()
    return data
