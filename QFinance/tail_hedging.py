def simulate_tail_hedging(data, put_spread_strikes, variance_swap_vol, delta_threshold, tail_prob):
    """
    Simulate the performance of a tail-risk hedging strategy.
    """
    lower_strike, upper_strike, premium_cost = put_spread_strikes
    notional = []
    put_spread_pnl = []
    variance_swap_pnl = []

    for i in range(len(data)):
        
        hedge_notional = calculate_hedging_notional(
            data["PortfolioValue"].iloc[i], tail_prob, delta_threshold
        )
        notional.append(hedge_notional)

        
        realized_vol = np.random.uniform(0.1, 0.6)  
        spot = data["Price"].iloc[i]

        
        ps_pnl = hedge_notional * put_spread_payoff(spot, lower_strike, upper_strike, premium_cost)
        put_spread_pnl.append(ps_pnl)

        
        vs_pnl = variance_swap_payoff(realized_vol, variance_swap_vol, hedge_notional)
        variance_swap_pnl.append(vs_pnl)

    data["Notional"] = notional
    data["PutSpreadPnL"] = put_spread_pnl
    data["VarianceSwapPnL"] = variance_swap_pnl
    data["TotalPnL"] = data["PutSpreadPnL"] + data["VarianceSwapPnL"]

    return data
