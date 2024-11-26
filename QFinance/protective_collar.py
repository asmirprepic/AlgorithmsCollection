def protective_collar_strategy(data, put_strike_pct, call_strike_pct, expiration_days, option_cost_model):
    """
    Simulate a protective collar strategy.
    
    Parameters:
    - data: DataFrame with index prices.
    - put_strike_pct: Strike price of the protective put as % of spot.
    - call_strike_pct: Strike price of the covered call as % of spot.
    - expiration_days: Option expiration in days.
    - option_cost_model: Function to estimate option costs.
    """
    portfolio_value = []
    for i in range(len(data) - expiration_days):
        spot_price = data["Price"].iloc[i]
        put_strike = spot_price * put_strike_pct
        call_strike = spot_price * call_strike_pct
        
        # Estimate option costs
        put_cost = option_cost_model(spot_price, put_strike, expiration_days, "put")
        call_cost = option_cost_model(spot_price, call_strike, expiration_days, "call")
        net_option_cost = put_cost - call_cost

        # Calculate payoff
        future_price = data["Price"].iloc[i + expiration_days]
        portfolio_payoff = spot_price + net_option_cost
        if future_price < put_strike:
            portfolio_payoff += put_strike - future_price  # Put protection
        elif future_price > call_strike:
            portfolio_payoff += call_strike - future_price  # Call obligation
        else:
            portfolio_payoff += future_price - spot_price  # No option exercise
        
        portfolio_value.append(portfolio_payoff)
    
    # Fill the remaining days with the last portfolio value
    portfolio_value += [portfolio_value[-1]] * expiration_days
    data["PortfolioValue"] = portfolio_value
    return data
