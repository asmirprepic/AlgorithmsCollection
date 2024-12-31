def price_catastrophe_bond(event_times, losses, face_value, coupon_rate, risk_premium, discount_rate, T):
    """
    Price a catastrophe bond based on expected payouts.

    Parameters:
    - event_times: Times of disaster events
    - losses: Simulated losses for each event
    - face_value: Bond's face value ($)
    - coupon_rate: Annual coupon rate (%)
    - risk_premium: Additional risk premium (%)
    - discount_rate: Risk-free discount rate (%)
    - T: Time to maturity (years)

    Returns:
    - bond_price: Fair price of the catastrophe bond
    """
    
    payouts = np.clip(losses - face_value, 0, None)
    expected_payout = np.mean(payouts)
    
    
    annual_coupon = face_value * coupon_rate
    discount_factors = [(1 + discount_rate + risk_premium)**-t for t in range(1, T+1)]
    present_value_coupons = sum(annual_coupon * df for df in discount_factors)
    present_value_principal = face_value * discount_factors[-1]
    
    
    bond_price = present_value_coupons + present_value_principal - expected_payout
    return bond_price
