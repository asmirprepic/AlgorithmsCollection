import numpy as np

def calculate_swap_value(fixed_rate, floating_rate, notional_principal, tenure, payment_freq, discount_factors):
    """
    Calculate the NPV of the interest rate swap.

    fixed_rate: Fixed interest rate (annual)
    floating_rate: Floating interest rate (annual)
    notional_principal: Notional principal amount
    tenure: Tenure of the swap in years
    payment_freq: Number of payments per year
    discount_factors: List of discount factors for each payment period
    """
    num_payments = int(tenure * payment_freq)
    payment_times = np.arange(1, num_payments + 1) / payment_freq

    # Calculate cash flows for fixed leg
    fixed_leg_cash_flows = notional_principal * fixed_rate / payment_freq
    fixed_leg_pv = np.sum(fixed_leg_cash_flows * discount_factors[:num_payments])

    # Calculate cash flows for floating leg
    floating_leg_cash_flows = notional_principal * floating_rate / payment_freq
    floating_leg_pv = np.sum(floating_leg_cash_flows * discount_factors[:num_payments])

    # Swap value from the perspective of the fixed rate payer
    swap_value = floating_leg_pv - fixed_leg_pv
    return swap_value

# Example usage
fixed_rate = 0.02  # 2%
floating_rate = 0.025  # 2.5%
notional_principal = 1000000  # $1,000,000
tenure = 5  # 5 years
payment_freq = 2  # Semi-annually

# Assuming flat discount factors for simplicity
# In practice, use the appropriate yield curve to derive these
discount_factors = np.exp(-np.arange(1, tenure * payment_freq + 1) / payment_freq * fixed_rate)

swap_value = calculate_swap_value(fixed_rate, floating_rate, notional_principal, tenure, payment_freq, discount_factors)
print("Swap Value:", swap_value)
