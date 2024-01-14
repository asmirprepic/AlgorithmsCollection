import numpy as np

def bond_duration_convexity(coupon, par_value, yield_rate, maturity, frequency=1):
    """
    Calculate the Macaulay Duration and Convexity of a bond.
    
    :param coupon: Annual coupon payment
    :param par_value: Par value of the bond
    :param yield_rate: Yield to maturity (annual)
    :param maturity: Number of years to maturity
    :param frequency: Number of coupon payments per year
    :return: Macaulay Duration, Modified Duration, and Convexity
    """
    periods = maturity * frequency
    coupon_value = coupon * par_value / frequency
    discount_rates = [(1 + yield_rate / frequency) ** -i for i in range(1, periods + 1)]

    # Cash flows
    cash_flows = np.array([coupon_value] * periods)
    cash_flows[-1] += par_value  # Add par value to the last payment

    # Macaulay Duration
    macaulay_duration = np.sum(discount_rates * cash_flows * np.arange(1, periods + 1)) / np.sum(discount_rates * cash_flows)

    # Modified Duration
    modified_duration = macaulay_duration / (1 + yield_rate / frequency)

    # Convexity
    convexity = np.sum(discount_rates * cash_flows * np.arange(1, periods + 1) ** 2) / (np.sum(discount_rates * cash_flows) * (1 + yield_rate / frequency) ** 2)

    return macaulay_duration, modified_duration, convexity

# Example usage
coupon_rate = 0.05  # 5% coupon rate
par_value = 1000  # $1000 par value
yield_rate = 0.06  # 6% yield to maturity
maturity = 10  # 10 years to maturity
frequency = 2  # Semi-annual payments

duration, modified_duration, convexity = bond_duration_convexity(coupon_rate, par_value, yield_rate, maturity, frequency)
print(f"Macaulay Duration: {duration:.2f} years")
print(f"Modified Duration: {modified_duration:.2f}")
print(f"Convexity: {convexity:.2f}")
