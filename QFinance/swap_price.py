def price_swap(fixed_rate, floating_rate, notional, swap_term, payment_frequency):
    """
    Calculate the value of a plain vanilla interest rate swap.

    Parameters:
    fixed_rate (float): The fixed interest rate in the swap.
    floating_rate (float): The floating interest rate in the swap.
    notional (float): The notional principal amount.
    swap_term (int): The total term of the swap in years.
    payment_frequency (int): Number of payments per year.

    Returns:
    float: The net present value of the swap for the fixed rate payer.
    """
    fixed_payment = fixed_rate * notional / payment_frequency
    floating_payment = floating_rate * notional / payment_frequency
    npv = 0

    for i in range(1, swap_term * payment_frequency + 1):
        npv += (floating_payment - fixed_payment) / (1 + floating_rate / payment_frequency)**i

    return npv

# Example usage
fixed_rate = 0.05  # 5%
floating_rate = 0.04  # 4%
notional = 1000000  # $1,000,000
swap_term = 5  # 5 years
payment_frequency = 2  # semi-annual payments

swap_value = price_swap(fixed_rate, floating_rate, notional, swap_term, payment_frequency)
print(f"Swap Value: {swap_value}")
