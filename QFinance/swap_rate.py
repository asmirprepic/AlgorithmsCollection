def calculate_swap_rate(principal, fixed_rate_guess, floating_rate, years, frequency):
    total_payments = years * frequency
    payment_amounts = [principal * floating_rate / frequency] * total_payments

    def fixed_payments(rate):
        return [principal * rate / frequency] * total_payments

    def present_value(cashflows, rate):
        return sum(cf / ((1 + rate / frequency) ** (i + 1)) for i, cf in enumerate(cashflows))

    fixed_rate = fixed_rate_guess
    tolerance = 1e-7
    max_iterations = 1000
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        fixed_cashflows = fixed_payments(fixed_rate)
        pv_fixed = present_value(fixed_cashflows, fixed_rate / frequency)
        pv_floating = present_value(payment_amounts, floating_rate / frequency)
        
        diff = pv_floating - pv_fixed
        
        if abs(diff) < tolerance:
            break

        # Adjust rate - ensure the adjustment is not too aggressive
        fixed_rate += (diff / (principal * years)) * 0.5

        # Additional check to prevent overflow
        if fixed_rate < 0 or fixed_rate > 1:
            raise ValueError("Fixed rate out of bounds. Check input values.")

    return fixed_rate

# Example usage
principal = 1000000
fixed_rate_guess = 0.02
floating_rate = 0.018
years = 5
frequency = 2

swap_rate = calculate_swap_rate(principal, fixed_rate_guess, floating_rate, years, frequency)
print(f"The calculated swap rate is: {swap_rate:.4f} or {swap_rate * 100:.2f}%")
