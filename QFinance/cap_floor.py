import numpy as np
from numpy.random import normal as norm

class CapFloor:
    def __init__(self, notional, cap_rate, floor_rate, time_to_maturity, payments_per_year, discount_curve, volatility):
        self.notional = notional
        self.cap_rate = cap_rate
        self.floor_rate = floor_rate
        self.time_to_maturity = time_to_maturity
        self.payments_per_year = payments_per_year
        self.discount_curve = discount_curve
        self.volatility = volatility

    def black_formula(self, forward_rate, strike_rate, time_to_maturity, volatility, option_type='cap'):
        d1 = (np.log(forward_rate / strike_rate) + 0.5 * volatility ** 2 * time_to_maturity) / (volatility * np.sqrt(time_to_maturity))
        d2 = d1 - volatility * np.sqrt(time_to_maturity)
        if option_type == 'cap':
            price = forward_rate * norm.cdf(d1) - strike_rate * norm.cdf(d2)
        elif option_type == 'floor':
            price = strike_rate * norm.cdf(-d2) - forward_rate * norm.cdf(-d1)
        return price

    def cap_floor_price(self, option_type='cap'):
        dt = 1 / self.payments_per_year
        cap_floor_price = 0
        for t in range(1, int(self.time_to_maturity * self.payments_per_year) + 1):
            forward_rate = self.cap_rate if option_type == 'cap' else self.floor_rate
            discount_factor = self.discount_curve(t * dt)
            price = self.black_formula(forward_rate, self.cap_rate if option_type == 'cap' else self.floor_rate, t * dt, self.volatility, option_type)
            cap_floor_price += price * discount_factor * self.notional * dt
        return cap_floor_price

# Example usage
notional = 1000000
cap_rate = 0.05
floor_rate = 0.03
time_to_maturity = 5
payments_per_year = 2
volatility = 0.2

# Example discount curve (flat curve with 5% discount rate)
def discount_curve(t):
    return np.exp(-0.05 * t)

cap = CapFloor(notional, cap_rate, floor_rate, time_to_maturity, payments_per_year, discount_curve, volatility)
cap_price = cap.cap_floor_price(option_type='cap')
floor_price = cap.cap_floor_price(option_type='floor')

print(f"Cap Price: {cap_price:.2f}")
print(f"Floor Price: {floor_price:.2f}")
