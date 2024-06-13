import numpy as np

class FixedIncomeSecurity:
    def __init__(self, face_value, coupon_rate, years_to_maturity, payments_per_year=2):
        self.face_value = face_value
        self.coupon_rate = coupon_rate
        self.years_to_maturity = years_to_maturity
        self.payments_per_year = payments_per_year

    def price(self, market_rate):
        """
        Calculate the price of the bond given the market rate.
        
        :param market_rate: Annual market interest rate as a decimal.
        :return: Price of the bond.
        """
        coupon_payment = self.coupon_rate * self.face_value / self.payments_per_year
        discount_rate = market_rate / self.payments_per_year
        periods = self.years_to_maturity * self.payments_per_year

        price = sum([coupon_payment / (1 + discount_rate) ** t for t in range(1, periods + 1)]) \
                + self.face_value / (1 + discount_rate) ** periods

        return price

    def yield_to_maturity(self, current_price, tol=1e-6, max_iter=1000):
        """
        Calculate the yield to maturity of the bond given its current price.
        
        :param current_price: Current price of the bond.
        :param tol: Tolerance level for convergence.
        :param max_iter: Maximum number of iterations.
        :return: Yield to maturity as a decimal.
        """
        guess_rate = self.coupon_rate
        low = 0
        high = 1

        for i in range(max_iter):
            price_guess = self.price(guess_rate)
            if abs(price_guess - current_price) < tol:
                return guess_rate
            elif price_guess > current_price:
                high = guess_rate
            else:
                low = guess_rate
            guess_rate = (high + low) / 2

        raise Exception("Yield to maturity did not converge")

    def duration(self, market_rate):
        """
        Calculate the duration of the bond given the market rate.
        
        :param market_rate: Annual market interest rate as a decimal.
        :return: Duration of the bond.
        """
        coupon_payment = self.coupon_rate * self.face_value / self.payments_per_year
        discount_rate = market_rate / self.payments_per_year
        periods = self.years_to_maturity * self.payments_per_year

        weighted_cash_flows = sum([t * coupon_payment / (1 + discount_rate) ** t for t in range(1, periods + 1)])
        weighted_cash_flows += periods * self.face_value / (1 + discount_rate) ** periods

        bond_price = self.price(market_rate)

        duration = weighted_cash_flows / bond_price / self.payments_per_year

        return duration

# Example usage
bond = FixedIncomeSecurity(face_value=1000, coupon_rate=0.05, years_to_maturity=10)
market_rate = 0.04
current_price = bond.price(market_rate)
ytm = bond.yield_to_maturity(current_price)
duration = bond.duration(market_rate)

print(f"Bond Price: {current_price:.2f}")
print(f"Yield to Maturity: {ytm:.4f}")
print(f"Duration: {duration:.4f}")
