def price(self) -> float:
        """Compute the cat bond price."""
        survival_prob = np.exp(-self.lambda_event * self.maturity)
        discount_factors = np.exp(-self.risk_free_rate * np.arange(1, self.maturity + 1))

        # Expected present value of coupons
        pv_coupons = np.sum(self.coupon_rate * self.principal * discount_factors) * survival_prob

        # Expected present value of principal
        pv_principal = (
            self.principal * survival_prob * np.exp(-self.risk_free_rate * self.maturity)
            + self.principal * (1 - survival_prob) * (1 - self.loss_fraction) * np.exp(-self.risk_free_rate * self.maturity)
        )

        total_price = pv_coupons + pv_principal
        return total_price
