import math
import matplotlib.pyplot as plt

class BootStrapYC:
    """
    A class to construct a zero-coupon yield curve using the bootstrapping method.
    """

    def __init__(self):
        """
        Initializes the BootStrapYC class.
        """
        self.zero_rates = dict()  # Dictionary to store zero rates
        self.instruments = dict()  # Dictionary to store bond instruments

    def add_instrument(self, par, T, coup, price, compounding_freq=2):
        """
        Adds a bond instrument to the instruments dictionary.

        Parameters:
        par (float): The par value of the bond.
        T (float): The time to maturity of the bond.
        coup (float): The coupon rate of the bond.
        price (float): The market price of the bond.
        compounding_freq (int): The number of compounding periods per year (default is 2 for semi-annual).
        """
        self.instruments[T] = (par, coup, price, compounding_freq)

    def get_maturities(self):
        """
        Returns the sorted maturities from the instruments dictionary.
        """
        return sorted(self.instruments.keys())

    def get_zero_rates(self):
        """
        Calculates the zero rates for each instrument in the dictionary.
        """
        self.bootstrap_zero_coupons()
        self.get_bond_spot_rates()
        return [self.zero_rates[T] for T in self.get_maturities()]

    def bootstrap_zero_coupons(self):
        """
        Bootstraps the zero-coupon bonds to find the zero rates.
        """
        for (T, instrument) in self.instruments.items():
            (par, coup, price, freq) = instrument
            if coup == 0:
                spot_rate = self.zero_coupon_spot_rate(par, price, T)
                self.zero_rates[T] = spot_rate

    def get_bond_spot_rates(self):
        """
        Calculates the spot rates for coupon-bearing bonds.
        """
        for T in self.get_maturities():
            instrument = self.instruments[T]
            (par, coup, price, freq) = instrument
            if coup != 0:
                spot_rate = self.calculate_bond_spot_rate(T, instrument)
                self.zero_rates[T] = spot_rate

    def zero_coupon_spot_rate(self, par, price, T):
        """
        Calculates the spot rate for a zero-coupon bond.

        Parameters:
        par (float): The par value of the bond.
        price (float): The market price of the bond.
        T (float): The time to maturity of the bond.

        Returns:
        float: The spot rate of the zero-coupon bond.
        """
        spot_rate = math.log(par / price) / T
        return spot_rate

    def calculate_bond_spot_rate(self, T, instrument):
        """
        Calculates the spot rate for a coupon-bearing bond.

        Parameters:
        T (float): The time to maturity of the bond.
        instrument (tuple): The bond instrument tuple containing par, coup, price, and freq.

        Returns:
        float: The spot rate of the coupon-bearing bond.
        """
        try:
            (par, coup, price, freq) = instrument
            periods = T * freq
            value = price
            per_coupon = coup / freq
            for i in range(int(periods) - 1):
                t = (i + 1) / float(freq)
                spot_rate = self.zero_rates[t]
                discounted_coupon = per_coupon * math.exp(-spot_rate * t)
                value -= discounted_coupon
            last_period = int(periods) / float(freq)
            spot_rate = -math.log(value / (par + per_coupon)) / last_period
            return spot_rate
        except Exception as e:
            print("Spot rate not found for T=", T, "Error: ", str(e))

# Example usage
if __name__ == "__main__":
    yield_curve = BootStrapYC()
    yield_curve.add_instrument(100, 0.25, 0., 97.5)
    yield_curve.add_instrument(100, 0.5, 0., 94.9)
    yield_curve.add_instrument(100, 1.0, 0., 90.)
    yield_curve.add_instrument(100, 1.5, 8, 96., 2)
    yield_curve.add_instrument(100, 2., 12, 101.6, 2)

    y = yield_curve.get_zero_rates()
    x = yield_curve.get_maturities()

    fig = plt.figure(figsize=(12, 8))
    plt.plot(x, y)
    plt.title("Zero Curve")
    plt.ylabel("Zero Rate (%)")
    plt.xlabel("Maturity in Years")
    plt.show()
