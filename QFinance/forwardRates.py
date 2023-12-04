class ForwardRates:
    """
    A class for calculating forward interest rates from spot rates.
    """

    def __init__(self):
        """
        Initializes the ForwardRates class.
        """
        self.forward_rates = []  # List to store calculated forward rates
        self.spot_rates = dict()  # Dictionary to store spot rates with their corresponding time periods

    def add_spot_rate(self, T, spot_rate):
        """
        Adds a spot rate for a specific time period to the spot_rates dictionary.

        Parameters:
        T (float): The time period for the spot rate (in years).
        spot_rate (float): The spot rate for the given time period.
        """
        self.spot_rates[T] = spot_rate

    def get_forward_rates(self):
        """
        Calculates forward rates based on the spot rates added to the class.

        Returns:
        list: A list of calculated forward rates.
        """
        periods = sorted(self.spot_rates.keys())  # Sort the time periods
        for T2, T1 in zip(periods, periods[1:]):
            forward_rate = self.calculate_forward_rate(T1, T2)
            self.forward_rates.append(forward_rate)
        return self.forward_rates

    def calculate_forward_rate(self, T1, T2):
        """
        Calculates the forward rate between two time periods.

        Parameters:
        T1 (float): The earlier time period.
        T2 (float): The later time period.

        Returns:
        float: The calculated forward rate between T1 and T2.
        """
        R1 = self.spot_rates[T1]
        R2 = self.spot_rates[T2]
        forward_rate = (R2 * T2 - R1 * T1) / (T2 - T1)
        return forward_rate

# Example usage
if __name__ == "__main__":
    fr = ForwardRates()
    fr.add_spot_rate(0.25, 10.127)
    fr.add_spot_rate(0.50, 10.469)
    fr.add_spot_rate(1.00, 10.536)
    fr.add_spot_rate(1.50, 10.681)
    fr.add_spot_rate(2.00, 10.808)
    print(fr.get_forward_rates())
