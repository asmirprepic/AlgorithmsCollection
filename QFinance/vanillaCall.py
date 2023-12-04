import numpy as np
import math

class StochasticProcess:
    """
    A class representing a stochastic process for modeling asset price movements.
    """

    def __init__(self, asset_price, drift, delta_t, asset_volatility):
        """
        Initializes the StochasticProcess class.

        Parameters:
        asset_price (float): The initial price of the asset.
        drift (float): The drift term of the asset price, representing the expected return.
        delta_t (float): The time increment for each step of the simulation.
        asset_volatility (float): The volatility of the asset price.
        """
        self.current_asset_price = asset_price
        self.asset_prices = [asset_price]
        self.drift = drift
        self.delta_t = delta_t
        self.asset_volatility = asset_volatility

    def time_step(self):
        """
        Simulates the next time step in the stochastic process.
        """
        dW = np.random.normal(0, math.sqrt(self.delta_t))  # Random increment based on normal distribution
        dS = (self.drift * self.current_asset_price * self.delta_t +
              self.asset_volatility * self.current_asset_price * dW)
        self.current_asset_price += dS
        self.asset_prices.append(self.current_asset_price)


class Call:
    """
    A class representing a call option.
    """

    def __init__(self, strike):
        """
        Initializes the Call class.

        Parameters:
        strike (float): The strike price of the call option.
        """
        self.strike = strike


class EuroCallSim:
    """
    A class for simulating the price of a European call option using stochastic processes.
    """

    def __init__(self, Call, n_options, initial_asset_price, drift, delta_t, volatility, tte, rfr):
        """
        Initializes the EuroCallSim class.

        Parameters:
        Call (Call): The call option to be priced.
        n_options (int): The number of option simulations to run.
        initial_asset_price (float): The initial price of the underlying asset.
        drift (float): The drift term of the asset price.
        delta_t (float): The time increment for each step of the simulation.
        volatility (float): The volatility of the asset price.
        tte (float): Time to expiration of the option.
        rfr (float): Risk-free rate.
        """
        stochastic_processes = [StochasticProcess(initial_asset_price, drift, delta_t, volatility)
                                for _ in range(n_options)]

        for stochastic_process in stochastic_processes:
            ttei = tte
            while (ttei - stochastic_process.delta_t) > 0:
                ttei -= stochastic_process.delta_t
                stochastic_process.time_step()

        payoffs = [(max(stochastic_process.asset_prices[-1] - Call.strike, 0))
                   for stochastic_process in stochastic_processes]

        self.price = np.average(payoffs) * math.exp(-tte * rfr)

# Example usage
print(EuroCallSim(Call(130), 1000, 295.40, 0, 1/365, 1.0625, 36/365, 0.08).price)
