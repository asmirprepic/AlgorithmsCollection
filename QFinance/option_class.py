from scipy.optimize import brentq
import numpy as np
import scipy.stats as stats
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

class OptionAnalysis:
    def __init__(self, S, K, T, r, sigma, option_type='call'):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def option_price(self):
        d1 = self.d1()
        d2 = self.d2()
        if self.option_type == 'call':
            price = (self.S * stats.norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        elif self.option_type == 'put':
            price = (self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2) - self.S * stats.norm.cdf(-d1))
        else:
            raise ValueError("Invalid option type. Use 'call' or 'put'.")
        return price

    def delta(self):
        d1 = self.d1()
        if self.option_type == 'call':
            return stats.norm.cdf(d1)
        elif self.option_type == 'put':
            return stats.norm.cdf(d1) - 1

    def gamma(self):
        d1 = self.d1()
        return stats.norm.pdf(d1) / (self.S * self.sigma * np.sqrt(self.T))

    def vega(self):
        d1 = self.d1()
        return self.S * stats.norm.pdf(d1) * np.sqrt(self.T)

    def theta(self):
        d1 = self.d1()
        d2 = self.d2()
        if self.option_type == 'call':
            theta = (-self.S * stats.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) - 
                     self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(d2))
        elif self.option_type == 'put':
            theta = (-self.S * stats.norm.pdf(d1) * self.sigma / (2 * np.sqrt(self.T)) + 
                     self.r * self.K * np.exp(-self.r * self.T) * stats.norm.cdf(-d2))
        return theta / 365

    def rho(self):
        d2 = self.d2()
        if self.option_type == 'call':
            return self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(d2)
        elif self.option_type == 'put':
            return -self.K * self.T * np.exp(-self.r * self.T) * stats.norm.cdf(-d2)

    def binomial_tree(self, steps=100):
        dt = self.T / steps
        u = np.exp(self.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp(self.r * dt) - d) / (u - d)
        
        asset_prices = np.zeros((steps + 1, steps + 1))
        for i in range(steps + 1):
            for j in range(i + 1):
                asset_prices[j, i] = self.S * (u ** (i - j)) * (d ** j)
        
        option_values = np.zeros((steps + 1, steps + 1))
        if self.option_type == 'call':
            option_values[:, steps] = np.maximum(0, asset_prices[:, steps] - self.K)
        elif self.option_type == 'put':
            option_values[:, steps] = np.maximum(0, self.K - asset_prices[:, steps])
        
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                option_values[j, i] = (q * option_values[j, i + 1] + (1 - q) * option_values[j + 1, i + 1]) * np.exp(-self.r * dt)
        
        return option_values[0, 0]

    def monte_carlo(self, simulations=10000):
        dt = self.T / simulations
        S = np.zeros(simulations)
        S[0] = self.S
        for t in range(1, simulations):
            z = np.random.standard_normal()
            S[t] = S[t - 1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z)
        
        if self.option_type == 'call':
            payoff = np.maximum(S[-1] - self.K, 0)
        elif self.option_type == 'put':
            payoff = np.maximum(self.K - S[-1], 0)
        
        return np.exp(-self.r * self.T) * np.mean(payoff)

    @staticmethod
    def fetch_option_data(ticker, start_date, end_date):
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date)
        return hist

    @staticmethod
    def plot_stock_data(data):
        plt.figure(figsize=(10, 6))
        plt.plot(data['Close'], label='Close Price')
        plt.title('Stock Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.show()

    @staticmethod
    def calculate_historical_volatility(data):
        log_returns = np.log(data['Close'] / data['Close'].shift(1))
        volatility = log_returns.std() * np.sqrt(252)  # Annualize volatility
        return volatility

    def implied_volatility(self, market_price, tol=1e-5, max_iterations=100):
        """
        Calculate the implied volatility using the market price of the option.
        :param market_price: The market price of the option
        :param tol: Tolerance for convergence
        :param max_iterations: Maximum number of iterations
        :return: Implied volatility
        """
        def objective_function(vol):
            self.sigma = vol
            return self.option_price() - market_price
        
        # Use Brent's method to find the root (volatility)
        implied_vol = brentq(objective_function, 1e-5, 5.0, xtol=tol, maxiter=max_iterations)
        return implied_vol
    
    @staticmethod
    def fetch_option_chain(ticker, expiration_date):
        """
        Fetch the option chain for a given ticker and expiration date.
        :param ticker: Stock ticker symbol
        :param expiration_date: Expiration date of the options
        :return: DataFrame with option chain
        """
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration_date)
        calls = opt_chain.calls
        puts = opt_chain.puts
        return calls, puts
    
    @staticmethod
    def analyze_option_prices(ticker, expiration_date, r):
        """
        Analyze market option prices and compare them with theoretical prices.
        :param ticker: Stock ticker symbol
        :param expiration_date: Expiration date of the options
        :param r: Risk-free interest rate
        """
        # Fetch historical stock data for calculating volatility
        end_date = expiration_date
        start_date = (pd.to_datetime(end_date) - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
        data = OptionAnalysis.fetch_option_data(ticker, start_date, end_date)
        
        # Calculate historical volatility
        hist_volatility = OptionAnalysis.calculate_historical_volatility(data)

        # Fetch option chain
        calls, puts = OptionAnalysis.fetch_option_chain(ticker, expiration_date)
        
        # Analyze calls
        print("\nCall Options Analysis:")
        for index, row in calls.iterrows():
            S = data['Close'].iloc[-1]
            K = row['strike']
            T = (pd.to_datetime(expiration_date) - pd.to_datetime(data.index[-1])).days / 365
            market_price = row['lastPrice']
            
            option = OptionAnalysis(S, K, T, r, hist_volatility, option_type='call')
            theoretical_price = option.option_price()
            implied_vol = option.implied_volatility(market_price)
            
            print(f"Strike: {K}, Market Price: {market_price}, Theoretical Price: {theoretical_price:.2f}, Implied Volatility: {implied_vol:.2%}")

        # Analyze puts
        print("\nPut Options Analysis:")
        for index, row in puts.iterrows():
            S = data['Close'].iloc[-1]
            K = row['strike']
            T = (pd.to_datetime(expiration_date) - pd.to_datetime(data.index[-1])).days / 365
            market_price = row['lastPrice']
            
            option = OptionAnalysis(S, K, T, r, hist_volatility, option_type='put')
            theoretical_price = option.option_price()
            implied_vol = option.implied_volatility(market_price)
            
            print(f"Strike: {K}, Market Price: {market_price}, Theoretical Price: {theoretical_price:.2f}, Implied Volatility: {implied_vol:.2%}")

    @staticmethod
    def analyze_implied_volatility_skew(calls, puts):
        """
        Analyze the implied volatility skew from call and put options.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        """
        calls['impliedVolatility'] = calls['impliedVolatility'].astype(float)
        puts['impliedVolatility'] = puts['impliedVolatility'].astype(float)

        plt.figure(figsize=(10, 6))
        plt.plot(calls['strike'], calls['impliedVolatility'], label='Call IV')
        plt.plot(puts['strike'], puts['impliedVolatility'], label='Put IV')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title('Implied Volatility Skew')
        plt.legend()
        plt.show()

    @staticmethod
    def calculate_put_call_ratio(calls, puts):
        """
        Calculate the put/call ratio.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        :return: Put/Call ratio
        """
        total_call_volume = calls['volume'].sum()
        total_put_volume = puts['volume'].sum()
        put_call_ratio = total_put_volume / total_call_volume
        return put_call_ratio

    @staticmethod
    def open_interest_and_volume_analysis(calls, puts):
        """
        Analyze open interest and volume for calls and puts.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        """
        plt.figure(figsize=(10, 6))

        plt.subplot(2, 1, 1)
        plt.bar(calls['strike'], calls['openInterest'], color='blue', label='Calls OI')
        plt.bar(puts['strike'], puts['openInterest'], color='red', alpha=0.5, label='Puts OI')
        plt.xlabel('Strike Price')
        plt.ylabel('Open Interest')
        plt.title('Open Interest Analysis')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.bar(calls['strike'], calls['volume'], color='blue', label='Calls Volume')
        plt.bar(puts['strike'], puts['volume'], color='red', alpha=0.5, label='Puts Volume')
        plt.xlabel('Strike Price')
        plt.ylabel('Volume')
        plt.title('Volume Analysis')
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def implied_volatility_statistics(calls, puts):
        """
        Calculate skewness and kurtosis of implied volatility.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        :return: Dictionary with skewness and kurtosis for calls and puts
        """
        calls_iv = calls['impliedVolatility'].astype(float)
        puts_iv = puts['impliedVolatility'].astype(float)

        stats = {
            'call_iv_skewness': calls_iv.skew(),
            'call_iv_kurtosis': calls_iv.kurtosis(),
            'put_iv_skewness': puts_iv.skew(),
            'put_iv_kurtosis': puts_iv.kurtosis()
        }
        return stats
    
    @staticmethod
    def analyze_volatility_smile(calls, puts):
        """
        Analyze the volatility smile from call and put options.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(calls['strike'], calls['impliedVolatility'], label='Call IV', color='blue')
        plt.scatter(puts['strike'], puts['impliedVolatility'], label='Put IV', color='red')
        plt.xlabel('Strike Price')
        plt.ylabel('Implied Volatility')
        plt.title('Volatility Smile')
        plt.legend()
        plt.show()

    @staticmethod
    def probability_distribution(calls, puts):
        """
        Calculate the risk-neutral probability distribution of future prices.
        :param calls: DataFrame with call options
        :param puts: DataFrame with put options
        :return: DataFrame with risk-neutral probabilities
        """
        strikes = sorted(set(calls['strike']).union(set(puts['strike'])))
        probabilities = []
        
        for strike in strikes:
            call_price = calls[calls['strike'] == strike]['lastPrice'].values[0] if strike in calls['strike'].values else 0
            put_price = puts[puts['strike'] == strike]['lastPrice'].values[0] if strike in puts['strike'].values else 0
            probabilities.append((call_price + put_price) / 2)
        
        df = pd.DataFrame({'Strike': strikes, 'Probability': probabilities})
        df['Probability'] /= df['Probability'].sum()  # Normalize to sum to 1
        return df