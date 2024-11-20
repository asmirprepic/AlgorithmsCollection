import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Fetch historical data for multiple stocks (e.g., AAPL, MSFT, and TSLA)
tickers = ['AAPL', 'MSFT', 'TSLA']
data = yf.download(tickers, start="2015-01-01", end="2024-01-01")['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Portfolio optimization
def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate portfolio performance: expected return and volatility.
    """
    port_return = np.sum(weights * mean_returns)
    port_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return port_return, port_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0.01):
    """
    Calculate negative Sharpe ratio for optimization.
    """
    port_return, port_volatility = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(port_return - risk_free_rate) / port_volatility

# Expected returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Initial guess (equal weight portfolio)
initial_guess = len(tickers) * [1. / len(tickers)]

# Constraints and bounds
constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})  # Weights must sum to 1
bounds = tuple((0, 1) for _ in range(len(tickers)))  # Weights between 0 and 1

# Optimize portfolio
opt_result = minimize(negative_sharpe_ratio, initial_guess, args=(mean_returns, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal portfolio weights
optimal_weights = opt_result.x

# Calculate optimal portfolio performance
opt_return, opt_volatility = portfolio_performance(optimal_weights, mean_returns, cov_matrix)

print(f"Optimal Portfolio Weights: {optimal_weights}")
print(f"Expected Portfolio Return: {opt_return}")
print(f"Portfolio Volatility: {opt_volatility}")
