import numpy as np
import pandas as pd
import yfinance as yf

def download_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

def calculate_returns(data):
    returns = data.pct_change().dropna()
    return returns

def simulate_portfolio_returns(returns, weights):
    # Calculate daily portfolio returns
    portfolio_returns = returns.dot(weights)
    return portfolio_returns

def calculate_var(portfolio_returns, confidence_level):
    # Calculate the historical VaR
    var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
    return var

def main():
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']  # Example portfolio
    weights = np.array([0.25, 0.25, 0.25, 0.25]) # Equal weighting
    start_date = '2020-01-01'
    end_date = '2023-01-01'
    confidence_level = 0.95
    
    # Step 1: Download historical price data
    data = download_data(tickers, start_date, end_date)
    
    # Step 2: Calculate daily returns
    returns = calculate_returns(data)
    
    # Step 3: Simulate portfolio returns
    portfolio_returns = simulate_portfolio_returns(returns, weights)
    
    # Step 4: Calculate VaR
    var = calculate_var(portfolio_returns, confidence_level)
    
    print(f"1-day Value at Risk (VaR) at {confidence_level*100}% confidence level: {-var:.2f}")

if __name__ == "__main__":
    main()
