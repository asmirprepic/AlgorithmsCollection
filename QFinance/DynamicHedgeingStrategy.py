import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Dict, List

class DynamicHedgeingStrategy:
  def __init__(
    self,
    stock_data: pd.DataFrame, 
    option_positions: pd.DataFrame,
    short_interest_data: pd.DataFrame

  ) -> None: 
  """
  Initalize the dynamic hedgeing strategy. 

  Parameters: 
  ------------
  stock_data: Dataframe with stock prices. Columns: ['date','tickers','price']
  option_positions: Dataframe with option position. Columns: ['date','ticker','delta']
  short_interest_delta: Dataframe with short interest. Columns ['date','ticker','short_interest']
  """

  self.stock_data = stock_data
  self.option_position = option_positions
  self.short_interest_data = short_interest_data
  self.last_hedge_times = Dict[str,pd.Timestamp] = {}
  self.hedge_log = List[Dict[str,any]] = []

  def get_short_interest(self, ticker: str, date: pd.Timestamp) -> float: 
    """
    Retreive short interest for a given ticker and date. 

    Parameters: 
    -------------
    ticker: Stock ticker symbol
    date: Current date

    Returns: 
    -------------
    float: Short interest percentage
    
    """
    data = self.short_interest_data
    si = data[(data['ticker'] == ticker) & (data['date'] <= date)].sort_values(by = 'date', ascending = False)
    if not si.empty: 
      return si.iloc[0]['short_interest']
    else: 
      return 0.0

  def calculate_hedge_interval(self,short_interest: float) -> timedelta: 
    """
    Calculate hedge interval based on short interest. 

    Parameters: 
    ------------
    short_interest: Short interest percentage

    Returns: 
    -------------
    timedelta: hedge interval

    """

    base_interval= timedelta(minutes = 60)
    adjustment_factor = max(short_interest,0.01)
    adjusted_interval = base_interval/ adjustment_factor
    return adjusted_interval

  def should_hedge(self,ticker: str, date: pd.Timestamp) -> bool: 
    """
    Determine if hedging is needed

    Parameters: 
    ---------------
    ticker: ticker of the stock
    date: Current date

    Returns: 
    -------------
    bool: True if hedgeing is needed, False otherwise
    
    """
    short_interest = self.get_short_interest(ticker,date)
    hedge_interval = self.calculate_hedge_interval(short_interest)
    last_hedge_time = self.last_hedge_times.get(ticker,pd.Timestamp.min)

    return (date-last_hedge_time) >= hedge_interval

  def execute_hedge(self,ticker: str, date: pd.Timestamp) -> None: 
    """
    Execute hedgeing action

    Parameters: 
    ------------
    ticker: Stock ticker symbol
    date: Current date

    """
    position = self.option_positions[(self.option_positions['ticker'] == ticker) & (self.option_positions['date'] == date)]
    net_delta = position['delta'].sum()

    self.hedge_log.append({
      'date': date, 
      'ticker': ticker, 
      'action': 'hedge', 
      'net_delta': net_delta
    })
    self.last_hedge_times[ticker] = date

def run(self) -> None: 
  """
  Running dynamic hedgeing strategy
  """
  dates = self.stock_data['date'].sort_values().unique()
  tickers = self.stock_data['ticker'].unique()

  for date in dates: 
    for ticker in tickers: 
      if self.should_hedge(ticker,date):
        self.execute_hedge(ticker,date)







    
