import numpy as np
import matplotlib.pyplot as plt

class LSMCAmerican:
  """
    A class used to represent an American Option and price it using the Least Squares Monte Carlo method.
    
    Attributes:
    -----------
    S : float
        The initial stock price.
    K : float
        The strike price of the option.
    T : float
        The time to maturity of the option (in years).
    r : float
        The risk-free interest rate (annual).
    sigma : float
        The volatility of the underlying stock.
    simulations : int
        The number of Monte Carlo simulations to run.
    steps : int
        The number of time steps in the Monte Carlo simulations.

    Methods:
    --------
    simulate_price_paths(self):
        Simulates the price paths for the underlying asset using Geometric Brownian Motion.

    price_option(self, option_type='call'):
        Prices the option using the Least Squares Monte Carlo method.

    plot_price_paths(self, paths_to_show=10):
        Plots a specified number of simulated price paths.

    plot_distribution_at_time(self, time_point):
        Plots the distribution of simulated asset prices at a specific time point.

    plot_payoffs(self):
        Plots the distribution of final asset prices and option payoffs.
  """

  def __init__(self,S,K,T,r,sigma,simulations,steps):
    """
      Constructs all the necessary attributes for the LSMCAmerican object.
    """
    self.S = S
    self.K = K
    self.T = T
    self.r = r
    self.sigma = sigma
    self.simulations = simulations
    self.steps = steps
    self.dt =T/steps
    self.discount = np.exp(-r*self.dt)
    self.price_paths = self.simulate_price_paths()

  def simulate_price_paths(self):
    """
      Simulates the price paths for the underlying asset using Geometric Brownian Motion.

      Returns:
      --------
      numpy.ndarray
          A numpy array of shape (steps + 1, simulations) containing simulated asset prices.
    """
    price_paths = np.zeros((self.steps + 1, self.simulations),dtype = np.float64)
    price_paths[0] = self.S

    Z = np.random.standard_normal((self.steps,self.simulations))
    percentage_change = (self.r-0.5*self.sigma**2)*self.dt + self.sigma*np.sqrt(self.dt)*Z
    
    price_paths[1:] = self.S*np.cumprod(np.exp(percentage_change),axis = 0)

    return price_paths
  
  def price_option(self,option_type = 'call'):
    """
      Prices the option using the Least Squares Monte Carlo method.

      Parameters:
      -----------
      option_type : str, optional
          The type of the option to price ('call' or 'put'). Default is 'call'.

      Returns:
      --------
      float
          The priced value of the option.
    """
    price_paths= self.price_paths
    
    payoffs = np.maximum(self.K - price_paths, 0) if option_type == 'put' else np.maximum(price_paths - self.K, 0)

    values = np.zeros_like(payoffs)
    values[-1] = payoffs[-1]

    for t in range(self.steps - 1,0,-1):
      reg = np.polyfit(price_paths[t],values[t+1]*self.discount,2)
      continuation_value = np.polyval(reg,price_paths[t])

      exercise = payoffs[t] > continuation_value
      values[t] = np.where(exercise,payoffs[t],values[t+1]*self.discount)
    
    option_price = np.mean(values[1])*self.discount
    return option_price

  def plot_price_paths(self,paths_to_show = 10):
    """
      Plots a specified number of simulated price paths.

      Parameters:
      -----------
      paths_to_show : int, optional
          The number of paths to display on the plot. Default is 10.
    """
    plt.figure(figsize = (10,6))
    plt.plot(self.price_paths[:,:paths_to_show])
    plt.title(f"Simulated Price Paths for the Underlying Asset (First {paths_to_show})")
    plt.xlabel("Time Steps")
    plt.ylabel("Asset Price")
    plt.show()


  def plot_distribution_at_time(self,time_point):
    """
      Plots the distribution of simulated asset prices at a specific time point.

      Parameters:
      -----------
      time_point : float
          The time point (in years) at which to plot the distribution of asset prices.
    """
    time_step = int(time_point*self.steps/self.T)
    plt.figure(figsize = (10,6))
    plt.hist(self.price_paths[time_step],bins = 50,alpha =0.5)
    plt.title(f"Distribution of Asset Prices at Time {time_point} years")
    plt.xlabel("Asset Price")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()
  
  def plot_payoffs(self):
    """
      Plots the distribution of final asset prices and option payoffs.
    """
    plt.figure(figsize =(12,7))
    final_prices = self.price_paths[-1]
    put_payoffs = np.maximum(self.K - final_prices, 0)
    call_payoffs = np.maximum(final_prices - self.K, 0)

    plt.hist(final_prices, bins=50, alpha=0.5, label="Asset Prices")
    plt.hist(put_payoffs, bins=50, alpha=0.5, label="Put Option Payoffs")
    plt.hist(call_payoffs, bins=50, alpha=0.5, label="Call Option Payoffs")

    plt.title("Distribution of Final Asset Prices and Option Payoffs")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()



    
