import numpy as np
import math


class StocahsticProcess:
  def time_step(self):
    
    dW = np.random.normal(0,math.sqrt(self.delta_t))
    dS = self.drift*self.current_asset_price*self.delta_t + self.asset_volatility*self.current_asset_price*dW
    self.asset_prices.append(self.current_asset_price + dS)
    self.current_asset_price = self.current_asset_price + dS
    


  def __init__(self,asset_price,drift,delta_t,asset_volatility):
    self.current_asset_price = asset_price
    self.asset_prices = []
    self.asset_prices.append(asset_price)
    self.drift = drift
    self.delta_t = delta_t
    self.asset_volatility = asset_volatility


class Call:
  def __init__(self,strike):
    self.strike = strike



class EuroCallSim:
  def __init__(self,Call,n_options,initial_asset_price,drift,delta_t,volatility,tte,rfr):
    stochastic_processes = []
    for i in range(0,n_options):
      stochastic_processes.append(StocahsticProcess(initial_asset_price,drift,delta_t,volatility))

  
    for stochastic_process in stochastic_processes:
      ttei = tte
      while((ttei-stochastic_process.delta_t)>0):
        ttei = ttei-stochastic_process.delta_t
        stochastic_process.time_step()
        
      
      
    payoffs = []
    for stochastic_process in stochastic_processes:
      payoff = stochastic_process.asset_prices[len(stochastic_process.asset_prices)-1]-Call.strike
      z = payoff if payoff > 0 else 0
      payoffs.append(z)

    self.price = np.average(payoffs)*math.exp(-tte*rfr)


print(EuroCallSim(Call(130),1000,295.40,0,1/365,1.0625,36/365,0.08).price)

    




    
