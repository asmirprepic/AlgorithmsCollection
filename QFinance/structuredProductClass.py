import numpy as np

class StructuredProductPrice: 
  def __init__(self,S0: float, K: float, T: float, r: float, sigma: float,face_value: float, participation_rate: float = 1.0, num_simulations: int, num_steps: int):
    """
    Initalize the structure product price
    Parameters: 
    ---------------
    S0
    K
    T
    r
    sigma
    face_value
    participation_rate
    num_simulations
    num_steps
    """

    self.S0 = S0
    self.K = K
    self.T = T
    self.r = r
    self.sigma = sigma
    self.face_avlue = face_value
    self.participation_rate = participation_rate
    self.num_simulations = num_simulations
    self.num_steps = num_steps


  def simulate_asset_paths(self): -> np.array
    """
    Simulates paths for the underlying asset using geometric brownian motion
    Returns: 
    -------------
    Simulated asset paths 

    """
    dt = self.T / self.num_steps
        Z = np.random.normal(0, 1, (self.num_simulations, self.num_steps))  # Vectorized Brownian motion
        asset_paths = np.zeros((self.num_simulations, self.num_steps + 1))
        asset_paths[:, 0] = self.S0
        
        for t in range(1, self.num_steps + 1):
            asset_paths[:, t] = asset_paths[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * Z[:, t-1])
  def price_bond(self):
    """
    Price of the zero coupon bond
    """
    bond_price = self.face_value*np.exp(-self.r*self.T)
    return bond_price
    
  def price_call_option(self):
    """
    Pricing the call option using Monte Carlo simulation
    Returns: 
    ------------
    - Present value of the call option
    """
    asset_paths = self.simulate_asset_paths()
    final_prices = asset_paths[:,-1]
    pay_offs = np.maximum(final_prices - self.K, 0)
    call_option_price = np.exp(-self.r*self.T)*np.mean(payoffs)

    return call_option_price

  def price_structured_product(self): 
    bond_price = self.price_bond()
    call_option_price = self.price_call_option()
    structured_product_price = bond_price + self.participation_rate*call_option_price
    return structured_product_price
    


    
