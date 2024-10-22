def merton_jump_diffusion(u,S0,r,T,sigma,jump_intensity,jump_mean,jump_std): 

  """
  Characteristic function for Merton Jump Diffusion
  Parameters: 
  -------------
  u (complex): Fourier variable
  S0 (float): Initial stock price
  r (float): Risk free rate
  T (float): Time to maturity
  sigma (float): Volatility
  jump_intensity (float): Intensity of jumps
  jump_std (float): Standard deviation of jump size

  Returns: 
  -------------
  complex: Value of the charactestic functin
  """

  drift = r-0.5*sigma**2-jump_intensity *(np.exp(jump_mean + 0.5*jump_std**2) - 1)
  jump_term = np.exp(u*jump_mean * 1j - 0.5* jump_std**2*u**2)
  dif
