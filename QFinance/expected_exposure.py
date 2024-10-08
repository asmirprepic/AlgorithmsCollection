
def asset_prices(S0,r,sigma,T,M,I):

  paths = np.zeros((M + 1, I))
  paths[0] = S0
  for t in range(1, M + 1):
      z = np.random.standard_normal(I)  # Generate standard normal random variables
      paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    
    return paths

def calculate_expected_exposure(paths,strike_price,is_call = True)
  M,I = paths.shape
  exposures = np.zeros((M,I))

  if is_call: 
      exposure = np.maximum(paths-strike_price,0)

  expected_exposure = np.mean(exposures,axis = 1)
  return expected_exposure
