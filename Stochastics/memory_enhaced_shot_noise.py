import numpy as np

def memory_enhanced_shot_noise(T: float, lambda_0: float,alpha1:float,  alpha2: float, beta1: float, beta2:float, memory_factor: float):
  """
  Implementation fo the memory enhaced shot noise process. 

  Parameters: 
  ------------
  T: Simulation time
  lambda_0: Base intensity 
  alpha1, alpha2: Initial impacts of events
  beta1, beta2: Decay rates for short-term, long-term components
  memory_factor: Scales intensity based on time since last events

  Returns: 
  -------------
  times (np.array): list of times
  shot_noise (np.array): Intensity of the shot noise

  """
  times = []
  intensity = lambda_0
  shot_noise = []

  t= 0
  while t < T:
    intensity = lambda_0*(1+memory_factor*len(times))

    # Time to next event
    u = np.random.uniform()
    t += -np.log(u)/intensity

    if t >= T: 
      break

    times.append(t)
    current_shot_noise = sum(alpha1*np.exp(beta1*(t-ti))+ alpha2*np.exp(-beta2*(t-ti)) for ti in times)
    shot_noise.append(current_shot_noise)

  return np.array(times), np.array(shot_noise)


    
