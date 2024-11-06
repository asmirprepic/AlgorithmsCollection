import numpy as np

def self_exciting_jump_process(T: float, lambda_base: float, jump_magnitude: float, decay_rate: float): 

  """
  T: total simulation time
  lambda_base: Baseline jump intensity
  jump_magnitude: Magnitude for jumps to boost intensity
  decay_rate: At which rate the jump intensity goes back 

  Returns: 
  times: jump times
  jumps: jumps

  """

  times = []
  jumps = []
  intensity = lambda_base
  process_value = 0
  t = 0
  while t < T: 
    intensity = lambda_base + sum(jump_magnitude*np.exp(-decay_rate*(t-ti)) for ti in times)
    u = np.random.uniform()
    w = -np.log(u)/intensity
    t += w
   
     if t >= T:
      break

        # Record jump
        times.append(t)
        process_value += np.random.normal(jump_magnitude, 0.5 * jump_magnitude)  # Random jump size
        jumps.append(process_value)

    return np.array(times), np.array(jumps)
