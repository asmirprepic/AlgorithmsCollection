import numpy as np
import matplotlib.pyplot as plt

def levy_flight(n_steps, alpha = 1.5):
  """
  Simulate a 2D Levy flight

  Parameters:
  ---------------
  n_steps: Number of steps in the flight
  alpha: Stability parameter for Levy Distribution

  Returns: 
  tuple: Two arrays representing x and y coordinates for the flight
  """

  step_lengths = np.random.pareto(alpha,n_steps)
  angles = np.random.uniform(0,2*np.pi,n_steps)

  x_steps = step_lengths*np.cos(angles)
  y_steps = step_lengths*np.sin(angles)

  x_position = np.cumsum(x_steps)
  y_position= np.cumsum(y_steps)

  return x_positions,y_positions

## Example of how to use
    
plt.subplot(1,2,1)
plt.plot(levy_x,levy_y)
plt.scatter(levy_x[-1],levy_y[-1])
