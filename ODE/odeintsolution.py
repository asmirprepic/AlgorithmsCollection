## Solving wiht ODEint  'y(t) = -y*t+13 


import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def odefunct(y,t):
  return -y*t+13


y0 = 1

t = np.linspace(0,5)

y = odeint(odefunct,y0,t)

plt.plot(t,y)
plt.show()
