"""
 Solving Predator Prey
  x' = x(alpha - beta*y)
  y' = y*(-delta + gamma*x)

"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def systemOde(X,t,alpha,beta,gamma,delta):
  x,y = X
  dx = x*(alpha-beta*y)
  dy = y*(-delta +gamma *x)

  return np.array([dx,dy])



alpha = 1.
beta = 1.
delta = 1.
gamma = 1.
x0 = 4.
y0 = 2.

Nt = 1000
tmax = 30.
t =np.linspace(0.,tmax,Nt)
X0 = [x0,y0]
solOde = odeint(systemOde,X0,t,args = (alpha,beta,gamma,delta))

x,y = solOde.T
plt.figure()
plt.grid()
plt.plot(t,x)
plt.plot(t,y)
plt.show()
