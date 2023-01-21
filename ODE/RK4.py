## Fourth order Runge Kutta method

import numpy as np
import matplotlib.pyplot as plt
import math

def RK4(F,x,y,xstop,h):
  
  def stepRK4(F,x,y,h):
    K0 = h*F(x,y)
    K1 = h*F(x+h/2.0,y+K0/2.0)
    K2 = h*F(x+h/2.0,y+K1/2.0)
    K3 = h*F(x+h,y+K2)
    
    return (K0+2.0*K1+2.0*K2 + K3)/6.0

  X = []
  Y = []

  X.append(x)
  Y.append(y)

  while x < xstop:
    h = min(h,xstop-x)
    y = y + stepRK4(F,x,y,h)
    x = x + h 
    X.append(x)
    Y.append(y)

  return np.array(X),np.array(Y)


## Solving ivp  y' = sin(x)


def F(x,y):
  return math.sin(y)


x = 0.0
xstop = 10
y = 1
h = 0.01

X,Y = RK4(F,x,y,xstop,h)
plt.plot(X,Y)
plt.grid(True)
plt.show()
  
  
  
