import numpy as np
import matplotlib.pyplot as plt

def forwardEuler(F,x,y,xstop,h):

  X = []
  Y = []

  X.append(x)
  Y.append(y)

  while x < xstop:
    h = min(h,xstop-x)
    y = y + h*F(x,y)
    x = x+h
    X.append(x)
    Y.append(y)

  return np.array(X),np.array(Y)



### Solving the ivp y'' = -0.1y'-x, y(0) = 0, y'(0) = 1

def F(x,y):
  F = np.zeros(2)
  F[0] = y[1]
  F[1] = -0.1*y[1]-x
  return F

x = 0.0
xstop = 2.0
y = np.array([0.0,1.0])
h=0.01

X,Y = forwardEuler(F,x,y,xstop,h)
plt.plot(X,Y[:,0])
plt.grid(True)
plt.show()

