## solving IVP of the type u' = f(u,t), u(0)
import numpy as np
import matplotlib.pyplot as plt

def forwardEuler(f,U0,T,N):

    t = np.zeros(N+1)
    u = np.zeros(N+1)

    u[0] = U0
    t[0] = 0
    dt = T/N

    for n in range (N):
      t[n+1] = t[n]+dt
      u[n+1] = u[n] + dt*f(u[n],t[n])

    return u,t

def f(u,n):
  alpha = 0.2
  R = float(1)
  U0 = 0.1
  return alpha*u*(1-u/R)

U0 = 0.1
T = 40
N= 4001
u,t = forwardEuler(f,U0,T,N)

plt.plot(t,u)
plt.show()

  
