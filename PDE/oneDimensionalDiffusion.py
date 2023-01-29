


## Solve the 1D duffusion equation with 

import numpy 
from matplotlib import pyplot

def oneDDiffusion():
  nx = 41 # spatial points
  L = 2 # doman length
  dx = L/(nx-1) ##spatial grid size
  nu = 0.3 
  sigma = 0.2 ## condition for stability
  dt = sigma*dx**2/nu ## time step size for stability
  nt = 20 ## number of time steps

  ## coordinates 
  x=numpy.linspace(0.0,L,num=nx)

  # initial conditions
  u0 = numpy.ones(nx)
  mask = numpy.where(numpy.logical_and(x >= 0.5,x<=1.0))
  u0[mask] = 2.0

  ### integrate 
  u = u0.copy()
  for i in range(nt):
    u[1:-1] = u[1:-1] + nu*dt/dx**2*(u[2:]-2*u[1:-1]+u[:-2])
  return u
