### example Laplace equation in 2D
import numpy as np
import matplotlib.pyplot as plt
def solving():
  N=100
  iter = 500

  v = np.zeros((N,N))
  v[:,0] =10.0
  v[N-1] = 6.0
  E = []
  Niter = range(iter)

  for iter in Niter:
    v0 = v.copy()

    for i in range(1,N-1):
      for j in range(1,N-1):
        v[i,j] = 0.25*(v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1])
  
  return v



