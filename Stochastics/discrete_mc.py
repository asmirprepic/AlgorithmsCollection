"""
Thos code shows an example of a discrete markov chain

"""

import numpy as np
import matplotlib.pyplot as plt

N = 100 # maximum population size
a = .5/N # birth rate
b = 0.5/N # death rate

nsteps = 1000 # number of time steps
x = np.zeros(nsteps) # initalization of time
x[0] = 25 # starting population

for t in range(nsteps-1):
    if 0 < x[t] < N-1:
        # Check if birth
        birth = np.random.rand() <= a*x[t]
        
        # Check if death
        death = np.random.rand() <= b*x[t]
        
        #population size
        x[t+1] = x[t] +1*birth -1*death
        
    else:
        x[t+1] =x[t]
        
fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(x, lw=2)
