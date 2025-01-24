"""
Implementation of a acceptance rejection algorithm 
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import gamma
import seaborn as sns
sns.set()
np.random.seed(12345)

def f1(x):
    a = 2.7
    b=6.3
    beta = gamma(a) *gamma(b) / gamma(a+b)
    p= x**(a-1)*(1-x)**(b-1)
    return 1/beta*p

mode = (2.7-1)/(2.7+6.3-2)
c = f1(mode)

def beta_gen(n):
    i = 0
    output = np.zeros(n)
    while i < n:
        U = np.random.uniform(size=1)
        V = np.random.uniform(size=1)
        
        if U< 1/c*f1(V):
            output[i] = V
            i = i+1
        
    return output

px = np.arange(0,1+0.01,0.01)
py = f1(px)

Y = beta_gen(n=1000)
fig,ax = plt.subplots()
temp = ax.hist(Y,density=True)
ax.plot(px,py)
plt.show()

