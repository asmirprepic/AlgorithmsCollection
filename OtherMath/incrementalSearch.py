
"""
Implements the incremental search for a equation
"""


import numpy as np
import matplotlib.pyplot as plt

def incremental_search(func,a,b,dx):
    
    fa = func(a)
    c = a + dx
    fc = func(c)
    n = 1
    while np.sign(fa) == np.sign(fc):
        
        if a>=b:
            return a-dx,n
        
        a = c
        fa=fc
        c = a + dx
        fc = func(c)
        n += 1
       
    if fa==0:
        return a,n
    elif fc==0:
        return c,n
    else:
        return(a+c)/2.,n
        
    
y = lambda x: x**3 + 2.*x**2 - 5.

root,iterations = incremental_search(y,-5.,5.,0.001)
print(root)
print(iterations)

steps =np.arange(-5.,5.,0.001)
plt.plot(steps,y(steps))
