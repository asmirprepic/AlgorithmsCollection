"""
Performs the bisection search for a equation
"""

import numpy as np
import matplotlib.pyplot as plt

def bisection_search(func,a,b,tol=0.1,maxiter=100):
    
    c=(a+b)*0.5
    n=1
    
    while n <= maxiter:
        c = (a+b)*0.5
        if func(c)==0 or abs(a-b)*0.5 < tol:
            return c,n
        
        n += 1 
        if func(c) < 0:
            a = c
        else:
            b=c
        
    return c,n


y = lambda x: x**3 + 2.*x**2 - 5
root, iterations = bisection_search(y,-5,5,0.00001,100)
print(root)
print(iterations)

steps = np.arange(-5,5,0.001)
plt.plot(steps,y(steps))
