import numpy as np
import matplotlib.pyplot as plt


def secant(func,a,b, tol=0.001,maxiter =100):
    n = 1
    
    while n <= maxiter:
        c = b-func(b)*((b-a)/(func(b)-func(a)))
        
        if abs(c-b) < tol:
            return c,n
        
        a=b
        b=c
        n +=1
    
    return None,n


y = lambda x: x**3 + 2.*x**2 - 5

root, iterations = secant(y,-5.0,5.0,0.00001,100)
print(root)
print(iterations)

steps = np.arange(-5,5,0.001)
plt.plot(steps,y(steps))
