import numpy as np
import matplotlib.pyplot as plt


def newton_rhapson(func,df,x,tol=0.001,maxiter=100):
    n=1
    
    while n<=maxiter:
        x1  = x - func(x)/df(x)
        
        if abs(x1-x) < tol :
            return x1,n
        
        x = x1
        n += 1
        
    return None,n


y = lambda x: x**3 + 2.*x**2 - 5
dy = lambda x: 3*x**2 + 4*x
root, iterations = newton_rhapson(y,dy,5.0,0.0001,100)
print(root)
print(iterations)

steps = np.arange(-5,5,0.001)
plt.plot(steps,y(steps))
