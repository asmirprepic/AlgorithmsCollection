import numpy as np
import math
import matplotlib.pyplot as plt

def vasicek(r0,K,theta,sigma,T=1.,N=10):
    dt =T/float(N)
    rates = [r0]
    for i in range(N):
        dr = K*(theta-rates[-1])*dt + sigma*np.random.normal()
        rates.append(rates[-1]+dr)
    
    return range(N+1),rates

for K in [0.002,0.02,0.2]:
    x,y = vasicek(0.005,K,0.15,0.05,T=10,N=200)
    plt.plot(x,y)
