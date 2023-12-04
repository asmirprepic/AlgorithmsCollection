

"""
Simulates interest rate changes using the Cox-Ingersoll-Ross (CIR) model.

Parameters:
r0 (float): The initial interest rate.
K (float): The speed of adjustment.
theta (float): The long-term mean interest rate.
sigma (float): The volatility of the interest rate.
T (float): The total time horizon of the simulation.
N (int): The number of steps in the simulation.

Returns:
tuple: A tuple containing two lists, the first with time points and the second with simulated interest rates.
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def CIR(r0,K,theta,sigma,T=1,N=10):
    dt = T/float(N)
    rates = [r0]
    for i in range(N):
        dr=K*(theta-rates[-1])*dt+sigma*np.sqrt(rates[-1])*np.sqrt(dt)*np.random.normal()
        print(dr)
        rates.append(rates[-1]+dr)
    return range(N+1), rates


for K in [0.002,0.02,0.2]:
    x,y = CIR(0.005,K,0.15,0.05,T=10,N=200)
    plt.plot(x,y)
