
"""
Simulation of a non-homogenous poisson process with specified intensity function
"""

import numpy as np
import math
from scipy.integrate import quad

# Define the intensity function
intensity_function = lambda x: 100 * (math.sin(x * math.pi) + 1)
# Integrated intensity function
Lambda = lambda t: quad(intensity_function, 0, t)[0]

def Ft(x, t):
    return 1 - np.exp(-Lambda(t + x) + Lambda(t))

def Ftinv(u):
    a = 0
    b = Tmax
    for j in range(50):
        if Ft((a + b) / 2, t) <= u:
            binf = (a + b) / 2
            bsup = b
        if Ft((a + b) / 2, t) >= u:
            bsup = (a + b) / 2
            binf = a

        a = binf
        b = bsup

    return (a + b) / 2


t = 0
X = [t]
Tmax = 10

while X[-1] <= Tmax:
    x = Ftinv(np.random.uniform(0, 1))
    t = t + x
    X.append(t)
