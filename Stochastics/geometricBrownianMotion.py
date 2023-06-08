import numpy as np
import matplotlib.pyplot as plt


# Define parameters

mu = 0.1
n=100
T=1
number_simulations = 100

S0= 100
sigma = 0.3


## Simulating
dt = T/n

St = np.exp(
    (mu-sigma**2)/2*dt
    + sigma*np.random.normal(0,np.sqrt(dt),size = (number_simulations,n)).T
    
    
    )

St = np.vstack([np.ones(number_simulations),St])

St = S0*St.cumprod(axis=0)

## time intervals
time=np.linspace(0,T,n+1)
tt = np.full(shape=(number_simulations,n+1),fill_value=time).T

plt.plot(tt,St)
plt.xlabel("Years $(t)$")
plt.ylabel("$S_t$")
plt.show()
