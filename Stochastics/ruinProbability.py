import numpy as np
import matplotlib.pyplot as plt

lmbda= 0.5 #Poisson process rate
claim_rate = 2 #Rate of claim size distribution
u0 = 3 #Initial capital level
T = 10 #Time horizon
theta = 0.1 #Loading factor
c = (1+theta)*lmbda/claim_rate #lambda/claimrate
runs = 100000 #number of runs
results = [] # store results
times = []
for r in range(runs):
  time = 0 #keep track of the time
  level = u0 #keep track of capital level
  interarrivaltime = np.random.exponential(1/lmbda)

  while time + interarrivaltime < T:
    time += interarrivaltime
    claim = np.random.exponential(1/claim_rate)

    #update capital
    level += c*interarrivaltime-claim

    if level < 0:
      break

    interarrivaltime = np.random.exponential(1/lmbda)

  times.append(time)
  results.append(level<0)
m = np.mean(results)
v = np.var(results)
confidence_interval = m - 1.96 * np.sqrt(v / runs), m + 1.96 * np.sqrt(v / runs)
print(confidence_interval)
