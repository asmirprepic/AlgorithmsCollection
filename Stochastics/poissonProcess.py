"""
Simulation of poisson process
"""


import numpy as np
import matplotlib.pyplot as plt

lmbda = 0.5 # Rate of the Poisson process
maxtime = 50 # Time limit
num_simulations = 100 # Number of simulations to perform

# Create a 2D array to store arrival times for each simulation
arrivaltimes = np.zeros((num_simulations, maxtime))

# Perform the simulations
for i in range(num_simulations):
    # Draw exponential random variables until maxtime is reached
    arr = np.cumsum(np.random.exponential(1/lmbda,  maxtime))
    # Store the arrival times in the array
    arrivaltimes[i, :len(arr)] = arr
    
# Plot the arrival times for each simulation
fig, ax = plt.subplots()
ax.set(xlim=(0, maxtime), ylim=(0, len(arrivaltimes[0])-1),
       xlabel='Time', ylabel='Arrivals')
for i in range(num_simulations):
    for j in range(len(arrivaltimes[i])-1):
        # Horizontal lines
        ax.plot([arrivaltimes[i,j], arrivaltimes[i,j+1]], [j, j], '-',color="black")
        # Vertical dashed lines
        ax.plot([arrivaltimes[i,j+1], arrivaltimes[i,j+1]], [j, j+1], '-',color="black")

    # Last piece of line in the graph
    ax.plot([arrivaltimes[i,-1], maxtime], [len(arrivaltimes[0])-1, len(arrivaltimes[0])-1], '-',color="black")

plt.show()

## plotting the histogram of counts until max time

counts = [len(arrivaltimes[i][arrivaltimes[i] < 50]) for i in range(num_simulations)]
plt.hist(counts, bins=range(max(counts)+2),density=True, align='left', rwidth=0.5)
plt.xlabel('Counts')
plt.ylabel('Frequency')
plt.show()
