import numpy as np
import matplotlib.pyplot as plt

nsteps = 1000  # Number of time steps
x = np.zeros(nsteps)  # X-coordinates of the walk
y = np.zeros(nsteps)  # Y-coordinates of the walk

# Random walk process
for t in range(1, nsteps):
    direction = np.random.choice(['up', 'down', 'left', 'right'])
    if direction == 'up':
        y[t] = y[t-1] + 1
        x[t] = x[t-1]
    elif direction == 'down':
        y[t] = y[t-1] - 1
        x[t] = x[t-1]
    elif direction == 'left':
        x[t] = x[t-1] - 1
        y[t] = y[t-1]
    else:  # 'right'
        x[t] = x[t-1] + 1
        y[t] = y[t-1]

# Plotting the random walk path
fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(x, y, lw=2, label='Random Walk')
ax.scatter(0, 0, color='red', zorder=5, label='Start')  # Mark the starting point
ax.scatter(x[-1], y[-1], color='blue', zorder=5, label='End')  # Mark the ending point
ax.set_title("2D Random Walk")
ax.legend()
plt.show()
