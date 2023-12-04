
"""
Simulation of shot noise process
"""

import numpy as np
import matplotlib.pyplot as plt

# Set parameters
intensity = 0.1
T = 1000
mean_Y = 1.0
std_Y = 0.5
r = -0.01

# Generate the arrival times of the Poisson process
inter_arrival_times = np.random.exponential(scale=1/intensity, size=int(intensity*T))
arrival_times = np.cumsum(inter_arrival_times)
arrival_times = arrival_times[arrival_times <= T]

# Generate the Y values
#Y_values = np.random.normal(loc=mean_Y, scale=std_Y, size=len(arrival_times))
Y_values = np.random.exponential(scale=1,size=len(arrival_times))

# Generate the shot noise process
t_values = np.linspace(0, T, num=1000)
shot_noise = np.zeros_like(t_values)
for T_i, Y_i in zip(arrival_times, Y_values):
    mask = (t_values >= T_i)
    shot_noise += Y_i * np.exp(r * (t_values - T_i)) * mask



# Plot the shot noise process
plt.plot(t_values, shot_noise,linestyle = "-",color = "black")

for T_i, Y_i in zip(arrival_times, Y_values):
   mask = (t_values > T_i)
   decay_line = np.where(mask, np.exp(r * (t_values - T_i))*shot_noise[np.argmax(t_values >= T_i)], None)
   plt.plot(t_values, decay_line, linestyle='-', color='black', alpha=0.5)

#plt.scatter(arrival_times, shot_noise[np.searchsorted(t_values, arrival_times)], color='red', marker='o', label='Jump')
plt.xlabel('Time')
plt.ylabel('Shot Noise Process')
plt.title('Simulation of Shot Noise Process')
plt.show()
