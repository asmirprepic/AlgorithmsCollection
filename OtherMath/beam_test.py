import numpy as np
import matplotlib.pyplot as plt
import control as ctrl

# System parameters
K = 1.0  # Gain
tau = 10.0  # Time constant
theta = 2.0  # Time delay

# Create the transfer function for the FOPTD model
num = [K]
den = [tau, 1]
sys = ctrl.TransferFunction(num, den) * ctrl.tf([1], [1, 0], delay=theta)

# PID parameters
Kp = 2.0
Ki = 0.1
Kd = 0.5

# Create the PID controller
pid = Kp + Ki / ctrl.tf([1, 0]) + Kd * ctrl.tf([1, 0])

# Closed-loop system
system_closed_loop = ctrl.feedback(pid * sys)

# Time for simulation
t = np.linspace(0, 100, 500)

# Step response
t, y = ctrl.step_response(system_closed_loop, T=t)

# Plotting the response
plt.figure()
plt.plot(t, y)
plt.axhline(1, color='r', linestyle='--')  # Set point line
plt.title('PID Controlled System Response')
plt.xlabel('Time')
plt.ylabel('Output')
plt.grid(True)
plt.show()
