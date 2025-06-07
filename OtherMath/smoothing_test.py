import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


x = np.linspace(0, 4*np.pi, 500)
y = np.sin(x) + 0.2 * np.random.normal(size=len(x))


y_smooth = savgol_filter(y, window_length=31, polyorder=3)


y_derivative = savgol_filter(y, window_length=31, polyorder=3, deriv=1, delta=x[1]-x[0])


plt.figure(figsize=(12,6))
plt.subplot(2,1,1)
plt.plot(x, y, label='Noisy Signal', alpha=0.5)
plt.plot(x, y_smooth, label='Smoothed Signal', linewidth=2)
plt.legend()
plt.title('Savitzky-Golay Filtering and Derivative')

plt.subplot(2,1,2)
plt.plot(x, y_derivative, color='orange', label='Derivative (Velocity)')
plt.legend()
plt.xlabel('x')
plt.tight_layout()
plt.show()
