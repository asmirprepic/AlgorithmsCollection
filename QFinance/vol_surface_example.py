from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot  as plt

# Synthetic implied volatility data
strikes = np.linspace(80, 120, 5)
maturities = np.linspace(0.1, 1.0, 5)
implied_vols = np.array([
    [0.25, 0.24, 0.23, 0.22, 0.21],
    [0.26, 0.25, 0.24, 0.23, 0.22],
    [0.27, 0.26, 0.25, 0.24, 0.23],
    [0.28, 0.27, 0.26, 0.25, 0.24],
    [0.29, 0.28, 0.27, 0.26, 0.25]
])

# Plot the implied volatility surface
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(strikes, maturities)
ax.plot_surface(X, Y, implied_vols, cmap='viridis')

ax.set_xlabel('Strike Price')
ax.set_ylabel('Time to Maturity')
ax.set_zlabel('Implied Volatility')
ax.set_title('Implied Volatility Surface')
plt.show()
