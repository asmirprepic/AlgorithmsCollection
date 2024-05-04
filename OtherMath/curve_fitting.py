import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Simulate some synthetic data for illustration
years = np.arange(2000, 2021)
K_true = 1.0  # 100% market penetration
r_true = 0.3
t0_true = 2010
market_penetration = K_true / (1 + np.exp(-r_true * (years - t0_true))) + np.random.normal(scale=0.02, size=years.shape)

# Define the logistic growth model
def logistic_growth(t, K, r, t0):
    return K / (1 + np.exp(-r * (t - t0)))

# Fit the logistic model to the data
popt, pcov = curve_fit(logistic_growth, years, market_penetration, p0=[1, 0.3, 2010])

# Generate fitted data
fitted_penetration = logistic_growth(years, *popt)

# Plotting
plt.figure(figsize=(10, 6))
plt.scatter(years, market_penetration, label='Simulated Data', color='red')
plt.plot(years, fitted_penetration, label='Fitted Logistic Model', color='blue')
plt.title('Market Penetration of Solar Panels')
plt.xlabel('Year')
plt.ylabel('Market Penetration (Fraction)')
plt.legend()
plt.grid(True)
plt.show()

# Output estimated parameters
K_est, r_est, t0_est = popt
print(f"Estimated Carrying Capacity (K): {K_est:.2f}")
print(f"Estimated Growth Rate (r): {r_est:.2f}")
print(f"Estimated Inflection Point (Year): {int(t0_est)}")
