import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit

# Logistic growth function
def logistic_growth(t, S, alpha, K):
    return alpha * S * (1 - S / K)

# Parameters
alpha = 0.5  # Growth rate
K = 100      # Carrying capacity (maximum sentiment level)
S0 = 10      # Initial sentiment
T = 20       # Time horizon (e.g., days)

# Solve ODE
t_span = (0, T)
t_eval = np.linspace(0, T, 500)
solution = solve_ivp(logistic_growth, t_span, [S0], args=(alpha, K), t_eval=t_eval)

# Plot sentiment over time
plt.figure(figsize=(10, 6))
plt.plot(solution.t, solution.y[0], label="Sentiment (S)")
plt.axhline(K, color="red", linestyle="--", label="Carrying Capacity (K)")
plt.xlabel("Time")
plt.ylabel("Sentiment")
plt.title("Sentiment Dynamics (Logistic Growth)")
plt.legend()
plt.show()


def logistic_model(t, alpha, K):
    return K / (1 + (K - S0) / S0 * np.exp(-alpha * t))

# Generate synthetic data
noise = np.random.normal(0, 2, len(t_eval))
sentiment_data = logistic_model(t_eval, alpha, K) + noise

# Estimate parameters
popt, pcov = curve_fit(logistic_model, t_eval, sentiment_data, p0=[0.4, 90])
alpha_est, K_est = popt

# Plot actual vs. fitted
plt.figure(figsize=(10, 6))
plt.plot(t_eval, sentiment_data, label="Synthetic Sentiment Data", linestyle="dotted")
plt.plot(t_eval, logistic_model(t_eval, *popt), label=f"Fitted Model (α={alpha_est:.2f}, K={K_est:.2f})")
plt.axhline(K, color="red", linestyle="--", label="True Carrying Capacity (K)")
plt.xlabel("Time")
plt.ylabel("Sentiment")
plt.title("Sentiment Dynamics with Parameter Estimation")
plt.legend()
plt.show()
