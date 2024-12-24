import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

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
