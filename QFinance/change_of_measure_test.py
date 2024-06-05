import numpy as np
import matplotlib.pyplot as plt

def simulate_gbm(S0, mu, sigma, T, dt, M):
    """
    Simulate Geometric Brownian Motion (GBM) paths under the real-world measure.

    Parameters:
    S0 : float - Initial stock price
    mu : float - Drift under the real-world measure
    sigma : float - Volatility
    T : float - Time to maturity
    dt : float - Time step size
    M : int - Number of simulations

    Returns:
    t : numpy.ndarray - Time vector
    S : numpy.ndarray - Simulated stock price paths
    """
    N = int(T / dt)
    t = np.linspace(0, T, N + 1)
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for i in range(1, N + 1):
        Z = np.random.normal(0, 1, M)
        S[:, i] = S[:, i - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    
    return t, S

def change_measure(S, mu, r, sigma, dt):
    """
    Change the measure from the real-world (P) to the risk-neutral (Q).

    Parameters:
    S : numpy.ndarray - Simulated stock price paths under the real-world measure
    mu : float - Drift under the real-world measure
    r : float - Risk-free rate
    sigma : float - Volatility
    dt : float - Time step size

    Returns:
    S_Q : numpy.ndarray - Simulated stock price paths under the risk-neutral measure
    """
    M, N = S.shape
    Z = (np.log(S[:, 1:] / S[:, :-1]) - (mu - 0.5 * sigma ** 2) * dt) / (sigma * np.sqrt(dt))
    Z = np.concatenate((np.zeros((M, 1)), Z), axis=1)  # Add zero for the initial time step

    S_Q = np.zeros_like(S)
    S_Q[:, 0] = S[:, 0]
    for i in range(1, N):
        S_Q[:, i] = S_Q[:, i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, i])
    
    return S_Q

def price_european_call(S_Q, K, r, T):
    """
    Price a European call option under the risk-neutral measure.

    Parameters:
    S_Q : numpy.ndarray - Simulated stock price paths under the risk-neutral measure
    K : float - Strike price
    r : float - Risk-free rate
    T : float - Time to maturity

    Returns:
    option_price : float - European call option price
    """
    payoff = np.maximum(S_Q[:, -1] - K, 0)
    option_price = np.exp(-r * T) * np.mean(payoff)
    return option_price

def plot_paths(t, S, S_Q):
    """
    Plot the simulated paths under both real-world and risk-neutral measures.

    Parameters:
    t : numpy.ndarray - Time vector
    S : numpy.ndarray - Simulated stock price paths under the real-world measure
    S_Q : numpy.ndarray - Simulated stock price paths under the risk-neutral measure
    """
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    for i in range(10):
        plt.plot(t, S[i], lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Asset Price Paths under Real-World Measure (P)')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    for i in range(10):
        plt.plot(t, S_Q[i], lw=0.5)
    plt.xlabel('Time')
    plt.ylabel('Asset Price')
    plt.title('Asset Price Paths under Risk-Neutral Measure (Q)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def plot_distributions(S, S_Q):
    """
    Plot the distributions of terminal asset prices under both measures.

    Parameters:
    S : numpy.ndarray - Simulated stock price paths under the real-world measure
    S_Q : numpy.ndarray - Simulated stock price paths under the risk-neutral measure
    """
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.hist(S[:, -1], bins=50, alpha=0.75, edgecolor='black')
    plt.xlabel('Asset Price')
    plt.ylabel('Frequency')
    plt.title('Terminal Asset Prices under Real-World Measure (P)')

    plt.subplot(1, 2, 2)
    plt.hist(S_Q[:, -1], bins=50, alpha=0.75, edgecolor='black')
    plt.xlabel('Asset Price')
    plt.ylabel('Frequency')
    plt.title('Terminal Asset Prices under Risk-Neutral Measure (Q)')

    plt.tight_layout()
    plt.show()

# Parameters
S0 = 100        # Initial stock price
mu = 0.08       # Drift under the real-world measure
sigma = 0.2     # Volatility
T = 1.0         # Time to maturity (1 year)
dt = 0.01       # Time step size
M = 1000        # Number of simulations
r = 0.05        # Risk-free rate
K = 100         # Strike price of the European call option

# Simulate asset prices under the real-world measure (P)
t, S = simulate_gbm(S0, mu, sigma, T, dt, M)

# Change measure to risk-neutral (Q)
S_Q = change_measure(S, mu, r, sigma, dt)

# Price the European call option under the risk-neutral measure
option_price = price_european_call(S_Q, K, r, T)
print(f"European Call Option Price under Risk-Neutral Measure: {option_price:.2f}")

# Plot the simulated paths under both measures
plot_paths(t, S, S_Q)

# Plot the distributions of terminal asset prices under both measures
plot_distributions(S, S_Q)
