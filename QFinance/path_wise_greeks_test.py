import numpy as np
import matplotlib.pyplot as plt

# Parameters
S0 = 100         # Initial stock price
K = 100          # Strike price
T = 1            # Time to maturity (1 year)
r = 0.05         # Risk-free rate
sigma = 0.2      # Volatility
num_paths = 100000  # Number of Monte Carlo simulations


def simulate_gbm(S0, T, r, sigma, num_paths):
    """
    Simulate asset paths using Geometric Brownian Motion.
    """
    dt = T / num_paths
    Z = np.random.normal(0, 1, num_paths)
    return S0 * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)


def calculate_pathwise_greeks(S0, K, T, r, sigma, num_paths):
    """
    Calculate Delta, Gamma, and Vega using pathwise sensitivities.
    """
    ST = simulate_gbm(S0, T, r, sigma, num_paths)
    discount_factor = np.exp(-r * T)
    payoffs = np.maximum(ST - K, 0)  

    
    deltas = np.where(ST > K, 1, 0) * discount_factor

    
    gammas = np.zeros_like(deltas)  

    
    vegas = (np.where(ST > K, (ST - K), 0) * discount_factor) * (np.log(ST / S0) / sigma)

    
    delta = np.mean(deltas)
    gamma = np.mean(gammas)
    vega = np.mean(vegas)
    
    return delta, gamma, vega




def plot_results(S0, K, T, r, sigma, num_paths):
    ST = simulate_gbm(S0, T, r, sigma, num_paths)
    payoffs = np.maximum(ST - K, 0)
    deltas = np.where(ST > K, 1, 0)

    plt.figure(figsize=(12, 6))
    plt.scatter(ST, payoffs, alpha=0.3, label="Payoffs")
    plt.plot(ST, deltas, color="red", label="Delta", alpha=0.7)
    plt.title("Payoffs and Pathwise Delta")
    plt.xlabel("Stock Price at Maturity")
    plt.ylabel("Value")
    plt.legend()
    plt.show()

plot_results(S0, K, T, r, sigma, num_paths)
