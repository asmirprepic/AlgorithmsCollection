import numpy as np
import matplotlib.pyplot as plt

# Model parameters
num_factors = 4
num_simulations = 1000
num_steps = 252  # daily steps for 1 year
T = 1.0  # time horizon in years
dt = T / num_steps
maturities = np.array([0.25, 0.5, 0.75, 1.0])
initial_forward_rates = np.array([0.02, 0.025, 0.03, 0.035])
volatilities = np.array([0.01, 0.015, 0.02, 0.025])
correlations = np.array([
    [1.0, 0.9, 0.7, 0.5],
    [0.9, 1.0, 0.8, 0.6],
    [0.7, 0.8, 1.0, 0.9],
    [0.5, 0.6, 0.9, 1.0]
])

# Calculate Cholesky decomposition of the correlation matrix
chol_corr = np.linalg.cholesky(correlations)

# Generate correlated random variables
def generate_correlated_normals(num_steps, num_simulations, num_factors, chol_corr):
    uncorrelated_normals = np.random.normal(size=(num_steps, num_simulations, num_factors))
    correlated_normals = np.einsum('ij,tmj->tmi', chol_corr, uncorrelated_normals)
    return correlated_normals

# Simulate forward rate paths
def simulate_forward_rate_paths(initial_forward_rates, volatilities, correlated_normals, dt):
    num_maturities = len(initial_forward_rates)
    forward_rates = np.zeros((num_steps + 1, num_maturities, num_simulations))
    forward_rates[0] = initial_forward_rates[:, np.newaxis]
    
    for t in range(1, num_steps + 1):
        for i in range(num_maturities):
            drift = 0  # drift is zero in the risk-neutral measure for LMM
            diffusion = volatilities[i] * correlated_normals[t-1, :, i]
            forward_rates[t, i] = forward_rates[t-1, i] * np.exp(drift * dt + diffusion * np.sqrt(dt))
    
    return forward_rates

# Generate correlated random variables
correlated_normals = generate_correlated_normals(num_steps, num_simulations, num_factors, chol_corr)

# Simulate forward rate paths
forward_rate_paths = simulate_forward_rate_paths(initial_forward_rates, volatilities, correlated_normals, dt)

# Price a European swaption
def price_swaption(forward_rate_paths, strike_rate, T, num_steps, dt, num_simulations):
    payoffs = np.maximum(forward_rate_paths[-1, -1] - strike_rate, 0)
    discounted_payoffs = np.exp(-initial_forward_rates[-1] * T) * payoffs
    swaption_price = np.mean(discounted_payoffs)
    return swaption_price

strike_rate = 0.03
swaption_price = price_swaption(forward_rate_paths, strike_rate, T, num_steps, dt, num_simulations)

print(f"The price of the European swaption is: ${swaption_price:.2f}")

# Plot some simulated forward rate paths
plt.figure(figsize=(12, 6))
for i in range(10):  # Plot 10 simulated paths for each maturity
    plt.plot(forward_rate_paths[:, -1, i], label=f'Path {i+1}')
plt.title('Simulated Forward Rate Paths')
plt.xlabel('Time Steps')
plt.ylabel('Forward Rate')
plt.legend()
plt.grid(True)
plt.show()
