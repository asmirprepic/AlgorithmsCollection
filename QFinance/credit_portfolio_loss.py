import numpy as np
from scipy.stats import norm

def credit_portfolio_loss(
    default_probs: np.ndarray, rho: float, num_simulations: int
) -> np.ndarray:
    """
    Simulate credit portfolio loss distribution using a Gaussian copula.

    Parameters:
    - default_probs (np.ndarray): Array of default probabilities for each asset in the portfolio.
    - rho (float): Correlation coefficient between assets (0 < rho < 1).
    - num_simulations (int): Number of Monte Carlo simulations.

    Returns:
    - np.ndarray: Portfolio loss distribution.
    """
    num_assets = len(default_probs)
    portfolio_losses = []

    # Cholesky decomposition for correlated normal variates
    cov_matrix = rho * np.ones((num_assets, num_assets)) + (1 - rho) * np.eye(num_assets)
    cholesky_matrix = np.linalg.cholesky(cov_matrix)

    for _ in range(num_simulations):
        # Generate correlated normal variables
        Z = np.random.normal(size=num_assets)
        correlated_Z = cholesky_matrix @ Z

        # Transform to uniform distribution (Gaussian copula)
        uniform_vars = norm.cdf(correlated_Z)

        # Simulate defaults based on each asset's default probability
        defaults = uniform_vars < default_probs
        loss = np.sum(defaults)  # Sum defaults as a simple loss measure
        portfolio_losses.append(loss)

    return np.array(portfolio_losses)

# Example usage
default_probs = np.array([0.02, 0.03, 0.015, 0.05, 0.025])  # Default probabilities for each asset
rho = 0.3  # Correlation between assets
num_simulations = 10000

loss_distribution = credit_portfolio_loss(default_probs, rho, num_simulations)

# Plotting the loss distribution
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(loss_distribution, bins=30, density=True, alpha=0.7, color='blue')
plt.xlabel("Number of Defaults")
plt.ylabel("Probability Density")
plt.title("Credit Portfolio Loss Distribution")
plt.show()
