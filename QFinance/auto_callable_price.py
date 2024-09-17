import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulate the underlying asset prices using Geometric Brownian Motion for multiple assets
def simulate_multi_asset_paths(S0, r, sigma, corr_matrix, T, M, I):
    """
    Simulate multiple asset price paths using correlated Geometric Brownian Motion (GBM).
    
    Parameters:
    S0 : list : initial stock prices for each asset
    r : float : risk-free rate
    sigma : list : volatilities of each asset
    corr_matrix : np.ndarray : correlation matrix for the assets
    T : float : time to maturity in years
    M : int : number of time steps
    I : int : number of simulations
    
    Returns:
    asset_paths : np.ndarray : simulated asset price paths for each asset
    """
    dt = T / M
    num_assets = len(S0)
    asset_paths = np.zeros((M + 1, num_assets, I))
    asset_paths[0] = S0[:, np.newaxis]

    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(corr_matrix)

    for t in range(1, M + 1):
        z = np.random.standard_normal((num_assets, I))
        correlated_z = L @ z
        for i in range(num_assets):
            asset_paths[t, i] = asset_paths[t - 1, i] * np.exp(
                (r - 0.5 * sigma[i] ** 2) * dt + sigma[i] * np.sqrt(dt) * correlated_z[i]
            )
    
    return asset_paths

# Pricing a multi-asset autocallable product with a barrier
def price_autocallable(S0, K, r, sigma, corr_matrix, T, M, I, redemption_level, barrier_level, coupon, early_redemption_dates):
    """
    Price a multi-asset autocallable structured product using Monte Carlo simulation.
    
    Parameters:
    S0 : list : initial stock prices for each asset
    K : list : strike prices for each asset
    r : float : risk-free rate
    sigma : list : volatilities for each asset
    corr_matrix : np.ndarray : correlation matrix for the assets
    T : float : time to maturity in years
    M : int : number of time steps
    I : int : number of simulations
    redemption_level : float : early redemption level for each asset
    barrier_level : float : barrier level for knock-in/knock-out events
    coupon : float : coupon payment for the product
    early_redemption_dates : list : dates where early redemption is possible
    
    Returns:
    product_price : float : calculated product price
    """
    asset_paths = simulate_multi_asset_paths(np.array(S0), r, sigma, corr_matrix, T, M, I)
    
    # Determine the autocallable structure and calculate payoffs
    num_assets = len(S0)
    dt = T / M
    early_redemption_steps = [int(date / dt) for date in early_redemption_dates]
    
    final_payoff = np.zeros(I)
    for i in range(I):
        redeemed = False
        for t in early_redemption_steps:
            if all(asset_paths[t, :, i] >= redemption_level * S0):  # Check if all assets are above redemption level
                final_payoff[i] = coupon * (t * dt / T)  # Early redemption: pay coupon proportional to time
                redeemed = True
                break
        
        if not redeemed:  # Product matures without early redemption
            if all(asset_paths[-1, :, i] >= K):  # Full payoff
                final_payoff[i] = coupon
            elif any(asset_paths[:, :, i].min(axis=0) < barrier_level * S0):  # Knock-in event
                final_payoff[i] = np.minimum(0, asset_paths[-1, :, i].min(axis=0) - K)  # Loss scenario
    
    # Discount the expected payoff back to present value
    product_price = np.exp(-r * T) * np.mean(final_payoff)
    
    return product_price

# Parameters for the autocallable structured product
S0 = [100, 100]  # Initial stock prices for two assets
K = [100, 100]   # Strike prices for both assets
r = 0.03         # Risk-free rate
sigma = [0.2, 0.25]  # Volatility of both assets
corr_matrix = np.array([[1, 0.5], [0.5, 1]])  # Correlation matrix between assets
T = 1            # Time to maturity (1 year)
M = 100          # Number of time steps
I = 100000       # Number of simulations
redemption_level = 1.05  # Early redemption if assets exceed 105% of initial price
barrier_level = 0.7  # Barrier level for knock-in event (70% of initial price)
coupon = 0.08    # Coupon payment (8% of notional)
early_redemption_dates = [0.25, 0.5, 0.75]  # Possible early redemption at 3 months, 6 months, 9 months

# Pricing the autocallable product
autocallable_price = price_autocallable(S0, K, r, sigma, corr_matrix, T, M, I, redemption_level, barrier_level, coupon, early_redemption_dates)

# Display the result
print(f"Autocallable Structured Product Price: {autocallable_price:.2f}")

# Plotting a few simulated asset paths for visualization
def plot_multi_asset_paths(asset_paths, T, M):
    time_steps = np.linspace(0, T, M + 1)
    plt.figure(figsize=(10, 6))
    for i in range(asset_paths.shape[1]):
        plt.plot(time_steps, asset_paths[:, i, :5], label=f'Asset {i+1}')  # Plot 5 sample paths per asset
    plt.title('Simulated Asset Price Paths')
    plt.xlabel('Time (Years)')
    plt.ylabel('Asset Price')
    plt.legend()
    plt.show()

# Simulate and plot multi-asset paths
simulated_paths = simulate_multi_asset_paths(np.array(S0), r, sigma, corr_matrix, T, M, 5)
plot_multi_asset_paths(simulated_paths, T, M)
