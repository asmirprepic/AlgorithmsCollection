import numpy as np
import pandas as pd
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt


def simulate_iv_surface(strikes, maturities):
    """Simulate a simple implied volatility surface with noise."""
    K, T = np.meshgrid(strikes, maturities)
    base_iv = 0.2 + 0.05 * np.sin(K / 100) + 0.1 * np.log1p(T)
    noise = 0.02 * np.random.randn(*base_iv.shape)
    iv_surface = base_iv + noise
    return pd.DataFrame({
        'Strike': K.flatten(),
        'Maturity': T.flatten(),
        'IV': iv_surface.flatten()
    })


def check_butterfly_arbitrage(df):
    """Check for butterfly arbitrage using convexity of call prices."""
    arbitrage_flags = []
    for T in df['Maturity'].unique():
        sub = df[df['Maturity'] == T].sort_values('Strike')
        K = sub['Strike'].values
        IV = sub['IV'].values
        F = K[len(K)//2]  # assume ATM forward
        prices = black_scholes_price(F, K, T, IV)
        convexity = np.diff(prices, 2)
        flags = np.concatenate([[False], convexity < 0, [False]])
        arbitrage_flags.extend(flags)
    df['ButterflyArb'] = arbitrage_flags
    return df


def check_calendar_arbitrage(df):
    """Check for calendar arbitrage across maturities for fixed strikes."""
    flags = []
    for K in df['Strike'].unique():
        sub = df[df['Strike'] == K].sort_values('Maturity')
        maturities = sub['Maturity'].values
        IV = sub['IV'].values
        F = K  # assume ATM
        prices = black_scholes_price(F, K, maturities, IV)
        diffs = np.diff(prices)
        flags.extend(diffs < 0)
        flags.append(False)  # for last entry
    df['CalendarArb'] = flags
    return df


def black_scholes_price(F, K, T, sigma, r=0.0, option_type='call'):
    """Compute Black-Scholes price for European option."""
    from scipy.stats import norm
    T = np.maximum(T, 1e-6)  # avoid div-by-zero
    d1 = (np.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = np.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))
    else:
        price = np.exp(-r * T) * (K * norm.cdf(-d2) - F * norm.cdf(-d1))
    return price


def smooth_surface(df):
    """Smooth the IV surface using radial basis functions (RBF)."""
    rbf = Rbf(df['Strike'], df['Maturity'], df['IV'], function='multiquadric', smooth=0.01)
    smoothed_iv = rbf(df['Strike'], df['Maturity'])
    df['IV_Smoothed'] = smoothed_iv
    return df


def plot_surfaces(df):
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(df['Strike'], df['Maturity'], df['IV'], c='r', label='Original IV')
    ax1.set_title("Original IV Surface")
    ax1.set_xlabel("Strike")
    ax1.set_ylabel("Maturity")
    ax1.set_zlabel("IV")

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(df['Strike'], df['Maturity'], df['IV_Smoothed'], c='b', label='Smoothed IV')
    ax2.set_title("Arbitrage-Free Smoothed Surface")
    ax2.set_xlabel("Strike")
    ax2.set_ylabel("Maturity")
    ax2.set_zlabel("IV")

    plt.tight_layout()
    plt.show()
