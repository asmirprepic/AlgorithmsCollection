import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Step 1: Simulate Bond Data

def simulate_bond_data(num_bonds=10, start_maturity=1, end_maturity=10, base_yield=0.02, yield_slope=0.005, seed=42):
    np.random.seed(seed)
    maturities = np.linspace(start_maturity, end_maturity, num_bonds)
    yields = base_yield + yield_slope * maturities + 0.001 * np.random.randn(num_bonds)
    prices = 100 / (1 + yields)**maturities
    return pd.DataFrame({'Maturity': maturities, 'Price': prices, 'Yield': yields})

bond_data = simulate_bond_data()
print(bond_data)

# Step 2: Construct the Yield Curve using Bootstrapping

def bootstrap_yield_curve(bond_data):
    bond_data = bond_data.sort_values(by='Maturity')
    maturities = bond_data['Maturity'].values
    prices = bond_data['Price'].values
    yields = np.zeros(len(maturities))

    for i in range(len(maturities)):
        def objective(y):
            return np.abs(np.sum(np.exp(-y * maturities[i]) * 100 - prices[i]))
        
        res = minimize(objective, 0.02, bounds=[(0, None)])
        yields[i] = res.x[0]

    yield_curve = pd.DataFrame({'Maturity': maturities, 'Yield': yields})
    return yield_curve

yield_curve = bootstrap_yield_curve(bond_data)
print(yield_curve)

# Step 3: Analyze the Yield Curve

def plot_yield_curve(yield_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(yield_curve['Maturity'], yield_curve['Yield'], marker='o')
    plt.title('Yield Curve')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Yield')
    plt.grid(True)
    plt.show()

plot_yield_curve(yield_curve)

# Calculate and plot forward rates
def calculate_forward_rates(yield_curve):
    maturities = yield_curve['Maturity'].values
    yields = yield_curve['Yield'].values
    forward_rates = np.zeros(len(maturities) - 1)
    
    for i in range(1, len(maturities)):
        forward_rates[i - 1] = (yields[i] * maturities[i] - yields[i - 1] * maturities[i - 1]) / (maturities[i] - maturities[i - 1])

    forward_curve = pd.DataFrame({'Maturity': maturities[1:], 'Forward Rate': forward_rates})
    return forward_curve

forward_curve = calculate_forward_rates(yield_curve)
print(forward_curve)

def plot_forward_curve(forward_curve):
    plt.figure(figsize=(10, 6))
    plt.plot(forward_curve['Maturity'], forward_curve['Forward Rate'], marker='o', color='red')
    plt.title('Forward Rate Curve')
    plt.xlabel('Maturity (Years)')
    plt.ylabel('Forward Rate')
    plt.grid(True)
    plt.show()

plot_forward_curve(forward_curve)
