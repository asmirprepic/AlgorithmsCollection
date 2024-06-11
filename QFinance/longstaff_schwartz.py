import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def generate_asset_paths(S0, r, sigma, T, M, I):
    dt = T / M
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        paths[t] = paths[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return paths

def longstaff_schwartz(S0, K, r, sigma, T, M, I):
    paths = generate_asset_paths(S0, r, sigma, T, M, I)
    dt = T / M
    H = np.maximum(K - paths, 0)
    V = np.copy(H)
    
    for t in range(M - 1, 0, -1):
        regression = LinearRegression()
        itm = np.where(H[t] > 0)[0]
        if len(itm) == 0:
            continue
        X = paths[t, itm].reshape(-1, 1)
        y = V[t+1, itm] * np.exp(-r * dt)
        regression.fit(X, y)
        C = regression.predict(X)
        exercise = np.where(H[t, itm] > C)[0]
        V[t, itm[exercise]] = H[t, itm[exercise]]
        V[t, itm] *= np.exp(-r * dt)
    
    V0 = np.mean(V[1] * np.exp(-r * dt))
    return V0

# Example
S0 = 100
K = 100
r = 0.05
sigma = 0.2
T = 1.0
M = 50
I = 10000


option_price = longstaff_schwartz(S0, K, r, sigma, T, M, I)
print(f"American Put Option Price (Longstaff-Schwartz): {option_price:.4f}")


paths = generate_asset_paths(S0, r, sigma, T, M, 10)
plt.plot(paths)
plt.title('Simulated Asset Paths')
plt.xlabel('Time Steps')
plt.ylabel('Asset Price')
plt.grid(True)
plt.show()
