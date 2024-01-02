import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm

class CrankNicolsonEuropeanOption:
    """
    A class to price European options using the Crank-Nicolson finite difference method.
    """

    def __init__(self, S_max, K, T, r, sigma, M, N, option_type='call'):
        """
        Initializes the parameters for the Crank-Nicolson option pricing model.
        """
        self.S_max = S_max
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.M = M
        self.N = N
        self.option_type = option_type
        self.dt = T / M
        self.dS = S_max / N
        self.grid = np.zeros((M+1, N+1))
        self.S = np.linspace(0, S_max, N+1)

    def solve(self):
        """
        Solves the PDE using the Crank-Nicolson method and updates the price grid.
        """
        # Set up boundary conditions at maturity
        if self.option_type == 'call':
            self.grid[-1, :] = np.maximum(self.S - self.K, 0)
        elif self.option_type == 'put':
            self.grid[-1, :] = np.maximum(self.K - self.S, 0)
        
        # Coefficients for the Crank-Nicolson scheme
        alpha = 0.25 * self.dt * ((self.sigma**2) * (np.arange(self.N+1)**2) - self.r * np.arange(self.N+1))
        beta = -self.dt * 0.5 * ((self.sigma**2) * (np.arange(self.N+1)**2) + self.r)
        gamma = 0.25 * self.dt * ((self.sigma**2) * (np.arange(self.N+1)**2) + self.r * np.arange(self.N+1))
        
        # Tridiagonal matrix setup
        M1 = -np.diag(alpha[2:self.N], -1) + np.diag(1 - beta[1:self.N]) - np.diag(gamma[1:self.N-1], 1)
        M2 = np.diag(alpha[2:self.N], -1) + np.diag(1 + beta[1:self.N]) + np.diag(gamma[1:self.N-1], 1)

        # Solve the system at each time step
        for i in range(self.M, 0, -1):
            self.grid[i-1, 1:self.N] = linalg.solve(M1, M2 @ self.grid[i, 1:self.N])
        
    def plot_price_surface(self):
        """
        Plots the price surface over stock price and time to maturity.
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Creating mesh for stock price (S) and time (T) dimensions
        T_grid, S_grid = np.meshgrid(np.linspace(0, self.T, self.M+1), self.S)
        
        # Ensure the grids are correctly oriented
        Z = np.transpose(self.grid)  # Transpose grid to align with S and T meshes
        
        # Plotting the surface
        ax.plot_surface(S_grid, T_grid, Z, cmap=cm.coolwarm)
        
        ax.set_title('Option Price Surface')
        ax.set_xlabel('Stock Price')
        ax.set_ylabel('Time to Maturity')
        ax.set_zlabel(f'{self.option_type.capitalize()} Option Price')
        plt.show()

    def get_option_price(self, S0):
        """
        Retrieves the option price from the grid for a given current stock price S0.
        """
        return np.interp(S0, self.S, self.grid[0, :])

# Example usage
S_max = 100
K = 50
T = 1
r = 0.05
sigma = 0.2
M = 1000
N = 100
option_type = 'call'

model = CrankNicolsonEuropeanOption(S_max, K, T, r, sigma, M, N, option_type)
model.solve()
print(f"The {option_type} option price is: {model.get_option_price(S_max/2)}")
model.plot_price_surface()
