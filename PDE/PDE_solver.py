import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve

class PDESolver:
    def __init__(self, N, M, T):
        self.N = N
        self.M = M
        self.T = T
        self.dt = T / M
        self.dx = 1 / N
        self.u = np.zeros((M + 1, N + 1))
        
    def set_initial_conditions(self, u_init):
        self.u[0, :] = u_init(np.linspace(0, 1, self.N + 1))
        
    def set_boundary_conditions(self, u_left, u_right):
        self.u_left = u_left
        self.u_right = u_right
        
    def simulate(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
        
    def plot_result(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, self.N + 1), self.u[-1, :])
        plt.title(f"Result at Final Time T={self.T}")
        plt.xlabel("Position (x)")
        plt.ylabel("u")
        plt.grid(True)
        plt.show()
        
    def check_stability(self):
        raise NotImplementedError("This method should be implemented by subclasses.")
