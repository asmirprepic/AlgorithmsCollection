
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class HeatEquationSolver:
    """
    A class used to represent and solve the one-dimensional heat equation using the Crank-Nicolson method.
    
    Attributes:
    -----------
    N : int
        Number of discretization points in space.
    M : int
        Number of discretization points in time.
    T : float
        The total time to solve for.
    alpha : float
        The thermal diffusivity constant.
    """

    def __init__(self, N, M, T, alpha):
        """
            Solve the heat equation u_t = alpha * u_xx using the Crank-Nicolson method.
            
            Parameters:
            N : int
                Number of discretization points in space.
            M : int
                Number of discretization points in time.
            T : float
                The total time to solve for.
            alpha : float
                The thermal diffusivity constant.
            
            Returns:
            u : numpy.ndarray
                The numerical solution of heat equation at time T.
        """
        self.N = N
        self.M = M
        self.T = T
        self.alpha = alpha
        self.dt = T / M
        self.dx = 1 / N
        self.r = alpha * self.dt / self.dx**2

    def simulate(self):
        """
        Simulates the temperature evolution using the Crank-Nicolson method.
        
        Returns:
        --------
        numpy.ndarray
            The simulated temperature distribution at final time T.
        """
        # Initialize the temperature distribution array
        u = np.zeros((self.M + 1, self.N + 1), dtype=np.float64)
        u[0, :] = np.sin(np.pi * np.linspace(0, 1, self.N + 1))   # Initial condition (example: u(x,0) = sin(pi*x))
        
        # Create the diagonals for the Crank-Nicolson method
        A_diag = (1 + self.r) * np.ones(self.N + 1)
        off_diag = -self.r / 2 * np.ones(self.N)
        A = np.diag(A_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
        B = 2 * (1 - self.r) * np.diag(np.ones(self.N + 1)) + self.r * np.diag(np.ones(self.N), -1) + self.r * np.diag(np.ones(self.N), 1)
        
        # Time-stepping loop
        for t in range(1, self.M + 1):
            # Update right-hand side
            b = B @ u[t-1, :]
            # Solve system of equations
            u[t, :] = linalg.solve(A, b)
        
        self.u = u
        return u

    def plot_temperature_distribution(self):
        """
        Plots the temperature distribution across the spatial domain at final time T.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(np.linspace(0, 1, self.N + 1), self.u[-1, :])
        plt.title(f"Temperature Distribution at Final Time T={self.T}")
        plt.xlabel("Position (x)")
        plt.ylabel("Temperature (u)")
        plt.grid(True)
        plt.show()

    def animate_temperature(self):
        """
        Creates an animation showing the evolution of temperature distribution over time.
        """
        fig, ax = plt.subplots()
        line, = ax.plot([], [], lw=2)
        
        def init():
            line.set_data([], [])
            return line,

        def update(t):
            x = np.linspace(0, 1, self.N + 1)
            y = self.u[t, :]
            line.set_data(x, y)
            return line,

        self.ani = FuncAnimation(fig, update, frames=range(0, self.M + 1), init_func=init, blit=True)
        plt.show()


# Example Usage
solver = HeatEquationSolver(N=50, M=1000, T=0.5, alpha=0.01)
solver.simulate()
solver.plot_temperature_distribution()
solver.animate_temperature()
