import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from typing import Callable

class CrankNicolsonSolver:
    def __init__(self, a: Callable[[np.ndarray], np.ndarray], b: Callable[[np.ndarray], np.ndarray], 
                 alpha: Callable[[float], float], beta: Callable[[float], float], 
                 f: Callable[[np.ndarray, float], np.ndarray], phi: Callable[[np.ndarray], np.ndarray], 
                 nx: int, nt: int, T: float):
        """
        Initialize the Crank-Nicolson solver for a PDE.
        
        Parameters:
        - a: Function representing the PDE coefficient a(x).
        - b: Function representing the PDE coefficient b(x).
        - alpha: Function representing the boundary condition at x=0.
        - beta: Function representing the boundary condition at x=L.
        - f: Source term function f(x, t).
        - phi: Initial condition function phi(x).
        - nx: Number of spatial grid points.
        - nt: Number of time steps.
        - T: Total time for simulation.
        """
        self.a = a
        self.b = b
        self.alpha = alpha
        self.beta = beta
        self.f = f
        self.phi = phi
        self.nx = nx
        self.nt = nt
        self.T = T
        self.dx = 1 / (nx - 1)
        self.dt = T / nt
        self.x = np.linspace(0, 1, nx)
        self.u = np.zeros((nt, nx))
        self.u[0, :] = phi(self.x)
    
    def construct_matrices(self) -> tuple[sp.csc_matrix, sp.csc_matrix]:
        """
        Construct the matrices A and B for the Crank-Nicolson method.
        
        Returns:
        - A: Matrix for the left-hand side of the equation.
        - B: Matrix for the right-hand side of the equation.
        """
        dx2 = self.dx ** 2
        dt2 = self.dt / 2
        
        a_vals = self.a(self.x)
        b_vals = self.b(self.x)
        
        A = sp.diags([-a_vals / dx2 - b_vals / (2 * self.dx), 
                      2 / dt2 + 2 * a_vals / dx2, 
                      -a_vals / dx2 + b_vals / (2 * self.dx)], 
                     [-1, 0, 1], shape=(self.nx, self.nx)).tocsc()
        
        B = sp.diags([a_vals / dx2 + b_vals / (2 * self.dx), 
                      2 / dt2 - 2 * a_vals / dx2, 
                      a_vals / dx2 - b_vals / (2 * self.dx)], 
                     [-1, 0, 1], shape=(self.nx, self.nx)).tocsc()
        
        # Applying boundary conditions
        A[0, 0] = A[-1, -1] = 1
        A[0, 1] = A[-1, -2] = 0
        
        B[0, 0] = B[-1, -1] = 1
        B[0, 1] = B[-1, -2] = 0
        
        return A, B
    
    def solve(self) -> None:
        """
        Solve the PDE using the Crank-Nicolson method.
        """
        A, B = self.construct_matrices()
        for n in range(0, self.nt - 1):
            b = B @ self.u[n, :] + self.f(self.x, n * self.dt)
            b[0] = self.alpha(n * self.dt)
            b[-1] = self.beta(n * self.dt)
            self.u[n + 1, :] = spla.spsolve(A, b)
    
    def get_solution(self) -> np.ndarray:
        """
        Get the computed solution.
        
        Returns:
        - u: Solution matrix of shape (nt, nx).
        """
        return self.u

    def animate_solution(self) -> None:
        """
        Animate the solution over time.
        """
        fig, ax = plt.subplots()
        line, = ax.plot(self.x, self.u[0, :], lw=2)
        ax.set_ylim(-1, 1)
        ax.set_xlim(0, 1)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.set_title('Crank-Nicolson Solution')

        def update(frame):
            line.set_ydata(self.u[frame, :])
            return line,

        ani = FuncAnimation(fig, update, frames=range(0, self.nt, int(self.nt / 100)), blit=True)
        plt.show()

    def plot_final_solution(self) -> None:
        """
        Plot the final solution and compare to the analytical solution.
        """
        final_time = self.T
        analytical_solution = np.sin(np.pi * self.x) * np.exp(-np.pi**2 * final_time)
        
        plt.figure()
        plt.plot(self.x, self.u[-1, :], label='Numerical Solution')
        plt.plot(self.x, analytical_solution, label='Analytical Solution', linestyle='dashed')
        plt.xlabel('x')
        plt.ylabel('u')
        plt.title('Final Solution at t=T')
        plt.legend()
        plt.show()


# Example usage:
def a(x: np.ndarray) -> np.ndarray: return np.ones_like(x)
def b(x: np.ndarray) -> np.ndarray: return np.zeros_like(x)
def f(x: np.ndarray, t: float) -> np.ndarray: return np.zeros_like(x)
def phi(x: np.ndarray) -> np.ndarray: return np.sin(np.pi * x)
def alpha(t: float) -> float: return 0.0
def beta(t: float) -> float: return 0.0

nx, nt, T = 100, 1000, 1
solver = CrankNicolsonSolver(a, b, alpha, beta, f, phi, nx, nt, T)
solver.solve()
solution = solver.get_solution()
solver.plot_final_solution()

