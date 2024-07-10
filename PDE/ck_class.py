import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

class CrankNicolsonSolver:
    def __init__(self, a, b, alpha, beta, f, phi, nx, nt, T):
        """
        Initialize the Crank-Nicolson solver for a PDE.
        
        Parameters:
        - a, b: Functions representing the PDE coefficients.
        - alpha, beta: Boundary conditions at x=0 and x=L respectively.
        - f: Source term function.
        - phi: Initial condition function.
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
    
    def construct_matrices(self):
        """
        Construct the matrices A and B for the Crank-Nicolson method.
        """
        dx2 = self.dx ** 2
        dt2 = self.dt / 2
        
        A = sp.diags([-self.a / dx2 - self.b / (2 * self.dx), 
                      2 / dt2 + 2 * self.a / dx2, 
                      -self.a / dx2 + self.b / (2 * self.dx)], 
                     [-1, 0, 1], shape=(self.nx, self.nx)).tocsc()
        
        B = sp.diags([self.a / dx2 + self.b / (2 * self.dx), 
                      2 / dt2 - 2 * self.a / dx2, 
                      self.a / dx2 - self.b / (2 * self.dx)], 
                     [-1, 0, 1], shape=(self.nx, self.nx)).tocsc()
        
        # Applying boundary conditions
        A[0, 0] = A[-1, -1] = 1
        A[0, 1] = A[-1, -2] = 0
        
        B[0, 0] = B[-1, -1] = 1
        B[0, 1] = B[-1, -2] = 0
        
        return A, B
    
    def solve(self):
        """
        Solve the PDE using the Crank-Nicolson method.
        """
        A, B = self.construct_matrices()
        for n in range(0, self.nt - 1):
            b = B @ self.u[n, :] + self.f(self.x, n * self.dt)
            b[0] = self.alpha(n * self.dt)
            b[-1] = self.beta(n * self.dt)
            self.u[n + 1, :] = spla.spsolve(A, b)
    
    def get_solution(self):
        """
        Get the computed solution.
        
        Returns:
        - u: Solution matrix of shape (nt, nx).
        """
        return self.u

# Example usage:
def a(x): return 1
def b(x): return 0
def f(x, t): return 0
def phi(x): return np.sin(np.pi * x)
def alpha(t): return 0
def beta(t): return 0

nx, nt, T = 100, 1000, 1
solver = CrankNicolsonSolver(a, b, alpha, beta, f, phi, nx, nt, T)
solver.solve()
solution = solver.get_solution()
