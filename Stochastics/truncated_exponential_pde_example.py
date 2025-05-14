import numpy as np
import scipy

def solve_truncated_exponential_pde(lambda_: float, c: float, N: int = 100, T: float = 1.0) -> np.ndarray:
    """
    Solve a truncated exponential process using finite differences.

    Args:
        lambda_ (float): Rate parameter of the exponential process.
        c (float): Upper bound.
        N (int): Number of spatial grid points.
        T (float): Time horizon.

    Returns:
        np.ndarray: Solution vector at time T.
    """
    x_vals = np.linspace(0, c, N)
    dx = x_vals[1] - x_vals[0]

    
    A = np.diag([-lambda_] * (N - 1), k=0) + np.diag([lambda_] * (N - 1), k=-1)
    A[-1, :] = 0  # Enforce boundary condition p(c) = e^(-λc)

    
    p0 = np.ones(N)
    p0[-1] = np.exp(-lambda_ * c)

    
    p_T = scipy.linalg.expm(A * T) @ p0
    return p_T
