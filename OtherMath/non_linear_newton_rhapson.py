import numpy as np

def jacobian(f, x, eps=1e-6):
    """
    Compute the Jacobian matrix using finite differences.
    
    Parameters:
        f (function): A vector function returning a NumPy array.
        x (np.ndarray): Current estimate of the root.
        eps (float): Perturbation for finite difference approximation.
    
    Returns:
        np.ndarray: Approximated Jacobian matrix.
    """
    n = len(x)
    J = np.zeros((n, n))
    f_x = f(x)  
    
    for i in range(n):
        x_step = x.copy()
        x_step[i] += eps  
        J[:, i] = (f(x_step) - f_x) / eps  # Finite difference approximation
    
    return J

def newton_raphson_solver(f, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson method for solving nonlinear systems of equations.
    
    Parameters:
        f (function): Function defining the system f(x) = 0.
        x0 (np.ndarray): Initial guess.
        tol (float): Tolerance for convergence.
        max_iter (int): Maximum number of iterations.
    
    Returns:
        np.ndarray: Solution vector if converged.
    """
    x = x0.copy()
    
    for i in range(max_iter):
        J = jacobian(f, x)  # Compute Jacobian matrix
        f_x = f(x)          # Compute function value at x
        
        if np.linalg.norm(f_x, ord=2) < tol:  
            return x
        
        try:
            delta_x = np.linalg.solve(J, -f_x)  # Solve J * delta_x = -f_x
        except np.linalg.LinAlgError:
            raise ValueError("Jacobian is singular or nearly singular")
        
        x += delta_x  
        
        if np.linalg.norm(delta_x, ord=2) < tol:  
            return x
    
    raise ValueError("Newton-Raphson method did not converge within max iterations")


def system_of_equations(x):
    """
    Example nonlinear system:
    f1(x, y) = x^2 + y^2 - 4
    f2(x, y) = x * y - 1
    """
    return np.array([
        x[0]**2 + x[1]**2 - 4
