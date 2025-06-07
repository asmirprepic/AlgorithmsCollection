mport numpy as np

def euler_maruyama(drift, diffusion, x0, t0, T, dt, num_paths=1, seed=None):
    """
    Numerically simulate an SDE using the Euler-Maruyama method.
    
    SDE: dX(t) = drift(t, X(t)) dt + diffusion(t, X(t)) dW(t).
    
    Parameters:
    -----------
    drift : callable
        Drift coefficient function, drift(t, x), returning a float or array.
    diffusion : callable
        Diffusion coefficient function, diffusion(t, x), returning a float or array.
    x0 : float
        Initial condition X(t0).
    t0 : float
        Start time.
    T : float
        End time.
    dt : float
        Time step size.
    num_paths : int, optional
        Number of simulation paths (default: 1).
    seed : int, optional
        Random seed for reproducibility (default: None).
    
    Returns:
    --------
    times : ndarray
        Array of time points [t0, t0+dt, ..., T].
    paths : ndarray
        Array of shape (num_steps, num_paths) containing simulated paths.
    """
    if seed is not None:
        np.random.seed(seed)
    
    num_steps = int((T - t0) / dt) + 1
    times = np.linspace(t0, T, num_steps)
    paths = np.zeros((num_steps, num_paths))
    paths[0, :] = x0
    
    dW = np.random.normal(0, np.sqrt(dt), size=(num_steps-1, num_paths))
    
    for i in range(num_steps-1):
        t = times[i]
        x = paths[i, :]
        paths[i+1, :] = x + drift(t, x) * dt + diffusion(t, x) * dW[i, :]
    
    return times, paths
