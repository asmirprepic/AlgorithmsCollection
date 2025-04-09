def simulate_brownian_motion(
    n_paths: int, n_steps: int, T: float, seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate standard Brownian motion.

    Parameters:
        n_paths (int): Number of simulated paths
        n_steps (int): Number of time steps
        T (float): Time horizon
        seed (int): Random seed for reproducibility

    Returns:
        t (np.ndarray): Time grid
        W (np.ndarray): Simulated Brownian paths, shape (n_paths, n_steps + 1)
    """
    dt = T / n_steps
    t = np.linspace(0, T, n_steps + 1)
    np.random.seed(seed)
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(n_paths, n_steps))
    W = np.cumsum(dW, axis=1)
    W = np.hstack((np.zeros((n_paths, 1)), W))
    return t, W


def simulate_under_physical_measure(
    W: np.ndarray, t: np.ndarray, mu: float, sigma: float
) -> np.ndarray:
    """
    Simulate X_t under physical measure P: dX = mu dt + sigma dW

    Returns:
        X (np.ndarray): Simulated process under P
    """
    return mu * t + sigma * W


def girsanov_transform(W: np.ndarray, t: np.ndarray, theta: float) -> np.ndarray:
    """
    Apply Girsanov's transform: W^Q = W^P + theta * t

    Returns:
        W_Q (np.ndarray): Transformed Brownian motion under Q
    """
    return W + theta * t


def compute_rn_derivative(W_T: np.ndarray, theta: float, T: float) -> np.ndarray:
    """
    Compute Radon-Nikodym derivative dQ/dP at terminal time.

    Parameters:
        W_T (np.ndarray): Terminal Brownian motion value
        theta (float): Market price of risk
        T (float): Horizon

    Returns:
        Z (np.ndarray): Radon-Nikodym derivative values
    """
    return np.exp(-theta * W_T - 0.5 * theta**2 * T)
