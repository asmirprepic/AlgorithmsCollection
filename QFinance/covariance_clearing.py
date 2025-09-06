import numpy as np
from numpy.linalg import eigh

def rmt_covariance_cleaning(returns: np.ndarray) -> np.ndarray:
    """
    Random Matrix Theory (RMT) covariance cleaning.
    Replaces noisy eigenvalues inside Marčenko–Pastur bulk with their average.

    Parameters
    ----------
    returns : (T, N) matrix of returns (T obs, N assets)

    Returns
    -------
    Sigma_clean : (N, N) cleaned covariance matrix
    """
    T, N = returns.shape
    X = returns - returns.mean(axis=0)
    Sigma = np.cov(X, rowvar=False)

    eigvals, eigvecs = eigh(Sigma)

    q = N / T
    lambda_min = (1 - np.sqrt(q)) ** 2
    lambda_max = (1 + np.sqrt(q)) ** 2

    mask_noise = (eigvals > lambda_min) & (eigvals < lambda_max)
    avg_noise = eigvals[mask_noise].mean() if np.any(mask_noise) else 0.0

    eigvals_clean = eigvals.copy()
    eigvals_clean[mask_noise] = avg_noise

    Sigma_clean = (eigvecs * eigvals_clean) @ eigvecs.T
    return Sigma_clean
