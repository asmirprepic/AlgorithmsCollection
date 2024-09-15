def mean_variance_optimization(mu, Sigma, target_return):
    """
    Solve the mean-variance optimization problem using linear algebra.

    Parameters:
    - mu: Expected returns of the assets (array of length n_assets).
    - Sigma: Covariance matrix of asset returns (n_assets x n_assets).
    - target_return: The desired expected return of the portfolio.

    Returns:
    - w: Optimal portfolio weights that minimize variance for the target return.
    """

    n = len(mu)
    ones = np.ones(n)

    # Assemble the matrices for the KKT conditions
    # | 2*Sigma   mu    ones |
    # |   mu^T     0     0   |
    # |  ones^T    0     0   |

    # Construct the KKT matrix and vector
    KKT_matrix = np.block([
        [2 * Sigma,         mu.reshape(-1, 1),      ones.reshape(-1, 1)],
        [mu.reshape(1, -1),      np.zeros((1, 1)),        np.zeros((1, 1))],
        [ones.reshape(1, -1),    np.zeros((1, 1)),        np.zeros((1, 1))]
    ])

    KKT_vector = np.concatenate([np.zeros(n), [target_return], [1]])

    # Solve the KKT system
    solution = np.linalg.solve(KKT_matrix, KKT_vector)

    # Extract the weights
    w = solution[:n]

    return w
