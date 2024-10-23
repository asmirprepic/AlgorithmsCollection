def multivariate_gaussian_pdf(x, mean, cov):
    """
    Computes the Probability Density Function of a multivariate Gaussian distribution.

    Parameters:
    - x: Input vector.
    - mean: Mean vector.
    - cov: Covariance matrix.

    Returns:
    - pdf_value: The computed PDF value.
    """
    k = len(x)
    cov_det = np.linalg.det(cov)
    cov_inv = np.linalg.inv(cov)
    norm_const = 1.0 / (np.sqrt((2 * np.pi) ** k * cov_det))
    x_mu = x - mean
    result = norm_const * np.exp(-0.5 * x_mu.T @ cov_inv @ x_mu)
    return result
