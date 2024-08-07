import numpy as np

def ar_model(y,p):
    """
    Fits an AR(p) model to timeseries data y

    Parameters: 
    y: array-like, shape (n,)
    Time series data

    p: int
    Order of the autoregression

    Returns:
    phi: array, shape (p,)
    Estimated AR coefficitents
    
    intercept: float
    Estimated intercept
    """

    n = len(y)
    Y = y[p:]
    X = np.zeros((n-p,p))

    for t in range(p,n):
        X[t-p,:] = y[t-p:t][::-1]
    
    X = np.column_stack([np.ones(n-p),X])

    # Estimating the coeffitients
    coeffs = np.linalg.inv(X.T @ X) @ X.T @ Y
    intercept = coeffs[0]
    phi = coeffs[1:]

    return phi,intercept