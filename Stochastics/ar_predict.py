import numpy as np

def ar_predict(y, phi, intercept,p,steps = 10):
    """
    Makes predictions using the parameters of the ar model and data. 

    Parameters: 
    y: array-like, shape (n,) Time series data
    phi: array, shape (p,) Estimated AR coefficitients
    intercept: float. Estimated intercept
    p: int. Order of the AR model
    steps: int . Number of steps to predict

    Returns: 
    y_pred: array, shape (steps,). Predicted values
    """

    y_pred = np.zeros(steps)
    y_hist = y[-p:]

    for t in range(steps):
        y_pred[t] = intercept + np.dot(phi,y_hist[::-1])
        y_hist = np.roll(y_hist,-1)
        y_hist[-1] = y_pred[t]

    return y_pred
