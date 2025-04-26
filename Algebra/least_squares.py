import numpy as np

def least_squares_polynomial(x_points,y_points,degree):
    """
    Perform a least squares method to fit 
    data to a function.

    Parameters:
    x_points (list or np.ndarray): x-coordinates
    y_points (list or np.npdarray): y-coordinates 

    Returns:
    np.ndarray: Coefficients of the best fit
    """

    x = np.array(x_points)
    y = np.array(y_points)

    A = np.vander(x,degree+1,increasing = False)

    ATA = A.T @ A
    ATy = A.T @ y

    coefficients = np.linalg.solve(ATA,ATy)

    return coefficients