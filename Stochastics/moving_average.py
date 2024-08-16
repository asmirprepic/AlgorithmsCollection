import numpy as np

def moving_average(data, window_size):
    """
    Compute the moving average of a 1D array.

    Parameters:
    data (array-like): The input data array.
    window_size (int): The number of data points to include in each average.

    Returns:
    numpy.ndarray: The array of moving averages.
    """
    cumsum = np.cumsum(np.insert(data, 0, 0)) 
    moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
    return moving_avg

# Example usage
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
window_size = 3
ma = moving_average(data, window_size)
print(ma)