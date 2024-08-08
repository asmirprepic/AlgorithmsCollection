import numpy as np

# Define the CDF function (replace this with your actual CDF function)
def cdf_function(x):
    # Example: CDF of the standard normal distribution
    return 0.5 * (1 + np.erf(x / np.sqrt(2)))

# Define the bisection method
def bisection_method(cdf_function, p, lower_bound, upper_bound, tolerance=1e-6, max_iterations=100):
    """
    Find x such that cdf_function(x) = p using the bisection method.
    
    Parameters:
    - cdf_function: The CDF function for which we want to find the inverse.
    - p: The target probability (0 < p < 1).
    - lower_bound: The lower bound of the search interval.
    - upper_bound: The upper bound of the search interval.
    - tolerance: The tolerance for the convergence of the method.
    - max_iterations: The maximum number of iterations to perform.
    
    Returns:
    - The value of x such that cdf_function(x) ≈ p.
    """
    
    # Initial checks
    if cdf_function(lower_bound) > p or cdf_function(upper_bound) < p:
        raise ValueError("The root is not within the given bounds.")
    
    for i in range(max_iterations):
        # Find the midpoint of the current interval
        midpoint = (lower_bound + upper_bound) / 2.0
        
        # Evaluate the CDF function at the midpoint
        midpoint_value = cdf_function(midpoint)
        
        # Check if we've found the root or are within the desired tolerance
        if np.abs(midpoint_value - p) < tolerance:
            return midpoint
        
        # Determine which half of the interval contains the root
        if midpoint_value < p:
            lower_bound = midpoint  # Root is in the upper half
        else:
            upper_bound = midpoint  # Root is in the lower half
        
    # If we reach here, the method did not converge within the maximum number of iterations
    raise ValueError("Bisection method did not converge.")

# Example usage:
p = 0.95  # We want the value x such that CDF(x) = 0.95
lower_bound = -10  # Reasonable lower bound for the search
upper_bound = 10   # Reasonable upper bound for the search

# Find the value corresponding to the 95th percentile
value_at_95th_percentile = bisection_method(cdf_function, p, lower_bound, upper_bound)

print(f"The value at the 95th percentile is approximately: {value_at_95th_percentile:.4f}")
