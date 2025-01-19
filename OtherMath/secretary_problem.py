import random

def secretary_problem(candidates):
    """
    Solves the Secretary Problem using the optimal stopping rule.

    Args:
        candidates: A list of candidate ranks (1 is best, higher is worse).

    Returns:
        The selected candidate rank and its index.
    """
    n = len(candidates)
    
    threshold = int(n / 2.718)
    
    
    best_observed = max(candidates[:threshold])
    
    
    for i in range(threshold, n):
        if candidates[i] < best_observed:  
            return candidates[i], i
    
    
    return candidates[-1], n - 1
