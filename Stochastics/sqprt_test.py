import numpy as np
from scipy.stats import binom

def sequential_binomial_test(p0, alpha=0.05, beta=0.1):
    """
    Perform a sequential binomial test testing.
    
    Parameters:
    - p0: probality under H0
    - alpha: significance level (type I error rate)
    - beta: power (type II error rate)
    
    Returns:
    - Function to process sequential observations.
    """
    # Define the thresholds
    lower_bound = beta / (1 - alpha)
    upper_bound = (1 - beta) / alpha

    successes = 0
    failures = 0
    
    def update(observation):
        """
        Process a new observation in the sequential test.
        Parameters:
        - observation: binary outcome (1 for success, 0 for failure)
        
        Returns:
        - None if the test continues
        - Decision (accept/reject) if the test concludes
        """
        nonlocal successes, failures
        
        # Update success and failure counts
        if observation == 1:
            successes += 1
        else:
            failures += 1
        
        # Calculate likelihood ratio (LRT) for the null hypothesis p0
        total_trials = successes + failures
        likelihood_ratio = ((p0 / (1 - p0)) ** successes) * (((1 - p0) / p0) ** failures)
        
        # Check the stopping criteria
        if likelihood_ratio <= lower_bound:
            return "Reject H0: Evidence for alternative hypothesis"
        elif likelihood_ratio >= upper_bound:
            return "Accept H0: Not enough evidence to reject the null hypothesis"
        else:
            return None  # Continue gathering data

    return update

