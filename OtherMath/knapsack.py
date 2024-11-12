import numpy as np

def knapsack(values: np.ndarray, weights:np.ndarray, capacity:int):
    n = len(values)
    
    # Create a (n+1) x (capacity+1) matrix to store the maximum values
    dp = np.zeros((n + 1, capacity + 1), dtype=int)
    
    # Fill the DP table
    for i in range(1, n + 1):
        for w in range(1, capacity + 1):
            if weights[i - 1] <= w:
                dp[i, w] = max(dp[i - 1, w], dp[i - 1, w - weights[i - 1]] + values[i - 1])
            else:
                dp[i, w] = dp[i - 1, w]
    
    # The value in the bottom-right corner is the maximum value
    return dp[n, capacity]

# Test the function
values = [60, 100, 120]
weights = [10, 20, 30]
capacity = 50
max_value = knapsack(values, weights, capacity)
print(f"The maximum value that can be achieved with a capacity of {capacity} is {max_value}.")
