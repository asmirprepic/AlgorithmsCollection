"""
THis is really an optimization problem

"""
import numpy as np

def objective_function(x, y):
    return x**2 + y**2


def random_search(objective, bounds, iterations):
    best_solution = None
    best_evaluation = float('inf')  # We are minimizing; initialize to "infinity"

    for _ in range(iterations):
        # Generate a random solution
        solution = np.random.rand(2) * (bounds[:, 1] - bounds[:, 0]) + bounds[:, 0]
        # Evaluate the random solution
        evaluation = objective(solution[0], solution[1])

        # Check if this is a new best solution
        if evaluation < best_evaluation:
            best_solution, best_evaluation = solution, evaluation
            print("New best: f(%s) = %.5f" % (best_solution, best_evaluation))
    
    return best_solution, best_evaluation

# Define the bounds of the search space
bounds = np.array([[-5.0, 5.0], [-5.0, 5.0]])  # for x and y respectively

# Perform the random search
best_solution, best_evaluation = random_search(objective_function, bounds, 1000)
print("Best Solution: f(%s) = %.5f" % (best_solution, best_evaluation))
Step 3: Running the Code
