import numpy as np

def objective_function(x):
    return np.sin(x[0]) + np.cos(x[1])

def random_neighbor(x, scale=0.1):
    return x + np.random.uniform(-scale, scale, size=x.shape)

def simulated_annealing(objective_function, initial_state, temp, cooling_rate, iterations):
    state = initial_state
    current_temp = temp
    best_state = state
    best_obj = objective_function(state)
    for i in range(iterations):
        candidate = random_neighbor(state)
        candidate_obj = objective_function(candidate)
        delta = candidate_obj - objective_function(state)
        if delta > 0 or np.random.rand() < np.exp(delta / current_temp):
            state = candidate
        if candidate_obj > best_obj:
            best_state = candidate
            best_obj = candidate_obj
        current_temp *= cooling_rate
    return best_state, best_obj

initial_state = np.array([0.0, 0.0])
temp = 10
cooling_rate = 0.99
iterations = 1000

best_state, best_obj = simulated_annealing(objective_function, initial_state, temp, cooling_rate, iterations)
print("Best state:", best_state)
print("Best objective value:", best_obj)
