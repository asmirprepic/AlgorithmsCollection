# %%
import numpy as np
import matplotlib.pyplot as plt

def simulate_ctmc(rate_matrix, initial_state, num_steps):
    num_states = rate_matrix.shape[0]
    state = initial_state
    times = [0]
    states = [state]
    
    current_time = 0
    for _ in range(num_steps):
        # Get rates for leaving the current state
        current_rates = rate_matrix[state, :]
        total_rate = -rate_matrix[state, state]  # The rate of leaving the state is on the diagonal and negative
        
        # Exponential time until the next transition
        time_to_next = np.random.exponential(1 / total_rate)
        current_time += time_to_next
        times.append(current_time)
        
        # Transition probabilities (non-diagonal elements)
        probabilities = np.maximum(current_rates / total_rate, 0)  # Ensure non-negative probabilities
        probabilities[state] = 0  # No self-transition

        # Normalize to ensure they sum to 1
        probabilities /= probabilities.sum()

        # Choose the next state
        next_state = np.random.choice(np.arange(num_states), p=probabilities)
        
        states.append(next_state)
        state = next_state
    
    return np.array(times), np.array(states)

# Define rate matrix for a simple 3-state model
rate_matrix = np.array([
    [-6, 2, 4],
    [5, -8, 3],
    [2, 1, -3]
])

# Simulation parameters
initial_state = 0
num_steps = 1000

# Simulate the CTMC
times, states = simulate_ctmc(rate_matrix, initial_state, num_steps)

# Plotting
plt.figure(figsize=(10, 4))
plt.step(times, states, where='post')
plt.xlabel('Time')
plt.ylabel('State')
plt.title('Continuous Time Markov Chain - State Transitions')
plt.yticks([0, 1, 2], ["State 0", "State 1", "State 2"])
plt.grid(True)
plt.show()
