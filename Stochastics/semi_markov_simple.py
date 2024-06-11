import numpy as np
import matplotlib.pyplot as plt

# Define states
states = ['AAA', 'BBB', 'Default']

# Transition probabilities matrix
transition_probabilities = {
    'AAA': {'AAA': 0.7, 'BBB': 0.2, 'Default': 0.1},
    'BBB': {'AAA': 0.1, 'BBB': 0.6, 'Default': 0.3},
    'Default': {'AAA': 0.0, 'BBB': 0.0, 'Default': 1.0}
}

# Time distributions (mean times spent in each state)
state_times = {
    'AAA': 3,
    'BBB': 2,
    'Default': 5
}

# Simulation parameters
num_simulations = 1000
max_steps = 50

# Function to simulate one path
def simulate_path(initial_state):
    state = initial_state
    path = [state]
    times = [0]

    for step in range(max_steps):
        current_time = np.random.exponential(state_times[state])
        times.append(times[-1] + current_time)

        if state == 'Default':
            break

        next_state = np.random.choice(
            states, 
            p=[transition_probabilities[state][s] for s in states]
        )
        path.append(next_state)
        state = next_state

    return path, times[:len(path)]

# Run simulations
paths = []
times = []
for _ in range(num_simulations):
    path, time = simulate_path('AAA')
    paths.append(path)
    times.append(time)

# Plot example paths
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.step(times[i], paths[i], where='post', label=f'Path {i+1}')
plt.title('Semi-Markov Model Simulation Paths')
plt.xlabel('Time')
plt.ylabel('State')
plt.legend()
plt.grid(True)
plt.show()
