import numpy as np

# Number of states
num_states = 3

# Base transition matrix (structure only, probabilities will be sampled)
base_transition_matrix = np.array([
    [0.0, 1.0, 1.0],  # Transition probabilities will be sampled for state 0
    [1.0, 0.0, 1.0],  # Transition probabilities will be sampled for state 1
    [1.0, 1.0, 0.0]   # Transition probabilities will be sampled for state 2
])

# Define alpha parameters for Dirichlet distribution for each state's transition probabilities
# Higher alpha values mean the distribution is more concentrated around the mean.
alpha_params = np.array([
    [0.0, 5.0, 5.0],  # Alpha parameters for state 0 transitions
    [5.0, 0.0, 5.0],  # Alpha parameters for state 1 transitions
    [5.0, 5.0, 0.0]   # Alpha parameters for state 2 transitions
])

# Mean holding times for each state (exponential distribution parameter)
mean_holding_times = np.array([5.0, 3.0, 4.0])


# Function to sample a transition matrix using Dirichlet distribution
def sample_transition_matrix(alpha_params):
    transition_matrix = np.zeros_like(alpha_params)
    for i in range(num_states):
        if np.sum(alpha_params[i]) > 0:
            transition_matrix[i] = np.random.dirichlet(alpha_params[i])
    return transition_matrix

# Function to sample holding time from an exponential distribution
def sample_holding_time(state):
    return np.random.exponential(mean_holding_times[state])

# Function to simulate the semi-markov process with dynamic transition probabilities
def simulate_dynamic_semi_markov(initial_state, max_time):
    current_state = initial_state
    current_time = 0
    state_sequence = [current_state]
    time_sequence = [current_time]
    
    while current_time < max_time:
        # Sample holding time in the current state
        holding_time = sample_holding_time(current_state)
        current_time += holding_time
        
        if current_time >= max_time:
            break
        
        # Sample new transition matrix for the current step
        transition_matrix = sample_transition_matrix(alpha_params)
        
        # Sample next state based on the dynamic transition matrix
        next_state = np.random.choice(num_states, p=transition_matrix[current_state])
        
        # Update state and record time
        state_sequence.append(next_state)
        time_sequence.append(current_time)
        
        # Move to the next state
        current_state = next_state
    
    return state_sequence, time_sequence

# Simulate the semi-markov process
initial_state = 0
max_time = 50
states, times = simulate_dynamic_semi_markov(initial_state, max_time)

# Display results
for s, t in zip(states, times):
    print(f"State: {s}, Time Entered: {t:.2f}")
