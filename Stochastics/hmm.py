# %%

import numpy as np

class HiddenMarkovModel:
    def __init__(self, states, observations, start_prob, trans_prob, emis_prob):
        self.states = states
        self.observations = observations
        self.start_prob = np.array(start_prob)
        self.trans_prob = np.array(trans_prob)
        self.emis_prob = np.array(emis_prob)

    def forward_algorithm(self, observed_seq):
        N = len(self.states)
        T = len(observed_seq)
        alpha = np.zeros((T, N))

        # Initialize base cases (t == 0)
        alpha[0, :] = self.start_prob * self.emis_prob[:, observed_seq[0]]

        # Recursive case
        for t in range(1, T):
            for j in range(N):
                alpha[t, j] = self.emis_prob[j, observed_seq[t]] * np.sum(alpha[t-1, :] * self.trans_prob[:, j])
        
        return alpha, np.sum(alpha[T-1, :])

    def decode(self, observed_seq):
        N = len(self.states)
        T = len(observed_seq)
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)

        # Initialization
        delta[0, :] = self.start_prob * self.emis_prob[:, observed_seq[0]]

        # Recursion
        for t in range(1, T):
            for j in range(N):
                delta[t, j] = np.max(delta[t-1, :] * self.trans_prob[:, j]) * self.emis_prob[j, observed_seq[t]]
                psi[t, j] = np.argmax(delta[t-1, :] * self.trans_prob[:, j])

        # Backtracking
        states_seq = np.zeros(T, dtype=int)
        states_seq[T-1] = np.argmax(delta[T-1, :])
        for t in range(T-2, -1, -1):
            states_seq[t] = psi[t+1, states_seq[t+1]]

        return states_seq

# Example initialization
states = ['Rainy', 'Sunny']
observations = ['walk', 'shop', 'clean']
start_prob = [0.6, 0.4]
trans_prob = [[0.7, 0.3], [0.4, 0.6]]
emis_prob = [[0.1, 0.4, 0.5], [0.6, 0.3, 0.1]]

model = HiddenMarkovModel(states, observations, start_prob, trans_prob, emis_prob)
