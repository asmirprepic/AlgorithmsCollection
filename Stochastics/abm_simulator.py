import numpy as np

class ABMSimulator:
    def __init__(self, N=100, beta=0.1, gamma=0.05, contacts_per_day=10, initial_infected=1, max_days=100, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.N = N
        self.beta = beta
        self.gamma = gamma
        self.contacts_per_day = contacts_per_day
        self.max_days = max_days

        self.agents = [Agent('S') for _ in range(N)]
        initial_infected_indices = np.random.choice(N, initial_infected, replace=False)
        for idx in initial_infected_indices:
            self.agents[idx].state = 'I'

        self.history = []
