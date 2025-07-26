import numpy as np
import matplotlib.pyplot as plt
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
    def step(self):
        new_infections = []
        new_recoveries = []

        for i, agent in enumerate(self.agents):
            if agent.state == 'I':
                agent.days_infected += 1

                # Infect others
                contacts = np.random.choice(self.agents, self.contacts_per_day, replace=True)
                for other in contacts:
                    if other.state == 'S' and np.random.rand() < self.beta:
                        new_infections.append(other)

                # Recover?
                if np.random.rand() < self.gamma:
                    new_recoveries.append(agent)

        for agent in new_infections:
            agent.state = 'I'
            agent.days_infected = 0
        for agent in new_recoveries:
            agent.state = 'R'

    def run(self):
        for day in range(self.max_days):
            S = sum(agent.state == 'S' for agent in self.agents)
            I = sum(agent.state == 'I' for agent in self.agents)
            R = sum(agent.state == 'R' for agent in self.agents)
            self.history.append((S, I, R))
            if I == 0:
                break
            self.step()

    def plot(self):
        data = np.array(self.history)
        plt.plot(data[:, 0], label='Susceptible')
        plt.plot(data[:, 1], label='Infected')
        plt.plot(data[:, 2], label='Recovered')
        plt.xlabel('Day')
        plt.ylabel('Number of People')
        plt.title('Agent-Based SIR Simulation')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
