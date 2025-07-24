import numpy as np
import matplotlib.pyplot as plt

class InfectionSimulator:
    def __init__(self, N, p, contact_rate, initial_infected=1, seed=None):
        """
        Initialize the simulation parameters.

        Parameters:
            N (int): Total population size
            p (float): Infection probability upon contact (0 ≤ p ≤ 1)
            contact_rate (float): Total rate of contact events (λ)
            initial_infected (int): Number of initially infected individuals
            seed (int): Optional random seed
        """
        assert 0 <= p <= 1, "Infection probability must be between 0 and 1"
        assert 0 < initial_infected <= N, "Initial infected must be between 1 and N"

        self.N = N
        self.p = p
        self.lambda_ = contact_rate
        self.I0 = initial_infected
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

        self.time = [0.0]
        self.infected = [self.I0]

    def _infection_rate(self, i):
        """Compute the infection rate q_{i, i+1} at current state i."""
        if i >= self.N:
            return 0.0
        susceptible = self.N - i
        return self.lambda_ * self.p * (2 * i * susceptible) / (self.N * (self.N - 1))

    def simulate(self, max_time=np.inf):
        """Run the simulation until full infection or max_time."""
        i = self.I0
        t = 0.0

        while i < self.N and t < max_time:
            rate = self._infection_rate(i)
            if rate <= 0:
                break
            dt = np.random.exponential(scale=1.0 / rate)
            t += dt
            i += 1
            self.time.append(t)
            self.infected.append(i)

    def run_once(self, max_time=np.inf):
        self.time = [0.0]
        self.infected = [self.I0]
        i = self.I0
        t = 0.0

        while i < self.N and t < max_time:
            rate = self._infection_rate(i)
            if rate <= 0:
                break
            dt = np.random.exponential(scale=1.0 / rate)
            t += dt
            i += 1
            self.time.append(t)
            self.infected.append(i)

        return t, list(self.infected), list(self.time)

    def run_many_trials(N_trials, **sim_kwargs):
        results = []
        for _ in range(N_trials):
            sim = InfectionSimulator(**sim_kwargs)
            final_time, infected_path, time_path = sim.run_once()
            results.append({
                'final_time': final_time,
                'infected': infected_path,
                'time': time_path
            })
        return results

    def plot(self):
        """Plot the infection curve over time."""
        plt.figure(figsize=(8, 4))
        plt.step(self.time, self.infected, where='post', label='Infected')
        plt.xlabel('Time')
        plt.ylabel('Number of Infected Individuals')
        plt.title('Infection Spread Simulation')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def get_results(self):
        """Return simulation results as arrays."""
        return np.array(self.time), np.array(self.infected)
