import numpy as np
import matplotlib.pyplot as plt

class JumpRegimeSwitchSimulator:
    def __init__(self,T,dt,regimes,trans_matrix,initial_state = 0, seed =42):
        """
        Simulate a jump diffusion process with Markov regime switching

        Parameters:
        - T: total time
        - dt: time step
        - regimes: list of dicts, each with keys 'mu', 'sigma', 'lambda', 'jump_mean', 'jump_std'
        - trans_matrix: regime transition matrix (NxN)
        - initial_state: starting regime

        """

        self.T = T
        self.dt = dt
        self.N = len(T/dt)
        self.regimes = regimes
        self.trans_matrix = np.array(trans_matrix)
        self.current_state = initial_state
        self.num_states = len(regimes)
        self.seed = seed
        np.random.seed(seed)

    def simulate(self):
        times = np.linspace(0, self.T, self.N + 1)
        prices = np.zeros(self.N + 1)
        states = np.zeros(self.N + 1, dtype=int)
        prices[0] = 100
        states[0] = self.current_state

        for i in range(1, self.N + 1):
            regime = self.regimes[self.current_state]
            dW = np.random.normal(0, np.sqrt(self.dt))
            J = 0
            if np.random.rand() < regime['lambda'] * self.dt:
                J = np.random.normal(regime['jump_mean'], regime['jump_std'])
            dS = (regime['mu'] - 0.5 * regime['sigma'] ** 2) * self.dt + regime['sigma'] * dW + J
            prices[i] = prices[i - 1] * np.exp(dS)

            # Regime switching
            self.current_state = np.random.choice(self.num_states, p=self.trans_matrix[self.current_state])
            states[i] = self.current_state

        return times, prices, states
