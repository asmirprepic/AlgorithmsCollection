import numpy as np

class ThompsonBandit:
    """
    K-armed Bernoulli Thompson Sampling.
    Maintains Beta(a_i, b_i) posteriors for each arm's success prob p_i.
    select() samples p_i ~ Beta and picks argmax; update(i, x) feeds outcome x∈{0,1}.
    """
    def __init__(self, K: int, a0: float = 1.0, b0: float = 1.0, seed: int = 0):
        self.K = int(K)
        self.a = np.full(K, float(a0))
        self.b = np.full(K, float(b0))
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(K, dtype=int)

    def select(self) -> int:
        """Sample p_i ~ Beta(a_i, b_i) and pick the arm with largest draw."""
        theta = self.rng.beta(self.a, self.b)
        return int(np.argmax(theta))

    def update(self, arm: int, reward: int):
        """Update Beta posterior with one Bernoulli reward x∈{0,1} for the chosen arm."""
        arm = int(arm); x = int(reward)
        self.a[arm] += x
        self.b[arm] += 1 - x
        self.counts[arm] += 1

    def post_means(self):
        """Posterior mean success rates for each arm."""
        return self.a / (self.a + self.b)

    def best_arm(self):
        """Current best by posterior mean."""
        return int(np.argmax(self.post_means()))
