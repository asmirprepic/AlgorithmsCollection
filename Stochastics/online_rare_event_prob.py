import math

class OnlineRareEventBeta:
    """
    Online estimator for rare-event probability with discounting.
    Event_t ∈ {0,1}. Posterior is Beta(a,b); discount γ∈(0,1] adapts to drift.
    """
    def __init__(self, a0=1.0, b0=1.0, discount=1.0):
        self.a = float(a0); self.b = float(b0)
        self.gamma = float(discount)  # e.g., 0.99 for slow forgetting

    def update(self, event: int):
        # Discount (forget) old evidence, then add new observation
        self.a = self.gamma * self.a + int(event)
        self.b = self.gamma * self.b + (1 - int(event))

    @property
    def mean(self) -> float:
        return self.a / (self.a + self.b)

    def ci(self, conf=0.95):
        from scipy.stats import beta
        lo = beta.ppf((1-conf)/2, self.a, self.b)
        hi = beta.ppf(1-(1-conf)/2, self.a, self.b)
        return float(lo), float(hi)

    @property
    def expected_wait(self) -> float:
        """Expected time between events ≈ 1 / E[p]."""
        m = self.mean
        return math.inf if m <= 0 else 1.0/m
