import math

class PoissonSPRT:
    """
    Sequential test for Poisson rate change.
    H0: λ = lambda0   vs   H1: λ = lambda1  (e.g., spike in errors or fills)
    """
    def __init__(self, lambda0: float, lambda1: float, alpha=0.05, beta=0.10):
        assert lambda0>0 and lambda1>0 and lambda0!=lambda1
        self.l0, self.l1 = float(lambda0), float(lambda1)
        self.A = math.log((1 - beta) / alpha)
        self.B = math.log(beta / (1 - alpha))
        self.llr = 0.0; self.t = 0.0; self.k = 0  # time, count
        self.decision = None

    def update(self, x: int, dt: float = 1.0):
        """
        Observe x events over interval dt (can be 1 bar or arbitrary seconds).
        """
        if self.decision: return self.decision
        self.k += int(x); self.t += float(dt)
        # LLR for Poisson with observation K~Pois(λ t):
        # log Λ = K*log(l1/l0) - t*(l1 - l0)
        self.llr = self.k * math.log(self.l1 / self.l0) - self.t * (self.l1 - self.l0)
        if self.llr >= self.A: self.decision = "H1"
        elif self.llr <= self.B: self.decision = "H0"
        return self.decision
