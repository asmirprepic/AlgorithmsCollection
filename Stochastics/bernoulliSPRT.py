import math

class BernoulliSPRT:
    """
    Sequential test for a change in success probability.
    H0: p = p0    vs.    H1: p = p1  (p1 > p0 for 'improvement' test)
    Controls Type I/II errors via alpha, beta.
    """
    def __init__(self, p0: float, p1: float, alpha: float = 0.05, beta: float = 0.10):
        assert 0 < p0 < 1 and 0 < p1 < 1 and p0 != p1
        self.p0, self.p1 = float(p0), float(p1)
        self.A = math.log((1 - beta) / alpha)   # upper threshold
        self.B = math.log(beta / (1 - alpha))   # lower threshold
        self.llr = 0.0                          # cumulative log-likelihood ratio
        self.n = 0
        self.decision = None  # "H1", "H0", or None (continue)

    def update(self, x: int):
        """
        Feed one Bernoulli outcome x in {0,1}. Returns current decision.
        """
        if self.decision is not None:
            return self.decision
        # log Λ_n += log f1(x)/f0(x) = x*log(p1/p0) + (1-x)*log((1-p1)/(1-p0))
        self.llr += x * math.log(self.p1 / self.p0) + (1 - x) * math.log((1 - self.p1) / (1 - self.p0))
        self.n += 1
        if self.llr >= self.A:
            self.decision = "H1"
        elif self.llr <= self.B:
            self.decision = "H0"
        return self.decision

    def reset(self):
        self.llr = 0.0
        self.n = 0
        self.decision = None
