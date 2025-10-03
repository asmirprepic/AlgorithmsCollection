import math

class SelfNormMeanEProcess:
    """
    Anytime-valid test of H0: E[X]=0 with unknown variance.
    Maintains an e-process based on self-normalized sums (mixture over t).
    If E_t grows large, reject; p_anytime = 1 / sup_s E_s.

    Update needs x_t clipped to [-M, M] for boundedness (choose M as a robust scale).
    """
    def __init__(self, M: float = 3.0):
        self.M = float(M)
        self.n = 0
        self.S = 0.0         # sum x
        self.Q = 0.0         # sum x^2
        self.E = 1.0
        self.E_sup = 1.0     # running sup

    def update(self, x: float):
        x = max(min(float(x), self.M), -self.M)
        self.n += 1
        self.S += x
        self.Q += x * x
        # E_t = exp( S^2 / (2(Q + c)) - 0.5 * log(1 + Q/c) ), choose c = M^2
        c = self.M * self.M
        num = (self.S * self.S) / (2.0 * (self.Q + c))
        den = 0.5 * math.log(1.0 + self.Q / c)
        self.E = math.exp(num - den)
        self.E_sup = max(self.E_sup, self.E)
        p_anytime = 1.0 / self.E_sup
        return {"n": self.n, "E": self.E, "p_anytime": p_anytime, "S": self.S, "Q": self.Q}
