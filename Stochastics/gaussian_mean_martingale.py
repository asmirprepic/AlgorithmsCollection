import math

class GaussianMeanMartingale:
    """
    Anytime-valid test for mean drift in x_t ~ N(mu, sigma^2) with KNOWN sigma.
    Null: H0: mu = 0.  Alt: mu ~ N(0, tau^2) (two-sided).
    Bayes factor after n points with sample mean m_n:
        BF_n = sqrt( sigma^2 / (sigma^2 + n*tau^2) ) * exp( n*tau^2*m_n^2 / (2*(sigma^2 + n*tau^2)) )
    BF_n is a nonnegative supermartingale under H0  =>  p_anytime = 1 / sup_{s<=n} BF_s.

    Args:
        sigma : known std deviation of x_t (use EW estimate if unknown).
        tau   : prior scale for alternative; larger tau = more tolerant to big drifts (e.g., ~sigma).
    """
    def __init__(self, sigma: float, tau: float = None):
        self.sigma = float(sigma)
        self.tau = float(tau if tau is not None else self.sigma)  # default: tau≈sigma
        self.n = 0
        self.sumx = 0.0
        self.BF = 1.0
        self.BF_sup = 1.0

    def update(self, x: float):
        """Feed one observation x. Returns dict with BF_n and anytime p-value."""
        self.n += 1
        self.sumx += float(x)
        m = self.sumx / self.n
        s2 = self.sigma ** 2
        t2 = self.tau ** 2
        denom = s2 + self.n * t2
        # Bayes factor vs mu=0 under N(0,tau^2) prior (two-sided)
        self.BF = math.sqrt(s2 / denom) * math.exp( (self.n * t2 * m * m) / (2 * denom) )
        self.BF_sup = max(self.BF_sup, self.BF)
        p_anytime = 1.0 / self.BF_sup
        return {"n": self.n, "mean": m, "BF": self.BF, "p_anytime": p_anytime}

    def reset(self):
        self.n = 0; self.sumx = 0.0; self.BF = 1.0; self.BF_sup = 1.0
