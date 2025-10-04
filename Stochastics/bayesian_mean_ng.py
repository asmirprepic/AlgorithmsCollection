import numpy as np
from math import sqrt
from scipy.stats import t as student_t

class BayesianMeanNG:
    """
    Conjugate Normal–Gamma updater for unknown mean/variance.
    Prior:  μ | τ ~ N(m0, (k0 τ)^-1),  τ ~ Gamma(a0, b0)  where τ=1/σ^2.
    After each x, update (m, k, a, b).  Posterior for μ is a Student t.
    """
    def __init__(self, m0=0.0, k0=1e-3, a0=2.0, b0=1.0):
        self.m, self.k, self.a, self.b = float(m0), float(k0), float(a0), float(b0)
        self.n = 0

    def update(self, x: float):
        x = float(x); self.n += 1
        k_new = self.k + 1.0
        m_new = (self.k * self.m + x) / k_new
        a_new = self.a + 0.5
        b_new = self.b + 0.5 * (self.k * (x - self.m)**2) / k_new
        self.m, self.k, self.a, self.b = m_new, k_new, a_new, b_new
        return self.posterior_summary()

    def posterior_summary(self, conf=0.95):
        # μ | data ~ Student-t(df=2a, loc=m, scale = sqrt(b/(a k)))
        df = 2*self.a
        scale = sqrt(self.b / (self.a * self.k))
        lo, hi = student_t.ppf([(1-conf)/2, 1-(1-conf)/2], df, loc=self.m, scale=scale)
        p_mu_gt_0 = 1 - student_t.cdf(0.0, df, loc=self.m, scale=scale)
        return {"n": self.n, "mu_mean": self.m, "ci": (float(lo), float(hi)), "p(mu>0)": float(p_mu_gt_0)}
