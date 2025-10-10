import math

class BOCPDNormalGamma:
    """
    Bayesian Online Change-Point Detection (Adams-MacKay style) with
    conjugate Normal–Gamma model (unknown mean & variance), constant hazard h.

    Posterior hyperparams (m,k,a,b):
      μ | τ ~ N(m, (k τ)^-1),   τ ~ Gamma(a,b)   (τ = 1/σ^2)

    Predictive: Student-t with df=2a, loc=m, scale = sqrt(b*(k+1)/(a*k)).
    After x:
      k' = k+1
      m' = (k m + x)/k'
      a' = a + 1/2
      b' = b + 0.5 * k (x - m)^2 / k'
    We track only two weights: P(change) and P(continue) (fast “lite” version).
    """
    def __init__(self, hazard=1/250, m0=0.0, k0=1e-3, a0=2.0, b0=1.0):
        self.h = float(hazard)
        self.m, self.k, self.a, self.b = float(m0), float(k0), float(a0), float(b0)
        self.p_run = 1.0
        self.t = 0
        self.p_change = 0.0

    @staticmethod
    def _log_t_pdf(x, df, loc, scale):
        v = df; z = (x - loc) / scale
        return (math.lgamma((v+1)/2) - math.lgamma(v/2)
                - 0.5*math.log(v*math.pi) - math.log(scale)
                - ((v+1)/2)*math.log(1 + z*z/v))

    def update(self, x: float):
        self.t += 1
        df = 2.0 * self.a
        scale = math.sqrt(self.b * (self.k + 1.0) / (self.a * self.k))
        log_pred = self._log_t_pdf(x, df=df, loc=self.m, scale=scale)

        # Run-length recursion (collapsed): mass that continues vs. change now
        # Unnormalized:
        cont = (1 - self.h) * self.p_run * math.exp(log_pred)
        chg  = self.h * self.p_run * 1.0  # new run uses prior predictive ~ integrates to 1

        z = cont + chg
        self.p_run = cont / (z + 1e-300)
        self.p_change = chg / (z + 1e-300)

        # Update posterior *as if* we continued; if a change is likely, the next step will reset
        k_new = self.k + 1.0
        m_new = (self.k * self.m + x) / k_new
        a_new = self.a + 0.5
        b_new = self.b + 0.5 * (self.k * (x - self.m)**2) / k_new

        # Soft reset: convex combo between "restart from prior at x" and "continue"

        if self.p_change > 0.5:
            self.m, self.k, self.a, self.b = x, 1.0, 2.0, 1.0
        else:
            self.m, self.k, self.a, self.b = m_new, k_new, a_new, b_new

        return {"t": self.t, "p_change": self.p_change, "run_prob": self.p_run,
                "pred_df": df, "pred_loc": self.m, "pred_scale": scale}
