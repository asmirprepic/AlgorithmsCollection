import math

class OnlineES:
    """
    Online Expected Shortfall (ES) at level alpha using stochastic approximation.
    Tracks VaR (quantile) via Robbins–Monro and ES via conditional mean below VaR.

    VaR update:  q_{t+1} = q_t + eta_q * (alpha - 1{r_t <= q_t})
    ES  update:  es_{t+1} = es_t + eta_es * ( (r_t * 1{r_t <= q_t})/alpha - es_t )

    Use small fixed etas for adaptation (e.g., 0.005–0.02). For slower drift, reduce.
    """
    def __init__(self, alpha=0.01, q0=0.0, es0=-0.01, eta_q=0.01, eta_es=0.01):
        if not (0 < alpha < 1):
            raise ValueError("alpha must be in (0,1).")
        self.alpha = float(alpha)
        self.q = float(q0)       # alpha (var)
        self.es = float(es0)     # mean below var
        self.eta_q = float(eta_q)
        self.eta_es = float(eta_es)
        self.t = 0

    def update(self, r: float):
        """
        Feed one return r (negative = loss). Returns (VaR, ES).
        """
        self.t += 1
        hit = 1.0 if r <= self.q else 0.0

        #  pinball loss gradient
        self.q += self.eta_q * (self.alpha - hit)

        # target is E[r * 1{r<=VaR}] / alpha
        target = (r * hit) / max(self.alpha, 1e-12)
        self.es += self.eta_es * (target - self.es)

        return self.q, self.es

    def state(self):
        return {"VaR": self.q, "ES": self.es, "alpha": self.alpha, "t": self.t}
