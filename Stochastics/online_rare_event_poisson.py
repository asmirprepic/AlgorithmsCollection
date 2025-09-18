class OnlineRareEventPoisson:
    """
    Online estimator for rare-event rate λ with discounting.
    Observations: count k over exposure t (time/volume). Prior: λ ~ Gamma(α, β).
    Posterior after (k, t): α←γ α + k, β←γ β + t.
    """
    def __init__(self, alpha0=0.1, beta0=100.0, discount=1.0):
        self.alpha = float(alpha0); self.beta = float(beta0)
        self.gamma = float(discount)

    def update(self, k: int, exposure: float = 1.0):
        self.alpha = self.gamma * self.alpha + int(k)
        self.beta  = self.gamma * self.beta  + float(exposure)

    @property
    def rate_mean(self) -> float:
        """E[λ] events per unit exposure."""
        return self.alpha / self.beta

    def ci(self, conf=0.95):
        from scipy.stats import gamma
        lo = gamma.ppf((1-conf)/2, a=self.alpha, scale=1/self.beta)
        hi = gamma.ppf(1-(1-conf)/2, a=self.alpha, scale=1/self.beta)
        return float(lo), float(hi)

    def prob_at_least_one(self, horizon: float) -> float:
        """P(N≥1 in future horizon) = 1 - E[e^{-λh}] ≈ 1 - (β/(β+h))^α (Gamma mixture)."""
        h = max(horizon, 0.0)
        return 1.0 - (self.beta / (self.beta + h)) ** self.alpha
