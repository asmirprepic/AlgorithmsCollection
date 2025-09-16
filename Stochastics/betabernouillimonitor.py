import math

class BetaBernoulliMonitor:
    """
    Online Beta-Bernoulli tracker with CI alarms for p (hit rate).
    Start with Beta(a,b) prior; update per success x∈{0,1}.
    """
    def __init__(self, a=1.0, b=1.0, target=0.55, conf=0.95):
        self.a, self.b = float(a), float(b)
        self.target = float(target)
        self.conf = float(conf)

    def update(self, x: int):
        self.a += int(x); self.b += 1 - int(x)
        # compute one-sided lower credible bound for p at level conf
        from scipy.stats import beta
        lower = beta.ppf((1 - self.conf), self.a, self.b)
        upper = beta.ppf(self.conf, self.a, self.b)
        alarm_low = upper < self.target      #  worse than target
        alarm_high = lower > self.target     #  better than target
        p_mean = self.a / (self.a + self.b)
        return {"p_mean": p_mean, "ci": (lower, upper),
                "alarm_below": alarm_low, "alarm_above": alarm_high}
