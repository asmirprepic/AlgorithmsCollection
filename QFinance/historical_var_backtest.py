import numpy as np
import pandas as pd
from scipy.stats import binom, chi2
import matplotlib.pyplot as plt

class HistoricalVARBacktester:
    def __init__(self,returns,var_level = 0.01,window = 252):
        self.returns = np.asarray(returns)
        self.var_level = var_level
        self.window = window
        self.var_series = []
        self.breaches = []

    def compute_var_series(self):
        n = len(self.breaches)
        x = self.breaches.sum()
        p = self.var_level
        likelihood_ratio = -2 * (
            x * np.log(p/(x/n)) + (n-x) * np.log((1-p) / (1-x/n))
        )

        p_value = 1 - chi2.cdf(likelihood_ratio,df = 1)
        return {
            "failures": x,
            "expected": round(n*p,2),
            "p_value": p_value,
            "passed": p_value > 0.05
        }

    def christofferson_test(self):
        b = self.breaches.astype(int)
        n00 = n01 = n10 = n11 = 0

        for i in range(1, len(b)):
            prev, curr = b[i-1],b[i]
            if prev == 0 and curr == 0:
                n00 += 1
            elif prev == 0 and curr == 1:
                n01 += 1
            elif prev == 1 and curr == 0:
                n10 += 1
            elif prev == 1 and curr == 1:
                n11 += 1
        pi_0 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
        pi_1 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
        pi = (n01 + n11) / (n00 + n01 + n10 + n11)

        # LR
        def ll(p,n1,n0):
            return n1 *np.log(p) + n0 * np.log(1-p) if p > 0 and p < 1 else 0

        lr_indep = -2 * (
            ll(pi, n01 + n11, n00 + n10) -
            (ll(pi_0, n01, n00) + ll(pi_1, n11, n10))
        )

        p_value = 1 - chi2.cdf(lr_indep, df=1)
        return {
            "p_value": p_value,
            "passed": p_value > 0.05,
            "transitions": {"00": n00, "01": n01, "10": n10, "11": n11}
        }

        # TODO : Implement run
