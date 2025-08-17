import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, binom

class GPDRiskModel:
    def __init__(self, returns: np.ndarray, threshold_quantile: float = 0.95):
        """
        EVT framework for fitting GPD to tail losses.
        :param returns: array of returns (e.g. daily log returns).
        :param threshold_quantile: quantile to set threshold for exceedances (default=95%).
        """
        self.returns = np.asarray(returns)
        self.threshold = np.percentile(-self.returns, threshold_quantile*100)
        self.exceedances = -self.returns[-self.returns > -self.threshold] - self.threshold
        self.shape = None
        self.scale = None

    def fit(self):
        """Fit GPD using MLE (scipy)."""
        shape, loc, scale = genpareto.fit(self.exceedances, floc=0)
        self.shape, self.scale = shape, scale
        return self.shape, self.scale

    def var_cvar(self, p: float = 0.99):
        """
        Compute Value-at-Risk (VaR) and Conditional VaR (CVaR).
        :param p: confidence level (e.g. 0.99 for 99% VaR).
        """
        n_exceed = len(self.exceedances)
        n_total = len(self.returns)
        prob_exceed = n_exceed / n_total

        # VaR from GPD formula
        var = self.threshold + (self.scale / self.shape) * ((( (1 - p) / prob_exceed ) ** (-self.shape)) - 1)

        # CVaR formula (Expected Shortfall)
        if self.shape < 1:
            cvar = (var + (self.scale - self.shape * self.threshold) / (1 - self.shape))
        else:
            cvar = np.inf

        return var, cvar

    def backtest_var(self, var_level: float = 0.99):
        """
        Kupiec unconditional coverage test for VaR.
        :param var_level: confidence level for VaR test.
        """
        var, _ = self.var_cvar(var_level)
        breaches = (-self.returns > var).astype(int)
        n_breaches = breaches.sum()
        n_total = len(breaches)

        p_hat = n_breaches / n_total
        LR_uc = -2 * (
            binom.logpmf(n_breaches, n_total, var_level) -
            binom.logpmf(n_breaches, n_total, p_hat)
        )
        return {
            "VaR": var,
            "Breaches": n_breaches,
            "Expected": (1-var_level)*n_total,
            "LR_stat": LR_uc,
            "Kupiec_pval": 1 - chi2_cdf(LR_uc, 1)
        }

    def plot_fit(self):
        """Plot histogram of exceedances with fitted GPD PDF + QQ plot."""
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))

        x = np.linspace(0, self.exceedances.max(), 200)
        ax[0].hist(self.exceedances, bins=40, density=True, alpha=0.6, label="Exceedances")
        ax[0].plot(x, genpareto.pdf(x, self.shape, loc=0, scale=self.scale), 'r-', lw=2, label="Fitted GPD")
        ax[0].set_title("Exceedances over threshold")
        ax[0].legend()

        emp_quantiles = np.sort(self.exceedances)
        theo_quantiles = genpareto.ppf(np.linspace(0.01, 0.99, len(emp_quantiles)), self.shape, loc=0, scale=self.scale)
        ax[1].scatter(theo_quantiles, emp_quantiles, alpha=0.5)
        ax[1].plot([0, max(emp_quantiles)], [0, max(emp_quantiles)], 'r--')
        ax[1].set_title("QQ-Plot: GPD Fit")
        ax[1].set_xlabel("Theoretical Quantiles")
        ax[1].set_ylabel("Empirical Quantiles")

        plt.tight_layout()
        plt.show()


def chi2_cdf(x, df):
    """Simple chi2 CDF for Kupiec backtest."""
    from scipy.stats import chi2
    return chi2.cdf(x, df)
