import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy.stats import genpareto, chi2
from scipy.optimize import minimize


@dataclass
class GPDParams:
    xi: float
    beta: float
    u: float
    pi_u: float


def _fit_gpd_exceedances(losses: np.ndarray, threshold_q: float) -> Optional[GPDParams]:
    """
    Fit GPD to exceedances above threshold u (POT).
    Returns None if insufficient exceedances.
    """
    if losses.ndim != 1:
        losses = losses.ravel()
    N = losses.size
    u = np.quantile(losses, threshold_q)
    excess = losses[losses > u] - u
    n_u = excess.size
    if n_u < 30:
        return None

    xi, loc, beta = genpareto.fit(excess, floc=0.0)
    pi_u = n_u / N
    return GPDParams(xi=xi, beta=beta, u=u, pi_u=pi_u)


def _var_from_gpd(p: float, params: GPDParams) -> float:
    """
    One-step-ahead VaR at level p using POT approximation.
    For xi -> 0, use exponential limit.
    """
    q = 1.0 - p
    xi, beta, u, pi_u = params.xi, params.beta, params.u, params.pi_u
    eps = 1e-10
    q = max(q, eps)
    pi_u = max(pi_u, eps)
    if abs(xi) < 1e-6:
        return float(u + beta * np.log(pi_u / q))
    return float(u + (beta / xi) * ((q / pi_u) ** (-xi) - 1.0))


def _cvar_from_gpd(var_p: float, params: GPDParams) -> float:
    """
    Expected Shortfall (CVaR/ES) for xi < 1:
    ES_p = (VaR_p + (beta - xi * u)) / (1 - xi)
    Else infinite.
    For xi -> 0, ES_p - VaR_p -> beta (exponential limit).
    """
    xi, beta, u = params.xi, params.beta, params.u
    if xi >= 1.0:
        return float(np.inf)
    denom = 1.0 - xi
    return float((var_p + (beta - xi * u)) / denom)


def kupiec_test(breaches: np.ndarray, p: float) -> Dict[str, float]:
    """
    Kupiec Unconditional Coverage test.
    H0: breach probability = 1 - p (correct coverage).
    """
    b = int(breaches.sum())
    n = int(breaches.size)
    if n == 0:
        return {"LR_uc": np.nan, "p_value": np.nan, "breaches": b, "expected": (1 - p) * n}
    alpha = 1 - p
    phat = max(b / n, 1e-12)
    ll_h0 = b * np.log(alpha + 1e-300) + (n - b) * np.log(1 - alpha + 1e-300)
    ll_hat = b * np.log(phat) + (n - b) * np.log(1 - phat)
    LR_uc = -2.0 * (ll_h0 - ll_hat)
    return {"LR_uc": LR_uc, "p_value": 1 - chi2.cdf(LR_uc, df=1), "breaches": b, "expected": alpha * n}

def mean_residual_life(losses, qs=np.linspace(0.80, 0.99, 40)):
    u = np.quantile(losses, qs)
    mrl = []
    for ui in u:
        exc = losses[losses > ui] - ui
        mrl.append(exc.mean() if exc.size > 0 else np.nan)
    return u, np.array(mrl)

def fit_gpd_regression(excess, Z):
    # excess: (n,), Z: (n,k) covariates (e.g., scaled RV)
    def nll(theta):
        xi = theta[0]
        logbeta = Z @ theta[1:]
        beta = np.exp(logbeta)
        y = excess / beta
        if np.any(1 + xi * y <= 0):
            return np.inf
        n = excess.size
        return n * np.log(beta).mean() + (1 + 1/xi) * np.log1p(xi * y).sum()
    k = Z.shape[1]; theta0 = np.r_[0.1, np.zeros(k)]
    res = minimize(nll, theta0, method="L-BFGS-B", bounds=[(-0.45, 0.95)] + [(-5,5)]*k)
    return res.x, res.success

class RollingGPDRiskMonitor:
    """
    Rolling EVT (POT/GPD) risk monitor producing OOS VaR/CVaR and backtests.
    """

    def __init__(
        self,
        returns: np.ndarray,
        window: int = 1000,
        threshold_q: float = 0.95,
        p: float = 0.99,
        min_exceed: int = 30
    ):
        """
        :param returns: array of returns (e.g., daily). Losses = -returns.
        :param window: rolling window length (observations).
        :param threshold_q: in-window quantile for POT threshold u.
        :param p: VaR/CVaR confidence level.
        :param min_exceed: min exceedances to fit GPD.
        """
        self.r = np.asarray(returns).astype(float)
        self.window = int(window)
        self.threshold_q = float(threshold_q)
        self.p = float(p)
        self.min_exceed = int(min_exceed)

    def run(self) -> pd.DataFrame:
        """
        Refit GPD each step on [t-window, t) and forecast VaR/CVaR for t.
        Returns DataFrame with VaR, CVaR, realized loss, and breach flag.
        """
        n = self.r.size
        losses = -self.r  # positive = loss
        var = np.full(n, np.nan, dtype=float)
        cvar = np.full(n, np.nan, dtype=float)

        for t in range(self.window, n):
            win_losses = losses[t - self.window:t]
            params = _fit_gpd_exceedances(win_losses, self.threshold_q)
            if params is None or params.pi_u * self.window < self.min_exceed:
                continue
            v = _var_from_gpd(self.p, params)
            var[t] = v
            cvar[t] = _cvar_from_gpd(v, params)

        df = pd.DataFrame({
            "loss": losses,
            "VaR": var,
            "CVaR": cvar
        })
        df["breach"] = (df["loss"] > df["VaR"]).astype(float)
        return df

    @staticmethod
    def backtests(df: pd.DataFrame, p: float) -> Dict[str, float]:
        """
        Run Kupiec (coverage) and Christoffersen (independence) tests on OOS breaches.
        NaNs in VaR rows are ignored.
        """
        mask = np.isfinite(df["VaR"].values)
        breaches = df.loc[mask, "breach"].values
        k = kupiec_test(breaches, p)
        c = christoffersen_independence_test(breaches)
        # Combined conditional coverage:
        if np.isfinite(k.get("LR_uc", np.nan)) and np.isfinite(c.get("LR_ind", np.nan)):
            LR_cc = k["LR_uc"] + c["LR_ind"]
            p_cc = 1 - chi2.cdf(LR_cc, df=2)
        else:
            LR_cc, p_cc = np.nan, np.nan
        out = {
            "breaches": k["breaches"],
            "expected_breaches": k["expected"],
            "kupiec_LR": k["LR_uc"],
            "kupiec_p": k["p_value"],
            "christoffersen_LR": c["LR_ind"],
            "christoffersen_p": c["p_value"],
            "conditional_coverage_LR": LR_cc,
            "conditional_coverage_p": p_cc
        }
        return out

    def tail_pit_test(excess, xi, beta):
        U = (1 + xi * (excess / beta))**(-1/xi) if abs(xi) >= 1e-6 else np.exp(-(excess / beta))
        stat, pval = kstest(U, 'uniform')
        return {"KS_stat": stat, "p_value": pval}

    def stationary_bootstrap(x, p=0.1, size=None, rng=None):
        rng = np.random.default_rng(rng)
        n = len(x); size = size or n
        idx = np.empty(size, dtype=int); i = 0
        while i < size:
            start = rng.integers(0, n); L = 1 + rng.geometric(p)
            L = min(L, size - i)
            idx[i:i+L] = (start + np.arange(L)) % n
            i += L
        return x[idx]

    def runs_declustering(losses, u, r=5):
        exc_idx = np.where(losses > u)[0]
        clusters, cur = [], [exc_idx[0]] if exc_idx.size else []
        for i in range(1, exc_idx.size):
            if exc_idx[i] - exc_idx[i-1] <= r:
                cur.append(exc_idx[i])
            else:
                clusters.append(cur); cur = [exc_idx[i]]
        if cur: clusters.append(cur)
        repr_idx = [c[np.argmax(losses[c])] for c in clusters]  # peak per cluster
        theta = len(clusters) / max(exc_idx.size, 1)
        return np.array(repr_idx), theta
    def mean_residual_life(losses, qs=np.linspace(0.80, 0.99, 40)):
        u = np.quantile(losses, qs)
        mrl = []
        for ui in u:
            exc = losses[losses > ui] - ui
            mrl.append(exc.mean() if exc.size > 0 else np.nan)
        return u, np.array(mrl)
