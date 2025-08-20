import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy.stats import genpareto, chi2


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
