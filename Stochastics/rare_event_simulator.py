from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

@dataclass(slots = True)
class RareEventEstimate:
    method: str
    probability_estimate: float
    standard_error: float
    relative_error: float
    n_paths: int
    event_count: int
    details: str
@dataclass(slots=True)
class WeightDiagnostics:
    ess: float
    normalized_weight_entropy: float
    max_normalized_weight: float
    mean_weight: float
    std_weight: float


def _safe_relative_error(est: float, se: float) -> float:
    if abs(est) < 1e-16:
        return float("inf")
    return abs(se / est)


def _weight_diagnostics(weights: np.ndarray) -> WeightDiagnostics:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0.0):
        raise ValueError("Weights must be non-negative.")

    sw = np.sum(w)
    if sw <= 0.0:
        raise ValueError("Sum of weights must be positive.")

    wn = w / sw
    ess = float((sw * sw) / np.sum(w * w))

    # normalized entropy in [0,1]
    n = len(w)
    entropy = -np.sum(np.where(wn > 0.0, wn * np.log(wn), 0.0))
    entropy_norm = float(entropy / math.log(n)) if n > 1 else 1.0

    return WeightDiagnostics(
        ess=ess,
        normalized_weight_entropy=entropy_norm,
        max_normalized_weight=float(np.max(wn)),
        mean_weight=float(np.mean(w)),
        std_weight=float(np.std(w, ddof=1)) if len(w) > 1 else 0.0,
    )

def simulate_terminal_gbm_naive(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Exact terminal GBM simulation under the original measure:
        S_T = S_0 * exp((mu - 0.5 sigma^2) T + sigma sqrt(T) Z)
    """
    if s0 <= 0.0 or sigma <= 0.0 or T <= 0.0 or n_paths <= 0:
        raise ValueError("Invalid inputs.")

    rng = np.random.default_rng(seed)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        m = n_paths // 2
        z = rng.standard_normal(m)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    return s0 * np.exp((mu - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * z)

def simulate_gbm_paths_naive(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> np.ndarray:
    """
    Simulate full GBM paths under the original measure.

    Returns
    -------
    np.ndarray
        Shape (n_paths, n_steps + 1)
    """
    if s0 <= 0.0 or sigma <= 0.0 or T <= 0.0 or n_steps <= 0 or n_paths <= 0:
        raise ValueError("Invalid inputs.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        m = n_paths // 2
        z = rng.standard_normal((m, n_steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0

    drift = (mu - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)

    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(drift + vol * z[:, i])

    return paths

def estimate_path_crash_event_naive(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> RareEventEstimate:
    """
    Naive Monte Carlo estimate of:

        P(min_{0 <= t <= T} S_t <= barrier)

    using discretely monitored GBM paths.
    """
    paths = simulate_gbm_paths_naive(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    running_min = np.min(paths, axis=1)
    indicators = (running_min <= barrier).astype(float)

    p_hat = float(np.mean(indicators))
    se = float(np.std(indicators, ddof=1) / math.sqrt(n_paths))
    rel_err = _safe_relative_error(p_hat, se)
    count = int(np.sum(indicators))

    details = (
        f"Naive MC Path Event\n"
        f"P(min_t S_t <= {barrier:.4f}) ≈ {p_hat:.10f}\n"
        f"Std. error = {se:.10f}\n"
        f"Relative error = {rel_err:.6f}\n"
        f"Event count = {count} / {n_paths}"
    )

    return RareEventEstimate(
        method="naive_path_mc",
        probability_estimate=p_hat,
        standard_error=se,
        relative_error=rel_err,
        n_paths=n_paths,
        event_count=count,
        details=details,
    )
def estimate_path_crash_event_importance_sampling(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_steps: int,
    n_paths: int,
    theta: float,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> tuple[RareEventEstimate, WeightDiagnostics]:
    """
    Importance sampling estimate of:

        P(min_{0 <= t <= T} S_t <= barrier)

    by shifting the Brownian increments pathwise.

    Under Q:
        dW^Q = dW + theta dt

    Equivalently, in discrete time:
        Delta W_tilted = sqrt(dt) * Z + theta * dt

    The likelihood ratio is:
        dP/dQ = exp(-theta * W_T^Q + 0.5 * theta^2 * T)
    """
    if s0 <= 0.0 or sigma <= 0.0 or T <= 0.0 or n_steps <= 0 or n_paths <= 0:
        raise ValueError("Invalid inputs.")

    rng = np.random.default_rng(seed)
    dt = T / n_steps
    sqrt_dt = math.sqrt(dt)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        m = n_paths // 2
        z = rng.standard_normal((m, n_steps))
        z = np.vstack([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))

    # Tilted Brownian increments under Q
    dW_tilted = sqrt_dt * z + theta * dt

    paths = np.empty((n_paths, n_steps + 1), dtype=float)
    paths[:, 0] = s0

    drift = (mu - 0.5 * sigma * sigma) * dt

    for i in range(n_steps):
        paths[:, i + 1] = paths[:, i] * np.exp(drift + sigma * dW_tilted[:, i])

    running_min = np.min(paths, axis=1)
    indicators = (running_min <= barrier).astype(float)

    # W_T under the tilted measure
    W_T_tilted = np.sum(dW_tilted, axis=1)

    # likelihood ratio dP/dQ
    weights = np.exp(-theta * W_T_tilted + 0.5 * theta * theta * T)

    weighted_indicators = indicators * weights

    p_hat = float(np.mean(weighted_indicators))
    se = float(np.std(weighted_indicators, ddof=1) / math.sqrt(n_paths))
    rel_err = _safe_relative_error(p_hat, se)
    count = int(np.sum(indicators))
    wd = _weight_diagnostics(weights)

    details = (
        f"Importance Sampling Path Event\n"
        f"P(min_t S_t <= {barrier:.4f}) ≈ {p_hat:.10f}\n"
        f"Std. error = {se:.10f}\n"
        f"Relative error = {rel_err:.6f}\n"
        f"Crash hits under tilted measure = {count} / {n_paths}\n"
        f"theta = {theta:.6f}\n"
        f"ESS = {wd.ess:.2f}\n"
        f"Normalized weight entropy = {wd.normalized_weight_entropy:.6f}\n"
        f"Max normalized weight = {wd.max_normalized_weight:.6f}"
    )

    return (
        RareEventEstimate(
            method="importance_sampling_path",
            probability_estimate=p_hat,
            standard_error=se,
            relative_error=rel_err,
            n_paths=n_paths,
            event_count=count,
            details=details,
        ),
        wd,
    )

def estimate_rare_event_naive(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> RareEventEstimate:
    """
    Naive Monte Carlo estimate of P(S_T <= barrier).
    """
    st = simulate_terminal_gbm_naive(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )
    indicators = (st <= barrier).astype(float)

    p_hat = float(np.mean(indicators))
    se = float(np.std(indicators, ddof=1) / math.sqrt(n_paths))
    rel_err = _safe_relative_error(p_hat, se)
    count = int(np.sum(indicators))

    details = (
        f"Naive MC\n"
        f"P(S_T <= {barrier:.4f}) ≈ {p_hat:.10f}\n"
        f"Std. error = {se:.10f}\n"
        f"Relative error = {rel_err:.6f}\n"
        f"Event count = {count} / {n_paths}"
    )

    return RareEventEstimate(
        method="naive_mc",
        probability_estimate=p_hat,
        standard_error=se,
        relative_error=rel_err,
        n_paths=n_paths,
        event_count=count,
        details=details,
    )

def compare_path_crash_naive_vs_is(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_steps: int,
    n_paths: int,
    theta: Optional[float] = None,
    seed: Optional[int] = 42,
) -> str:
    """
    Compare naive MC and importance sampling for:

        P(min_{0 <= t <= T} S_t <= barrier)
    """
    if theta is None:
        theta = suggested_theta_for_left_tail(
            s0=s0,
            mu=mu,
            sigma=sigma,
            T=T,
            barrier=barrier,
        )

    naive = estimate_path_crash_event_naive(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        barrier=barrier,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )

    is_est, wd = estimate_path_crash_event_importance_sampling(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        barrier=barrier,
        n_steps=n_steps,
        n_paths=n_paths,
        theta=theta,
        seed=seed,
        antithetic=True,
    )

    variance_ratio = (
        (naive.standard_error ** 2) / (is_est.standard_error ** 2)
        if is_est.standard_error > 0.0
        else float("inf")
    )

    return (
        "=== Path-Dependent Crash Event Comparison ===\n"
        f"S0={s0:.4f}, mu={mu:.4f}, sigma={sigma:.4f}, T={T:.4f}, "
        f"barrier={barrier:.4f}, n_steps={n_steps}\n"
        f"Suggested/used theta = {theta:.6f}\n\n"
        f"{naive.details}\n\n"
        f"{is_est.details}\n\n"
        f"Variance reduction factor ≈ {variance_ratio:.2f}x\n"
        f"ESS / N ≈ {wd.ess / n_paths:.6f}"
    )

def estimate_rare_event_importance_sampling(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_paths: int,
    theta: float,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> tuple[RareEventEstimate, WeightDiagnostics]:
    """
    Importance sampling estimate of P(S_T <= barrier) using exponential tilting.

    We simulate under a shifted normal:
        Z_tilted = Z + theta

    For GBM terminal simulation:
        S_T^tilted = S_0 exp((mu - 0.5 sigma^2)T + sigma sqrt(T)(Z + theta))

    The Radon-Nikodym weight correcting back to the original measure is:
        w(Z_tilted) = exp(-theta * Z_tilted + 0.5 theta^2)

    because if Y ~ N(theta, 1) and target is N(0,1), then:
        dP/dQ = exp(-theta Y + 0.5 theta^2)

    The estimator is:
        p_hat = E_Q[ 1{S_T^tilted <= B} * w(Y) ].
    """
    if s0 <= 0.0 or sigma <= 0.0 or T <= 0.0 or n_paths <= 0:
        raise ValueError("Invalid inputs.")

    rng = np.random.default_rng(seed)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        m = n_paths // 2
        z = rng.standard_normal(m)
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal(n_paths)

    y = z + theta  # tilted standard normal under Q
    st_tilted = s0 * np.exp((mu - 0.5 * sigma * sigma) * T + sigma * math.sqrt(T) * y)

    indicators = (st_tilted <= barrier).astype(float)

    # likelihood ratio dP/dQ
    weights = np.exp(-theta * y + 0.5 * theta * theta)

    weighted_indicators = indicators * weights

    p_hat = float(np.mean(weighted_indicators))
    se = float(np.std(weighted_indicators, ddof=1) / math.sqrt(n_paths))
    rel_err = _safe_relative_error(p_hat, se)
    count = int(np.sum(indicators))
    wd = _weight_diagnostics(weights)

    details = (
        f"Importance Sampling\n"
        f"P(S_T <= {barrier:.4f}) ≈ {p_hat:.10f}\n"
        f"Std. error = {se:.10f}\n"
        f"Relative error = {rel_err:.6f}\n"
        f"Rare event hits under tilted measure = {count} / {n_paths}\n"
        f"theta = {theta:.6f}\n"
        f"ESS = {wd.ess:.2f}\n"
        f"Normalized weight entropy = {wd.normalized_weight_entropy:.6f}\n"
        f"Max normalized weight = {wd.max_normalized_weight:.6f}"
    )

    return (
        RareEventEstimate(
            method="importance_sampling",
            probability_estimate=p_hat,
            standard_error=se,
            relative_error=rel_err,
            n_paths=n_paths,
            event_count=count,
            details=details,
        ),
        wd,
    )
def suggested_theta_for_left_tail(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
) -> float:
    """
    Heuristic tilt so that the rare event sits near the center under the tilted measure.

    We want:
        log(barrier / s0) ≈ (mu - 0.5 sigma^2)T + sigma sqrt(T) * theta

    so
        theta ≈ [log(barrier/s0) - (mu - 0.5 sigma^2)T] / (sigma sqrt(T))

    For a left-tail rare event, this will usually be negative.
    """
    if barrier <= 0.0:
        raise ValueError("barrier must be positive.")
    return (
        math.log(barrier / s0) - (mu - 0.5 * sigma * sigma) * T
    ) / (sigma * math.sqrt(T))


def compare_naive_vs_importance_sampling(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    barrier: float,
    n_paths: int,
    theta: Optional[float] = None,
    seed: Optional[int] = 42,
) -> str:
    """
    Run both estimators and return a compact comparison report.
    """
    if theta is None:
        theta = suggested_theta_for_left_tail(
            s0=s0,
            mu=mu,
            sigma=sigma,
            T=T,
            barrier=barrier,
        )

    naive = estimate_rare_event_naive(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        barrier=barrier,
        n_paths=n_paths,
        seed=seed,
        antithetic=True,
    )

    is_est, wd = estimate_rare_event_importance_sampling(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        barrier=barrier,
        n_paths=n_paths,
        theta=theta,
        seed=seed,
        antithetic=True,
    )

    variance_ratio = (
        (naive.standard_error ** 2) / (is_est.standard_error ** 2)
        if is_est.standard_error > 0.0
        else float("inf")
    )

    return (
        "=== Rare Event Simulation Comparison ===\n"
        f"S0={s0:.4f}, mu={mu:.4f}, sigma={sigma:.4f}, T={T:.4f}, barrier={barrier:.4f}\n"
        f"Suggested/used theta = {theta:.6f}\n\n"
        f"{naive.details}\n\n"
        f"{is_est.details}\n\n"
        f"Variance reduction factor ≈ {variance_ratio:.2f}x\n"
        f"ESS / N ≈ {wd.ess / n_paths:.6f}"
    )
def estimate_drawdown_event_naive(
    *,
    s0: float,
    mu: float,
    sigma: float,
    T: float,
    drawdown_level: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> RareEventEstimate:
    """
    Naive Monte Carlo estimate of the event:

        P( max drawdown >= drawdown_level )

    where max drawdown is defined pathwise as:

        max_t [ 1 - S_t / running_max_t ]

    Parameters
    ----------
    drawdown_level : float
        Relative drawdown threshold in (0,1), e.g. 0.30 for a 30% drawdown.
    """
    if not (0.0 < drawdown_level < 1.0):
        raise ValueError("drawdown_level must be in (0,1).")

    paths = simulate_gbm_paths_naive(
        s0=s0,
        mu=mu,
        sigma=sigma,
        T=T,
        n_steps=n_steps,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
    )

    running_max = np.maximum.accumulate(paths, axis=1)
    drawdowns = 1.0 - paths / running_max
    max_drawdown = np.max(drawdowns, axis=1)

    indicators = (max_drawdown >= drawdown_level).astype(float)

    p_hat = float(np.mean(indicators))
    se = float(np.std(indicators, ddof=1) / math.sqrt(n_paths))
    rel_err = _safe_relative_error(p_hat, se)
    count = int(np.sum(indicators))

    details = (
        f"Naive MC Drawdown Event\n"
        f"P(max drawdown >= {drawdown_level:.2%}) ≈ {p_hat:.10f}\n"
        f"Std. error = {se:.10f}\n"
        f"Relative error = {rel_err:.6f}\n"
        f"Event count = {count} / {n_paths}"
    )

    return RareEventEstimate(
        method="naive_drawdown_mc",
        probability_estimate=p_hat,
        standard_error=se,
        relative_error=rel_err,
        n_paths=n_paths,
        event_count=count,
        details=details,
    )
