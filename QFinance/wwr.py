from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


# =============================================================================
# Basic curves
# =============================================================================

@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:
    r: float  # continuously compounded

    def df(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("t must be non-negative.")
        return float(np.exp(-self.r * t))


# =============================================================================
# WWR model: equity + credit factor -> stochastic intensity
# =============================================================================

@dataclass(frozen=True, slots=True)
class WWRParams:
    # Time/grid
    T: float
    n_steps: int

    # Equity (GBM)
    s0: float
    mu: float       # risk-neutral drift (often r - q)
    sigma_s: float  # equity vol

    # Credit factor (OU)
    x0: float
    kappa: float    # mean reversion
    theta: float    # long-run mean
    sigma_x: float  # factor vol

    # Intensity coupling
    lambda0: float  # base intensity level (annualized)
    beta: float     # coupling strength (beta=0 -> no WWR through factor)

    # Correlation between equity and credit shocks
    rho: float      # in [-1, 1]

    # CVA settings
    lgd: float      # loss given default in [0,1]


@dataclass(slots=True)
class WWRSimulationResult:
    time_grid: np.ndarray              # (n_steps+1,)
    s_paths: np.ndarray                # (n_paths, n_steps+1)
    x_paths: np.ndarray                # (n_paths, n_steps+1)
    lambda_paths: np.ndarray           # (n_paths, n_steps+1)
    int_lambda: np.ndarray             # (n_paths, n_steps+1)  integral_0^t lambda du (left-Riemann)
    survival: np.ndarray               # (n_paths, n_steps+1)  exp(-int_lambda)


class WrongWayRiskSimulator:
    """
    Simulate joint equity + credit-factor paths and a stochastic default intensity:
        S_t follows GBM
        X_t follows OU
        lambda_t = lambda0 * exp(beta * X_t)

    with correlated Brownian increments between S and X.
    """

    def __init__(self, params: WWRParams, seed: Optional[int] = None) -> None:
        self.p = params
        if not (-1.0 <= self.p.rho <= 1.0):
            raise ValueError("rho must be in [-1,1].")
        if not (0.0 <= self.p.lgd <= 1.0):
            raise ValueError("lgd must be in [0,1].")
        if self.p.T <= 0.0 or self.p.n_steps <= 0:
            raise ValueError("T must be > 0 and n_steps must be > 0.")
        self.rng = np.random.default_rng(seed)

    def simulate(self, n_paths: int, antithetic: bool = True) -> WWRSimulationResult:
        p = self.p
        dt = p.T / p.n_steps
        t = np.linspace(0.0, p.T, p.n_steps + 1)

        # If antithetic, we simulate n_paths/2 base and mirror them.
        if antithetic:
            if n_paths % 2 != 0:
                raise ValueError("n_paths must be even when antithetic=True.")
            n_base = n_paths // 2
        else:
            n_base = n_paths

        # Correlated normals: [z_s, z_x] with corr rho
        z1 = self.rng.standard_normal((n_base, p.n_steps))
        z2 = self.rng.standard_normal((n_base, p.n_steps))
        z_s = z1
        z_x = p.rho * z1 + np.sqrt(max(0.0, 1.0 - p.rho * p.rho)) * z2

        if antithetic:
            z_s = np.vstack([z_s_]()_
