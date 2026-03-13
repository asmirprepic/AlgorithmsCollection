from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

@dataclass(slots = True)
class SABRParameters:
    alpha: float
    beta: float
    rho: float
    nu: float


@dataclass(slots=True)
class SABRCalibrationResult:
    params: SABRParameters
    objective_value: float
    market_vols: np.ndarray
    fitted_vols: np.ndarray
    strikes: np.ndarray
    rmse: float
    max_abs_error: float


def sabr_implied_vol_hagan(
    f: float,
    k: float,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float,
    eps: float = 1e-12,
) -> float:
    """
        Hagan et al. lognormal SABR implied volatility approximation.

        Parameters
        ----------
        f : float
            Forward.
        k : float
            Strike.
        t : float
            Time to maturity.
        alpha : float
            Initial volatility level.
        beta : float
            CEV elasticity parameter in [0, 1].
        rho : float
            Correlation in (-1, 1).
        nu : float
            Vol-of-vol > 0.

        Returns
        -------
        float
            Black implied volatility.

        Notes
        -----
        This is the classic Hagan lognormal asymptotic formula.

        For ATM (f ~= k), a dedicated limit formula is used for stability.
        """
    if f <= 0.0 or k <= 0.0:
        raise ValueError("f and k must be positive.")
    if t < 0.0:
        raise ValueError("t must be non-negative.")
    if alpha <= 0.0:
        raise ValueError("alpha must be positive.")
    if not (0.0 <= beta <= 1.0):
        raise ValueError("beta must be in [0, 1].")
    if not (-1.0 < rho < 1.0):
        raise ValueError("rho must be in (-1, 1).")
    if nu < 0.0:
        raise ValueError("nu must be non-negative.")

    one_minus_beta = 1.0 - beta

    # ATM special case
    if abs(f - k) < eps:
        fk_beta = f ** one_minus_beta

        term1 = alpha / fk_beta

        term2 = (
            ((one_minus_beta ** 2) / 24.0) * (alpha ** 2) / (f ** (2.0 * one_minus_beta))
            + (rho * beta * nu * alpha) / (4.0 * f ** one_minus_beta)
            + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
        ) * t

        return term1 * (1.0 + term2)

    log_fk = math.log(f / k)
    fk_beta = (f * k) ** (0.5 * one_minus_beta)

    z = (nu / alpha) * fk_beta * log_fk
    x_z_num = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
    x_z_den = 1.0 - rho
    x_z = math.log(x_z_num / x_z_den)

    log_fk2 = log_fk * log_fk
    log_fk4 = log_fk2 * log_fk2

    denom = fk_beta * (
        1.0
        + (one_minus_beta ** 2 / 24.0) * log_fk2
        + (one_minus_beta ** 4 / 1920.0) * log_fk4
    )

    time_corr = 1.0 + (
        ((one_minus_beta ** 2) / 24.0) * (alpha ** 2) / ((f * k) ** one_minus_beta)
        + (rho * beta * nu * alpha) / (4.0 * (f * k) ** (0.5 * one_minus_beta))
        + ((2.0 - 3.0 * rho ** 2) / 24.0) * nu ** 2
    ) * t

    if abs(z) < 1e-10:
        z_over_xz = 1.0
    else:
        z_over_xz = z / x_z

    return (alpha / denom) * z_over_xz * time_corr
