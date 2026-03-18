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

def sabr_surface_slice_vols(
    f: float,
    strikes: np.ndarray,
    t: float,
    alpha: float,
    beta: float,
    rho: float,
    nu: float
) -> np.ndarray:
    """
    Compute SABR implied vol
    """
    strikes = np.asarray(strikes, dtype = float)
    return np.array(
        [sabr_implied_vol_hagan(f,k,t, alpha, beta, rho, nu) for k in strikes],
        dtype = float
    )

def _objective(
    x: np.ndarray,
    f: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    t: float,
    beta: float,
    penalty: float = 1e6,

) -> float:
    alpha, rho, nu = x

    if alpha <= 0.0 or nu < 0.0 or not (-0.999 < rho <0.999):
        return penalty

    try:
        model_vols = sabr_surface_slice_vols(
            f =f,
            strikes=strikes,
            t = t,
            alpha=alpha,
            beta=beta,
            rho=rho,
            nu = nu,
        )
    except Exception:
        return penalty

    if np.any(~np.isinfinite(model_vols)) or np.any(model_vols <= 0.0):
        return penalty

    err = model_vols - market_vols
    return float(np.mean(err * err))

def calibrate_sabr_slice(
    f: float,
    strikes: np.ndarray,
    market_vols: np.ndarray,
    t: float,
    beta: float = 0.5,
    alpha_bounds: tuple[float, float] = (1e-4, 2.0),
    rho_bounds: tuple[float, float] = (-0.999, 0.999),
    nu_bounds: tuple[float, float] = (1e-4, 3.0),
    coarse_grid_size: int = 8,
    n_refinements: int = 1500,
    random_seed: Optional[int] = 42,
) -> SABRCalibrationResult:
    """
    Calibrate SABR parmaters (alpha, rho, nu) for single maturity slice,
    keeping beta fixed.

    Strategy
    ---------
    1. Coarse grid search
    2. Random local refinement
    """
    if strikes.ndim != 1 or market_vols.ndim != 1:
        raise ValueError("strikes and market_vols must be 1D arrays.")
    if strikes.shape[0] != market_vols.shape[0]:
        raise ValueError("strikes and market_vols must have same length.")
    if np.any(strikes <= 0.0):
        raise ValueError("strikes must be positive.")
    if np.any(market_vols <= 0.0):
        raise ValueError("market_vols must be positive.")
    if f <= 0.0:
        raise ValueError("f must be positive.")
    if t <= 0.0:
        raise ValueError("t must be positive.")

    rng = np.random.default_rng(random_seed)

    alpha_grid = np.linspace(alpha_bounds[0],alpha_bounds[1],coarse_grid_size)
    rho_grid = np.linspace(rho_bounds[0],rho_bounds[1],coarse_grid_size)
    nu_grid = np.linspace(nu_bounds[0],nu_bounds[1],coarse_grid_size)

    best_x = None
    best_obj = float('inf')

    for alpha in alpha_grid:
        for rho in rho_grid:
            for nu in nu_grid:
                x = np.array([alpha,rho,nu],dtype = float)
                obj = _objective(x,f,strikes,market_vols,t,beta)
                if obj < best_obj:
                    best_obj = obj
                    best_x = x.copy()

    if best_x is None:
        raise RuntimeError("SABR coarse calibration failed")

    alpha_width = 0.25*(alpha_bounds[1]-alpha_bounds[0])
    rho_width = 0.25 * (rho_bounds[1] - rho_bounds[0])
    nu_width = 0.25 * (nu_bounds[1] - nu_bounds[0])

    for i in range(n_refinements):
        shrink = max(0.02, 1.0 - i / max(1, n_refinements))

        cand = np.array([
            best_x[0] + rng.normal(scale=alpha_width * shrink),
                best_x[1] + rng.normal(scale=rho_width * shrink),
                best_x[2] + rng.normal(scale=nu_width * shrink),
        ],dtype = float)
        cand[0] = np.clip(cand[0], alpha_bounds[0], alpha_bounds[1])
        cand[1] = np.clip(cand[1], rho_bounds[0], rho_bounds[1])
        cand[2] = np.clip(cand[2], nu_bounds[0], nu_bounds[1])

        obj = _objective(cand, f, strikes, market_vols, t, beta)
        if obj < best_obj:
            best_obj = obj
            best_x = cand.copy()

    fitted_vols = sabr_surface_slice_vols(
        f=f,
        strikes=strikes,
        t=t,
        alpha=float(best_x[0]),
        beta=beta,
        rho=float(best_x[1]),
        nu=float(best_x[2]),
    )

    errors = fitted_vols - market_vols
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    max_abs_error = float(np.max(np.abs(errors)))

    return SABRCalibrationResult(
        params=SABRParameters(
            alpha=float(best_x[0]),
            beta=float(beta),
            rho=float(best_x[1]),
            nu=float(best_x[2]),
        ),
        objective_value=float(best_obj),
        market_vols=market_vols.copy(),
        fitted_vols=fitted_vols,
        strikes=strikes.copy(),
        rmse=rmse,
        max_abs_error=max_abs_error,
    )

if __name__ == "__main__":
    f = 0.025
    t = 5.0
    beta = 0.5

    true_params = SABRParameters(alpha=0.030, beta=beta, rho=-0.25, nu=0.40)

    strikes = np.array([0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040], dtype=float)

    market_vols = sabr_surface_slice_vols(
        f=f,
        strikes=strikes,
        t=t,
        alpha=true_params.alpha,
        beta=true_params.beta,
        rho=true_params.rho,
        nu=true_params.nu,
    )

    #Noise
    rng = np.random.default_rng(123)
    market_vols = market_vols + rng.normal(scale=0.0005, size=market_vols.shape)

    result = calibrate_sabr_slice(
        f=f,
        strikes=strikes,
        market_vols=market_vols,
        t=t,
        beta=beta,
        coarse_grid_size=7,
        n_refinements=2000,
        random_seed=7,
    )

    print("=== SABR Calibration Result ===")
    print(f"alpha = {result.params.alpha:.6f}")
    print(f"beta  = {result.params.beta:.6f}")
    print(f"rho   = {result.params.rho:.6f}")
    print(f"nu    = {result.params.nu:.6f}")
    print(f"RMSE  = {result.rmse:.8f}")
    print(f"MaxAbsError = {result.max_abs_error:.8f}")

    print("\nStrike    MarketVol    FittedVol")
    for k, vm, vf in zip(result.strikes, result.market_vols, result.fitted_vols):
        print(f"{k:>7.4f}   {vm:>10.6f}   {vf:>10.6f}")
