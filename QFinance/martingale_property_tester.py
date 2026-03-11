from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

@dataclass(slots = True)
class MartinGaleTestResult:
    """
    Diagnostics for martingale property
    Attributes
    ----------
    time_grid : np.ndarray
        Simulation times, shape (n_steps + 1,).
    sample_mean_process : np.ndarray
        Monte Carlo estimate of E[M_t], shape (n_steps + 1,).
    theoretical_mean : np.ndarray
        Theoretical martingale mean, usually constant, shape (n_steps + 1,).
    abs_error : np.ndarray
        Absolute error |E[M_t] - theoretical_mean_t|.
    rel_error : np.ndarray
        Relative error, with safe denominator.
    std_error : np.ndarray
        Standard error of the Monte Carlo mean estimate at each time.
    max_abs_error : float
        Maximum absolute error across the time grid.
    max_rel_error : float
        Maximum relative error across the time grid.
    passed : bool
        Whether the martingale check passes the requested tolerance rule.
    message : str
        Human-readable summary.
    """

    time_grid: np.ndarray
    sample_mean_process: np.ndarray
    theoretical_mean: np.ndarray
    abs_error: np.ndarray
    rel_error: np.ndarray
    std_error: np.ndarray
    max_abs_error: float
    max_rel_error: float
    passed: bool
    message: str

def _validate_inputs(
    time_grid: np.ndarray,
    paths: np.ndarray
) -> None:
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be 1D.")
    if paths.ndim != 2:
        raise ValueError("paths must be 2D with shape (n_paths, n_steps + 1).")
    if paths.shape[1] != time_grid.shape[0]:
        raise ValueError("paths.shape[1] must equal len(time_grid).")
    if np.any(np.diff(time_grid) <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")

def discounted_process(
    paths: np.ndarray,
    time_grid: np.ndarray,
    rate: float | np.ndarray
) -> np.ndarray:
    """
    Constructing discounted paths M_t = exp(-r t) S_t for flat time

    Params:
    --------------
    paths: np.ndarray
        SImulated asset paths, shape(n_paths, n_steps+1)
    time_grid: np.ndarrary
        Time_grid, shape (n_steps+1,)
    rate: float | np.ndarray
        Flat cc rate or array of instantenous rates on the same time grid

    """

    _validate_inputs(time_grid, paths)

    if np.isscalar(rate):
        dfs = np.exp(-float(rate) * time_grid)
    else:
        rate = np.asarra(rate, dtype = float)
        if rate.shape != time_grid.shape:
            raise  ValueError("If rate is an array, it must have the same shape as time_grid.")

        # Approx integral
        integ = np.zeros_like(time_grid, dtype = float)
        dt = np.diff(time_grid)
        integ[1:] = np.cumsum(0.5 * (rate[:-1] + rate[1:]) * dt)
        dfs = np.exp(-integ)

    return paths * dfs[None, :]

def martingale_property_test(
    process_paths: np.ndarray,
    time_grid: np.ndarray,
    theoretical_mean: float | np.ndarray,
    *,
    abs_tol: float = 1e-12,
    rel_tol: float = 1e-12,
    sigma_rule: float = 3.0
) -> MartinGaleTestResult:
    """
    Testing if a simulated process behaves like a martingale in expectation
    Parameters
    ----------
    process_paths : np.ndarray
        Paths of the process M_t to be tested, shape (n_paths, n_steps + 1).
        For a risk-neutral asset test, this is usually the discounted process.
    time_grid : np.ndarray
        Time grid, shape (n_steps + 1,).
    theoretical_mean : float | np.ndarray
        Theoretical expectation of M_t. Often constant and equal to M_0.
    abs_tol : float
        Absolute tolerance on the mean deviation.
    rel_tol : float
        Relative tolerance on the mean deviation.
    sigma_rule : float
        Additional statistical check: require
            |sample_mean - theoretical| <= sigma_rule * std_error
        at all times.

    """

    _validate_inputs(time_grid,process_paths)

    n_paths = process_paths.shape[0]
    if n_paths < 2:
        raise ValueError("Need at least 2 paths to estimate standard error")
    sample_mean = np.mean(process_paths, axis=0)
    sample_std = np.std(process_paths, axis=0, ddof=1)
    std_error = sample_std / np.sqrt(n_paths)

    if np.isscalar(theoretical_mean):
        theo = np.full_like(sample_mean, float(theoretical_mean), dtype=float)
    else:
        theo = np.asarray(theoretical_mean, dtype=float)
        if theo.shape != sample_mean.shape:
            raise ValueError("theoretical_mean array must have the same shape as time_grid.")

    abs_error = np.abs(sample_mean - theo)
    denom = np.maximum(np.abs(theo), 1e-12)
    rel_error = abs_error / denom

    max_abs_error = float(np.max(abs_error))
    max_rel_error = float(np.max(rel_error))

    abs_rel_pass = bool(np.all((abs_error <= abs_tol) | (rel_error <= rel_tol)))
    sigma_pass = bool(np.all(abs_error <= sigma_rule * std_error + 1e-15))
    passed = abs_rel_pass and sigma_pass

    message = (
    f"Martingale test {'PASSED' if passed else 'FAILED'} | "
    f"max_abs_error={max_abs_error:.6e}, "
    f"max_rel_error={max_rel_error:.6e}, "
    f"abs_tol={abs_tol:.2e}, rel_tol={rel_tol:.2e}, "
    f"sigma_rule={sigma_rule:.1f}"
    )

    return MartinGaleTestResult(
        time_grid=time_grid.copy(),
        sample_mean_process=sample_mean,
        theoretical_mean=theo,
        abs_error=abs_error,
        rel_error=rel_error,
        std_error=std_error,
        max_abs_error=max_abs_error,
        max_rel_error=max_rel_error,
        passed=passed,
        message=message,
    )
