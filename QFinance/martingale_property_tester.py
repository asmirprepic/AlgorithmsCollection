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
