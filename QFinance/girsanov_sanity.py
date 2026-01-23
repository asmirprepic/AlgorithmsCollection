from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

@dataclass(slots=True)
class GirsanovDiagnostics:
    time_grid: np.ndarray                # (n_steps+1,)
    Z_mean: np.ndarray                   # E_P[Z_t] over time (should ~ 1)
    Z_std: np.ndarray                    # Std_P[Z_t] over time (grows with t)
    martingale_max_abs_err: float        # max_t |E[Z_t]-1|
    eq_change_of_measure_abs_err: float  # |E_Q[F] - E_P[Z_T F]|
    eq_change_of_measure_rel_err: float  # relative version
    details: str


def simulate_brownian(
    *,
    T: float,
    n_steps: int,
    n_paths: int,
    seed: Optional[int] = None,
    antithetic: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate Brownian motion paths W_t under P on an equally spaced grid.

    Returns
    -------
    t : (n_steps+1,)
    W : (n_paths, n_steps+1)
    """
    if T <= 0.0:
        raise ValueError("T must be > 0.")
    if n_steps <= 0:
        raise ValueError("n_steps must be > 0.")
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0.")

    dt = T / n_steps
    t = np.linspace(0.0, T, n_steps + 1)

    rng = np.random.default_rng(seed)

    if antithetic:
        if n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True.")
        n_base = n_paths // 2
    else:
        n_base = n_paths

    dW = np.sqrt(dt) * rng.standard_normal((n_base, n_steps))
    if antithetic:
        dW = np.vstack([dW, -dW])

    W = np.zeros((n_paths, n_steps + 1), dtype=float)
    W[:, 1:] = np.cumsum(dW, axis=1)

    return t, W


def density_process_Z(
    *,
    theta: float,
    time_grid: np.ndarray,
    W: np.ndarray,
) -> np.ndarray:
    """
    Compute Z_t = exp(-theta W_t - 0.5 theta^2 t) pathwise.

    Parameters
    ----------
    theta : float
        Girsanov shift parameter.
    time_grid : (n_steps+1,)
    W : (n_paths, n_steps+1)

    Returns
    -------
    Z : (n_paths, n_steps+1)
    """
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be 1D.")
    if W.ndim != 2:
        raise ValueError("W must be 2D (n_paths, n_steps+1).")
    if W.shape[1] != time_grid.shape[0]:
        raise ValueError("W and time_grid shape mismatch.")

    t = time_grid[None, :]
    return np.exp(-theta * W - 0.5 * theta * theta * t)
