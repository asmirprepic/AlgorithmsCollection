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

def simulate_gbm(
    *,
    S0: float,
    mu: float,
    sigma: float,
    time_grid: np.ndarray,
    dW: np.ndarray,
) -> np.ndarray:
    """
    Simulate GBM using exact discretization:
        S_{t+dt} = S_t * exp((mu - 0.5 sigma^2) dt + sigma dW)

    dW must be Brownian increments under the measure you intend.
    """
    if S0 <= 0.0:
        raise ValueError("S0 must be > 0.")
    if sigma <= 0.0:
        raise ValueError("sigma must be > 0.")
    if time_grid.ndim != 1:
        raise ValueError("time_grid must be 1D.")
    if dW.ndim != 2:
        raise ValueError("dW must be 2D (n_paths, n_steps).")

    n_paths, n_steps = dW.shape
    if time_grid.shape[0] != n_steps + 1:
        raise ValueError("time_grid length must equal n_steps + 1.")

    dt = np.diff(time_grid)
    if np.any(dt <= 0.0):
        raise ValueError("time_grid must be strictly increasing.")

    S = np.empty((n_paths, n_steps + 1), dtype=float)
    S[:, 0] = S0

    for i in range(n_steps):
        dti = float(dt[i])
        S[:, i + 1] = S[:, i] * np.exp((mu - 0.5 * sigma * sigma) * dti + sigma * dW[:, i])

    return S

def girsanov_sanity_check(
    *,
    theta: float,
    T:float,
    n_steps: int,
    n_paths: int,
    payoff_fn: Callable[[np.ndarray],np.ndarray],
    gbm_s0: float = 100.0,
    gbm_sigma: float = 0.2,
    mu_P: float = 0.0,
    seed: Optional[int] = None,
) -> GirsanovDiagnostics:
    """
    Verify change of measure numerically in clean testable way.

    1. Under P: Simulate W^P, build Z_T, simulate S under P(drift mu_P)
    Compute E_P[Z_T *payoffs((S_T))].

    2. Under Q: Simulate W^Q directly by re-using W^P increments but shifting:
     W^Q = W^P + theta * t
     This corresponds to drift change in Brownian motion.
     Simulate S under Q drift mu_Q=mu_P - sigma * theta for GBM driven by W^P
     Then compute E_Q[payoff(S_T)]

    """
