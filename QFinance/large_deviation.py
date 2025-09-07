from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


def _ensure_spd(S: np.ndarray) -> np.ndarray:
    """Make sure covariance is SPD (tiny jitter if needed)."""
    S = np.asarray(S, dtype=float)
    if S.ndim != 2 or S.shape[0] != S.shape[1]:
        raise ValueError("Σ must be a square covariance matrix.")
    # jitter if near-singular
    eps = 1e-10
    try:
        np.linalg.cholesky(S)
        return S
    except np.linalg.LinAlgError:
        d = S.shape[0]
        return S + eps * np.eye(d)


def _chol(S: np.ndarray) -> np.ndarray:
    """Cholesky factor with safety."""
    return np.linalg.cholesky(_ensure_spd(S))


def _normal_tail_exact(mean: float, std: float, threshold: float, side: str = "right") -> float:
    """Exact Gaussian tail for sanity checks."""
    from math import erf, sqrt
    z = (threshold - mean) / max(std, 1e-16)
    if side == "right":   # P(Y >= threshold)
        return 0.5 * (1 - erf(z / np.sqrt(2)))
    elif side == "left":  # P(Y <= threshold)
        return 0.5 * (1 + erf(z / np.sqrt(2)))
    else:
        raise ValueError("side must be 'right' or 'left'.")
