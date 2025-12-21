from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Table
import numpy as np

DiscountFactorFn = Callable[[float],float]

@dataclass(frozen=True, slots=True)
class FlatDiscountCurve:

    """
    Flat continously compounded discount curve
    """
    r: float

    def df(self, t: float) -> float:
        if t < 0.0:
            raise   ValueError("t most be non negative")
        return float(np.exp(-self.r * t))

@dataclass(frozen=True, slots= True)
class PieceWiseHazardCurve:
    """
    Piecewise- constant hazard curve on intervals {t_{i-1},t_i}
    Attributes:
    -------------
    knots: np.ndarray
        Knot times [T1, T2, ..., Tn], strictily increasing.

    hazards:
        Hazard rate lambda_i on (T_{i-1}, T_i), with T0 = 0
    """

    knots = np.ndarray
    hazards = np.ndarray

    def __post_init__(self) -> None:
        if self.knots.ndim != 1 or self.hazards.ndim != 1 :
            raise ValueError("Knots and hazards must be 1D arrays")
        if self.knots.shape[0] != self.hazards.shape[0]:
            raise ValueError("Knots and hazards must have the same length")
        if self.knots.shape[0] == 0:
            raise ValueError("Empty curve")
        if np.any(self.knots <= 0.0):
            raise ValueError("All knots must be > 0.")
        if np.any(np.diff(self.knots) <= 0.0):
            raise ValueError("knots must be strictly increasing.")
        if np.any(self.hazards < 0.0):
            raise ValueError("hazards must be non-negative.")

    def survival(self, t:float) -> float:
        """
        S(0,t) under piecewise-constant hazards:
            S(t) = exp(-∫_0^t λ(u) du)      )
        """
        if t < 0.0:
            raise ValueError("t must be non-negative.")
        if t == 0.0:
            return 1.0

        knots = self.knots
        hz = self.hazards

        acc = 0.0
        t_prev = 0.0
        for i, T in enumerate(knots):
            lam = float(hz[i])
            if t <= T:
                acc += lam * (t - t_prev)
                return float(np.exp(-acc))
            acc += lam * (T - t_prev)
            t_prev = float(T)

        # Beyond last knot, hold last hazard flat
        acc += float(hz[-1]) * (t - float(knots[-1]))
        return float(np.exp(-acc))

    def default_prob(self, t0: float, t1: float) -> float:
        """
        P(t0 < tau <= t1) = S(t0) - S(t1)
        """
        if t1 < t0:
            raise ValueError("t1 must be >= t0.")
        return self.survival(t0) - self.survival(t1)
