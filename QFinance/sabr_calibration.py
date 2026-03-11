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
