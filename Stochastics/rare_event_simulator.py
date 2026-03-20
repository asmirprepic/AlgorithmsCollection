from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import math
import numpy as np

@dataclass(slots = True)
class RareEventEstimate:
    method: str
    probability_estimate: float
    standard_error: float
    relative_error: float
    n_paths: int
    event_count: int
    details: str
@dataclass(slots=True)
class WeightDiagnostics:
    ess: float
    normalized_weight_entropy: float
    max_normalized_weight: float
    mean_weight: float
    std_weight: float


def _safe_relative_error(est: float, se: float) -> float:
    if abs(est) < 1e-16:
        return float("inf")
    return abs(se / est)


def _weight_diagnostics(weights: np.ndarray) -> WeightDiagnostics:
    w = np.asarray(weights, dtype=float)
    if np.any(w < 0.0):
        raise ValueError("Weights must be non-negative.")

    sw = np.sum(w)
    if sw <= 0.0:
        raise ValueError("Sum of weights must be positive.")

    wn = w / sw
    ess = float((sw * sw) / np.sum(w * w))

    # normalized entropy in [0,1]
    n = len(w)
    entropy = -np.sum(np.where(wn > 0.0, wn * np.log(wn), 0.0))
    entropy_norm = float(entropy / math.log(n)) if n > 1 else 1.0

    return WeightDiagnostics(
        ess=ess,
        normalized_weight_entropy=entropy_norm,
        max_normalized_weight=float(np.max(wn)),
        mean_weight=float(np.mean(w)),
        std_weight=float(np.std(w, ddof=1)) if len(w) > 1 else 0.0,
    )
