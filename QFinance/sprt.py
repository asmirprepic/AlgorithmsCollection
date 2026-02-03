from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import math


class SPRTDecision(str, Enum):
    CONTINUE = "continue"
    ACCEPT_H1 = "accept_h1"
    ACCEPT_H0 = "accept_h0"

@dataclass(frozen=True, slots=True)
class SPRTConfig:
    """
    Gaussian SPRT for known variance.

    Tests:
        H0: X_t ~ N(mu0, sigma^2)
        H1: X_t ~ N(mu1, sigma^2)

    Parameters
    ----------
    mu0 : float
        Mean under null hypothesis.
    mu1 : float
        Mean under alternative hypothesis.
    sigma : float
        Known standard deviation of observations.
    alpha : float
        Type I error probability P(accept H1 | H0 true).
    beta : float
        Type II error probability P(accept H0 | H1 true).
    """

    mu0: float
    mu1: float
    sigma: float
    alpha: float = 0.01
    beta: float = 0.01

    def __post_init__(self) -> None:
        if self.sigma <= 0.0:
            raise ValueError("sigma must be > 0.")
        if self.mu0 == self.mu1:
            raise ValueError("mu0 and mu1 must be different.")
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0,1).")
        if not (0.0 < self.beta < 1.0):
            raise ValueError("beta must be in (0,1).")


@dataclass(slots=True)
class SPRTState:
    """
    Mutable state of the test.
    """
    n: int = 0
    llr: float = 0.0
    decision: SPRTDecision = SPRTDecision.CONTINUE

    def reset(self) -> None:
        self.n = 0
        self.llr = 0.0
        self.decision = SPRTDecision.CONTINUE
