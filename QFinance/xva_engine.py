from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Tuple

import numpy as np


class DiscountCurve(Protocol):
    """
    Interface for discounting future cashflows.

    Implementations should guarantee:
    - discount_factor(0.0) == 1.0 (up to numerical noise)
    - t >= 0
    """

    def discount_factor(self, t: float) -> float:
        ...

@dataclass
class FlatDiscountCurve:
    """
    Simple flat continously-compound discount curve:
        P(0,t) = exp(-r * t )

    Parameters:
    -----------
    rate: float
        Cosntant risk-free short rate r

    """

    rate: float

    def discount_factor(self, t, float) -> float:
        if t < 0.0:
            raise ValueError("Time t must be non-negative")

        return float(np.exp(-self.rate*t)
                )

class HazardCurve(Protocol):
    """
    Interface for default intensity / survival probability.
    """

    def intensity(self, t: float) -> float:
        """
        Instantaneous default intensity λ(t).
        """
        ...

    def survival_prob(self, t: float) -> float:
        """
        Survival probability S(0,t) = P(τ > t).
        """
        ...

class Trade(abc.ABC):
    """
    Abstract base class.

    A trade knows how to compute its risk neutral mark to market
    given simulated underlying paths

    """

    @abc.abstractmethod
    def exposure_paths(
        self,
        time_grid: np.ndarray,
        underlying_paths: np.ndarray,
        discount_curve: DiscountCurve
    ) -> np.ndarray:
        """
        Compute discounted mark-to-market of this trade along paths.

        Parameters:
        -------------
        time_grid: np.ndarray, shape (T+1, )
            Monotone increasing times, starting at 0.
        underlying_paths: np.ndarray, shape (N, T+1)
            Simulated underlying price paths. Underlying paths [i, j]
            is the underlying price of path i at time t_j
        discount_curve: DiscountCurve
            Discount curve for PV calculation

        Returns:
        --------------
        exposures: np.ndarray, shape (N, T+1)
            Discounted mark-to-market per path and time,
            from the perspective of party long the trade

        """

        raise NotImplementedError

@dataclass(slots = True)
class NettingSet:
    """
    Collection of trades subject to netting under CSA.

    """

    trades: List[Trade] = field(default_factory=list)

    def add_trade(self, trade: Trade) -> None:
        self.trades.append(trade)

    def exposure_paths(
        self,
        time_grid: np.ndarray,
        underlying_paths: np.ndarray,
        discount_curve: DiscountCurve
    ) -> np.ndarray:

        if not self.trades:
            raise ValueError("Net setting does not contain trades")

        total = np.zeros_like(underlying_paths, dtype = float)
        for trade in self.trades:
            total += trade.exposure_paths(time_grid, underlying_paths, discount_curve)
        return total


@dataclass(slots=True)
class Counterparty:
    """
    Representation of a counterparty for XVA purposes.

    Parameters
    ----------
    name : str
        Identifier.
    hazard_curve : HazardCurve
        Default intensity and survival probability.
    lgd : float
        Loss-given-default in [0,1].
    funding_spread : float
        Constant funding spread (annualized) over risk-free, used for FVA.
    """

    name: str
    hazard_curve: HazardCurve
    lgd: float
    funding_spread: float
