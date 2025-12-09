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
@dataclass(slots=True)
class EuropeanOptionTrade(Trade):
    """
    Vanilla European option on a single underlying.

    The underlying is assumed to be the simulated price process S_t.

    Parameters
    ----------
    strike : float
        Option strike K.
    maturity : float
        Maturity T (must match a point in time_grid).
    is_call : bool
        True for call, False for put.
    notional : float
        Notional of the option.
    """

    strike: float
    maturity: float
    is_call: bool
    notional: float = 1.0

    def exposure_paths(
        self,
        time_grid: np.ndarray,
        underlying_paths: np.ndarray,
        discount_curve: DiscountCurve,
    ) -> np.ndarray:
        """
        Discounted MtM at each node:

        Before maturity: we mark the option to its risk-neutral price
        by approximating via conditional expectation using continuation
        of the same simulated paths (here simplified to intrinsic at T).
        For a mini XVA engine, we can take a conservative proxy:
        - exposure(t < T) ≈ discounted expected payoff from T, conditional on path.
        That requires nested simulation; instead we approximate by
        re-using terminal payoffs and discounting back uniformly.

        To keep this mini-engine tractable, we:
        - compute payoff at maturity
        - discount payoff to all previous times using P(0,T)/P(0,t)
          as a simple proxy for mark-to-market evolution.

        This is not a production valuation model but a consistent,
        transparent approximation.
        """
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be 1D.")
        if underlying_paths.ndim != 2:
            raise ValueError("underlying_paths must be 2D (N, T+1).")
        if underlying_paths.shape[1] != time_grid.shape[0]:
            raise ValueError("underlying_paths.shape[1] must equal len(time_grid).")

        # Locate maturity index
        idx = np.where(np.isclose(time_grid, self.maturity))[0]
        if idx.size != 1:
            raise ValueError("Maturity must appear exactly once in time_grid.")
        m_idx = int(idx[0])

        s_T = underlying_paths[:, m_idx]
        if self.is_call:
            payoff = np.maximum(s_T - self.strike, 0.0)
        else:
            payoff = np.maximum(self.strike - s_T, 0.0)
        payoff *= self.notional

        # Discount payoff to each time along the grid as a simple proxy
        T = float(self.maturity)
        df_T = discount_curve.discount_factor(T)
        if df_T <= 0.0:
            raise ValueError("Invalid discount factor at maturity.")

        df_t = np.array([discount_curve.discount_factor(float(t)) for t in time_grid])
        # PV_t = payoff * P(0,T) / P(0,t)
        pv_paths = np.empty_like(underlying_paths)
        pv_at_T = payoff * df_T  # present value at t = 0
        # propagate PV back in a term-structure-consistent way
        for j, df in enumerate(df_t):
            if time_grid[j] <= T:
                # Value at time t_j, discounted to time 0: approximated constant PV
                pv_paths[:, j] = pv_at_T
            else:
                # After maturity, no exposure
                pv_paths[:, j] = 0.0

        return pv_paths

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


@dataclass(slots = True)
class XVAResults:
    """
    Container for outputs.

     Attributes
    ----------
    cva : float
        Credit valuation adjustment (loss due to counterparty default).
    dva : float
        Debit valuation adjustment (own default benefit).
    fva : float
        Funding valuation adjustment (cost of funding exposure).
    epe : np.ndarray
        Expected positive exposure along time grid.
    ene : np.ndarray
        Expected negative exposure along time grid.
    time_grid : np.ndarray
        Time grid corresponding to EPE/ENE.

    """
