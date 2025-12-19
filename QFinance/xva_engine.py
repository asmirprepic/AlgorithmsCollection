from __future__ import annotations
import abc
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Protocol, Tuple,Dict

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
class FlatHazardCurve:
    """
    Flat hazard rate curve:

        λ(t) = h
        S(0,t) = exp(-h * t)

    Parameters
    ----------
    hazard_rate : float
        Constant hazard rate h.
    """

    hazard_rate: float

    def intensity(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("Time t must be non-negative.")
        return self.hazard_rate

    def survival_prob(self, t: float) -> float:
        if t < 0.0:
            raise ValueError("Time t must be non-negative.")
        return float(np.exp(-self.hazard_rate * t))

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
@dataclass(slots=True)
class XVAEngine:
    """
    Mini XVA engine for a single netting set vs a counterparty.

    Workflow:
    ---------
    1. Simulate underlying price paths under the risk-neutral measure.
    2. Compute discounted exposure paths of the netting set.
    3. Compute EPE/ENE over time.
    4. Integrate against default probability and discount factors to
       obtain CVA/DVA.
    5. Integrate EPE against funding spread to obtain FVA (simple spec).

    This engine is intentionally transparent and modular, not fully
    production-optimized.
    """

    discount_curve: DiscountCurve
    netting_set: NettingSet
    counterparty: Counterparty
    own_hazard_curve: HazardCurve

    def simulate_underlying_paths(
        self,
        spot0: float,
        time_grid: np.ndarray,
        vol: float,
        drift: Optional[float] = None,
        n_paths: int = 10000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Simulate GBM paths for the underlying under risk-neutral measure.

        dS_t = μ S_t dt + σ S_t dW_t

        Typically μ = r (risk-free rate). If drift is None, we use
        discount_curve.rate for FlatDiscountCurve; otherwise drift must
        be provided.
        """
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be 1D.")
        if time_grid[0] != 0.0:
            raise ValueError("time_grid must start at 0.")

        n_steps = time_grid.shape[0] - 1
        paths = np.empty((n_paths, n_steps + 1), dtype=float)
        paths[:, 0] = spot0

        rng = np.random.default_rng(seed)

        # Determine drift
        if drift is None:
            if isinstance(self.discount_curve, FlatDiscountCurve):
                mu = self.discount_curve.rate
            else:
                raise ValueError("drift must be provided for non-flat discount curve.")
        else:
            mu = drift

        for i in range(n_steps):
            t0 = time_grid[i]
            t1 = time_grid[i + 1]
            dt = float(t1 - t0)
            if dt <= 0.0:
                raise ValueError("time_grid must be strictly increasing.")

            z = rng.standard_normal(n_paths)
            # GBM exact discretization
            paths[:, i + 1] = paths[:, i] * np.exp(
                (mu - 0.5 * vol * vol) * dt + vol * np.sqrt(dt) * z
            )

        return paths

    def _expected_exposure_profiles(
        self,
        time_grid: np.ndarray,
        underlying_paths: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute exposure paths and EPE/ENE.

        Returns
        -------
        exposures : np.ndarray, shape (N, T+1)
        epe : np.ndarray, shape (T+1,)
            Expected positive exposure E[max(V_t,0)].
        ene : np.ndarray, shape (T+1,)
            Expected negative exposure E[min(V_t,0)].
        """
        exposures = self.netting_set.exposure_paths(
            time_grid=time_grid,
            underlying_paths=underlying_paths,
            discount_curve=self.discount_curve,
        )
        epe = np.mean(np.maximum(exposures, 0.0), axis=0)
        ene = np.mean(np.minimum(exposures, 0.0), axis=0)
        return exposures, epe, ene

    def _cva_from_epe(self, time_grid: np.ndarray, epe: np.ndarray) -> float:
        """
        Compute CVA via Riemann sum approximation:

        CVA ≈ LGD * Σ_t EPE(t_mid) * [DF(t_mid)] * [S(0,t_{i}) - S(0,t_{i+1})]
        where default occurs between t_i and t_{i+1}.
        """
        lgd = self.counterparty.lgd
        hz = self.counterparty.hazard_curve
        dc = self.discount_curve

        if time_grid.shape[0] != epe.shape[0]:
            raise ValueError("time_grid and epe must have same length.")

        cva = 0.0
        for i in range(len(time_grid) - 1):
            t0 = float(time_grid[i])
            t1 = float(time_grid[i + 1])
            t_mid = 0.5 * (t0 + t1)

            s0 = hz.survival_prob(t0)
            s1 = hz.survival_prob(t1)
            default_prob = s0 - s1  # P(t0 < τ <= t1)

            df = dc.discount_factor(t_mid)
            exposure = max(epe[i], 0.0)

            cva += lgd * exposure * df * default_prob

        return float(cva)

    def _dva_from_ene(self, time_grid: np.ndarray, ene: np.ndarray) -> float:
        """
        DVA computed symetrically using own default hazard curve and
        expected negative exposure
        """
        hz = self.own_hazard_curve
        dc = self.discount_curve
        lgd_own = 1.0

        if time_grid.shape[0] != ene.shape[0]:
            raise ValueError("time_grid and ene must have same length")

        dva = 0.0
        for i in range(len(time_grid) - 1):
            t0 = float(time_grid[i])
            t1 = float(time_grid[i + 1])
            t_mid = 0.5 *(t0 + t1)

            s0 = hz.survival_prob(t0)
            s1 = hz.survival_prob(t1)
            default_prob = s0 - s1
            df = dc.discount_factor(t_mid)

            exposure = min(ene[i], 0.0)

            dva += lgd_own * exposure * df * default_prob
        return float(dva)

    def _fva_from_epe(self, time_grid: np.ndarray, epe: np.ndarray) -> float:
        """
        Simple FVA approximation:

        FVA = - Int from 0 to t funding_spread*DF(t)*EPE(t)*dt

        """

        spread = self.counterparty.funding_spread
        dc = self.discount_curve

        if time_grid.shape[0] != epe.shape[0]:
            raise ValueError("time_grid and epe must have same length")

        fva = 0.0
        for i in range(len(time_grid) -1 ):
            t0 = float(time_grid[i])
            t1 = float(time_grid[i + 1])
            dt = t1-t0
            t_mid = 0.5 * (t0 + t1)
            df = dc.discount_factor(t_mid)
            exposure_mid= 0.5*(epe[i] + epe[i+1])
            fva += -spread *df *exposure_mid*dt
        return float

    def compute_xva(
        self,
        time_grid: np.ndarray,
        underlying_paths: np.ndarray
    ) -> XVAResults:

        """
        Compute CVA , DVA and FVA give pre-simulated underlying paths

        Parameters:
        --------------
        time_grid : np.ndarray, shape (T+1,)
        underlying_paths : np.ndarray, shape (N, T+1)

        Returns
        -------
        XVAResults
        """
        _, epe, ene = self._expected_exposure_profiles(time_grid, underlying_paths)
        cva = self._cva_from_epe(time_grid, epe)
        dva = self._dva_from_ene(time_grid, ene)
        fva = self._fva_from_epe(time_grid, epe)

        return XVAResults(
            cva=cva,
            dva = dva,
            fva = fva,
            epe = epe,
            ene = ene,
            time_grid=time_grid.copy()
        )

@dataclass(slots=True)
class ExposureReport:
    """
    Standard exposure diagnostics used in XVA / counterparty risk.

    Attributes
    ----------
    time_grid : np.ndarray
        Times (years).
    ee : np.ndarray
        Expected exposure E[V_t].
    epe : np.ndarray
        Expected positive exposure E[max(V_t,0)].
    ene : np.ndarray
        Expected negative exposure E[min(V_t,0)].
    pfe : Dict[float, np.ndarray]
        Potential future exposure quantiles, keyed by q (e.g. 0.95).
        Each array is shape (T+1,).
    peak_epe : float
        max_t EPE(t)
    peak_pfe : Dict[float, float]
        max_t PFE_q(t) for each q
    """

    time_grid: np.ndarray
    ee: np.ndarray
    epe: np.ndarray
    ene: np.ndarray
    pfe: Dict[float, np.ndarray]
    peak_epe: float
    peak_pfe: Dict[float, float]


    def build_exposure_report(
        time_grid: np.ndarray,
        exposure_paths: np.ndarray,
        pfe_levels: Tuple[float, ...] = (0.95, 0.99),
    ) -> ExposureReport:
        """
        Build a standard exposure report from pathwise exposures.

        Parameters
        ----------
        time_grid : (T+1,)
        exposure_paths : (N, T+1)
            Discounted MtM/exposure paths (after netting, and after collateral
            if you apply a CSA).
        pfe_levels : tuple
            Quantiles for PFE (e.g. 0.95, 0.99).

        Returns
        -------
        ExposureReport
        """
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be 1D.")
        if exposure_paths.ndim != 2:
            raise ValueError("exposure_paths must be 2D (N, T+1).")
        if exposure_paths.shape[1] != time_grid.shape[0]:
            raise ValueError("exposure_paths.shape[1] must equal len(time_grid).")

        ee = exposure_paths.mean(axis=0)
        epe = np.maximum(exposure_paths, 0.0).mean(axis=0)
        ene = np.minimum(exposure_paths, 0.0).mean(axis=0)

        pfe: Dict[float, np.ndarray] = {}
        for q in pfe_levels:
            if not (0.0 < q < 1.0):
                raise ValueError("PFE quantiles must be in (0,1).")
            pfe[q] = np.quantile(exposure_paths, q=q, axis=0)

        peak_epe = float(np.max(epe))
        peak_pfe = {q: float(np.max(arr)) for q, arr in pfe.items()}

        return ExposureReport(
            time_grid=time_grid.copy(),
            ee=ee,
            epe=epe,
            ene=ene,
            pfe=pfe,
            peak_epe=peak_epe,
            peak_pfe=peak_pfe,
        )

@dataclass(slots=True)
class XVASanityCheckResult:
    passed: bool
    messages: list[str]


def run_xva_sanity_checks(
    time_grid: np.ndarray,
    exposure_paths: np.ndarray,
    discount_factors: np.ndarray,
    cva: float,
    dva: float,
    fva: float,
    funding_spread: float,
    tol: float = 1e-8,
) -> XVASanityCheckResult:
    """
    Run basic sanity checks on XVA outputs.

    These checks catch common modeling or integration bugs.
    """
    messages: list[str] = []
    passed = True

    # Shape checks
    if exposure_paths.shape[1] != time_grid.shape[0]:
        passed = False
        messages.append("Exposure paths and time grid shape mismatch.")

    # Exposure sign checks
    epe = np.mean(np.maximum(exposure_paths, 0.0), axis=0)
    ene = np.mean(np.minimum(exposure_paths, 0.0), axis=0)

    if np.any(epe < -tol):
        passed = False
        messages.append("EPE contains negative values.")

    if np.any(ene > tol):
        passed = False
        messages.append("ENE contains positive values.")

    # CVA / DVA sign checks
    if cva < -tol:
        passed = False
        messages.append("CVA is negative.")

    if funding_spread >= 0 and fva > tol:
        passed = False
        messages.append("FVA has wrong sign for positive funding spread.")

    # Discount factor monotonicity
    if np.any(np.diff(discount_factors) > tol):
        passed = False
        messages.append("Discount factors are not non-increasing.")

    # Terminal exposure
    if abs(epe[-1]) > tol:
        messages.append("Warning: EPE at maturity is not ~0.")

    return XVASanityCheckResult(passed=passed, messages=messages)


if __name__ == "__main__":
    # Time grid
    T = 5.0  # years
    n_steps = 50
    time_grid = np.linspace(0.0, T, n_steps + 1)

    # Market environment
    r = 0.02
    curve = FlatDiscountCurve(rate=r)

    # Counterparty & own credit
    cp_hazard = FlatHazardCurve(hazard_rate=0.02)      # 2% flat
    own_hazard = FlatHazardCurve(hazard_rate=0.01)     # 1% flat
    counterparty = Counterparty(
        name="CP_A",
        hazard_curve=cp_hazard,
        lgd=0.6,
        funding_spread=0.01,  # 1% funding spread
    )

    # Trades
    trade1 = EuropeanOptionTrade(
        strike=100.0,
        maturity=3.0,
        is_call=True,
        notional=1_000_000.0,
    )
    trade2 = EuropeanOptionTrade(
        strike=90.0,
        maturity=4.0,
        is_call=False,
        notional=500_000.0,
    )

    netting_set = NettingSet(trades=[trade1, trade2])

    engine = XVAEngine(
        discount_curve=curve,
        netting_set=netting_set,
        counterparty=counterparty,
        own_hazard_curve=own_hazard,
    )

    # Simulate paths
    spot0 = 100.0
    vol = 0.2
    n_paths = 20_000

    paths = engine.simulate_underlying_paths(
        spot0=spot0,
        time_grid=time_grid,
        vol=vol,
        drift=None,
        n_paths=n_paths,
        seed=42,
    )

    # Compute XVA
    results = engine.compute_xva(time_grid=time_grid, underlying_paths=paths)
