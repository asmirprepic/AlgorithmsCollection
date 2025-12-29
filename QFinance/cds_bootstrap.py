from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple,Table
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

@dataclass(slots=True)
class CDSFitDiagnostics:
    maturities: np.ndarray
    market_spreads: np.ndarray
    model_spreads: np.ndarray
    errors_bps: np.ndarray
    max_abs_error_bps: float
    rmse_bps: float

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

def generate_payment_times(maturity: float, pay_freq_per_year: int = 4) -> np.ndarray:
    """
    Generate CDS premium payment times from (0, maturity] at regular frequency.
    E.g. quarterly => 4 payments /year.

    Return times including maturity


    """
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0.")
    if pay_freq_per_year <= 0:
        raise ValueError("pay_freq_per_year must be positive")
    dt = 1.0/pay_freq_per_year
    n = int(np.round(maturity/dt))
    times = dt * np.arange(1, n + 1, dtype = float)
    times[-1] = maturity

    return times

def _integrate_simpson(f: Callable[[np.ndarray],np.ndarray], a: float, b: float, n: int = 20) -> float:
    """
    Simple simpson integration on [a,b] with even n
    """

    if b <= a:
        return 0.0
    if n % 2 == 1:
        n += 1
    x= np.linspace(a,b,n+1)
    y = f(x)
    h = (b-a)/n
    integral = float((h / 3.0) * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))

    return integral

def cds_leg_pv(
    *,
    maturity: float,
    spread: float,
    recovery: float,
    curve: PieceWiseHazardCurve,
    df: DiscountFactorFn,
    pay_freq_per_year: int = 4,
    accrual_on_default: bool = True,
) -> Tuple[float,float]:
    """
    Compute PV of premium and protection legs for CDS with given spread.

    Notes:
    -----------
    - Premium leg is approximated with paymets at scheduled times.
    - Protection leg is integrated (numerically) over time using survival and DF.
    - Accrual-on-default adds an approximation for the accrued premium between
    payment dates in case of default

    Returns:
    ---------
    pv_premium, pv_protection
    """

    if not (0.0 <= recovery < 1.0):
        raise ValueError("recovery must be in [0,1).")
    if spread < 0.0:
        raise ValueError("spread must be non-negative.")

    pay_times = generate_payment_times(maturity=maturity,pay_freq_per_year=pay_freq_per_year)
    dt = 1.0/pay_freq_per_year

    # spread * sum DF(t_i)* alpha_i * S(t_i]
    surv = np.array([curve.survival(float(t)) for t in pay_times])
    dfs = np.array([df(float(t)) for t in pay_times])
    pv_premium = float(spread * np.sum(dfs*dt*surv))

    ## Accrued premium
    if accrual_on_default:
        t_prev = 0.0
        accr = 0.0
        for t_i in pay_times:
            s_prev = curve.survival(float(t_prev))
            s_i = curve.survival(float(t_i))

            dp = s_prev-s_i
            t_mid = 0.5* (t_prev + float(t_i))
            accr += df(t_mid) * dp
            t_prev = float(t_i)
        pv_premium += float(spread * 0.5 *dt* accr)

    def integrand(x: np.ndarray) -> np.ndarray:
        # density approx: -dS/dt = λ(t)*S(t) under hazard model

        knots = curve.knots
        hz = curve.hazards
        idx = np.searchsorted(knots, x, side = "left")
        idx = np.clip(idx, 0, len(hz) - 1)
        lam = hz[idx]

        # S(t)
        S = np.vectorize(curve.surival)(x)
        DF = np.vectorize(df)(x)
        return DF*lam*S

    pv_protection = float((1.0 - recovery)* _integrate_simpson(integrand,0.0,maturity,n=60))
    return pv_premium, pv_protection

def _solve_monotone_bisection(
    f: Callable[[float],float],
    lo: float,
    hi: float,
    tol: float = 1e-10,
    max_iter: int = 200,

) -> float:
    """
    Robust bisection for monotone function with bracket f(lo)*f(hi) <= 0

    """

    flo = f(lo)
    fhi = f(hi)
    if np.isnan(flo) or np.isnan(fhi):
        raise ValueError("Function returned Nan at bracket endpoints.")
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if flo*fhi > 0.0:
        raise ValueError("Root not bracketed")

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if abs(mid) < tol or (hi - lo) < tol:
            return mid
        if flo * fmid <= 0.0:
            hi, fhi = mid, fmid
        else:
            lo, flo = mid, fmid

    return 0.5 * (lo + hi)

def bootstrap_hazard_curve_from_cds(
    maturities: Sequence[float],
    spreads: Sequence[float],
    *,
    recovery: float,
    df: DiscountFactorFn,
    pay_freq_per_year: int = 4,
    accrual_on_default: bool = True,
    hazard_upper: float = 5.0,

) -> PieceWiseHazardCurve:
    """
    Bootstrap a piecewise-constant hazard curve from CDS par spreads.

    Solve sequentially for lambda_i on (T_{i-1},T_i) such that:
        PV_protection(0, T_i) - PV_premium(0, T_i) = 0

    Parameters:
    --------------
    maturities : list[float]
        CDS maturities in years, strictly increasing.
    spreads : list[float]
        Par spreads (decimal, e.g. 0.0125 for 125 bps), same length as maturities.
    recovery : float
        Recovery rate R in [0,1).
    df : callable
        Discount factor function DF(t).
    pay_freq_per_year : int
        Premium payment frequency (4 = quarterly).
    accrual_on_default : bool
        Include accrued premium approximation.
    hazard_upper : float
        Upper bound for hazard rate search in each interval.

    Returns:
    -------------
    PiecewiseHazardCurve
    """

    mats = np.array(maturities, dtype=float)
    sps = np.array(spreads, dtype=float)

    if mats.ndim != 1 or sps.ndim != 1 or mats.shape[0] != sps.shape[0]:
        raise ValueError("maturities and spreads must be 1D arrays of equal length")
    if mats.shape[0] == 0:
        raise ValueError("Need at least one maturity")
    if np.any(mats <= 0.0) or np.any(np.diff(mats) <= 0.0):
        raise ValueError("maturities must be strictily increaseing and > 0")

    if np.any(sps < 0.0):
        raise ValueError("spreads must be nonnegative")
    if not (0.0 <= recovery < 1.0):
        raise ValueError("recovery must be in [0,1).")

    hazards = np.zeros_like(mats)

    # Sequential bootstrap
    for i in range(len(mats)):
        knots_i = mats[ : i + 1].copy()
        hz_i = mats[ : i + 1].copy()

        maturity_i = float(mats[i])
        spread_i = float(sps[i])

        def objective(lam: float) -> float:
            hz_i[-1] = max(0.0,lam)
            curve = PieceWiseHazardCurve(knots = knots_i, hazards = hz_i)
            pv_prem, pv_prot = cds_leg_pv(
                maturity = maturity_i,
                spread = spread_i,
                recovery = recovery,
                curve = curve,
                df = df,
                pay_freq_per_year = pay_freq_per_year,
                accrual_on_default = accrual_on_default,
            )

            return pv_prot - pv_prem

        lo = 1e-12
        hi = hazard_upper

        flo = objective(lo)
        fhi = objective(hi)

        # If not bracket expand hi a bit
        expand = 0
        while flo * hi > 0.0 and expand < 8:
            hi *= 2.0
            fhi = objective(hi)
            expand += 1
        if flo * fhi > 0.0:
            raise ValueError(
                f"Could not bracket hazard root for maturity = {maturity_i}, spread = {spread_i}."
                f"Try increasing hazard_upper or check inputs"
            )

        lam_star = _solve_monotone_bisection(objective, lo = lo, hi = hi, tol = 1e-11, max_iter = 300)
        hazards[i] = lam_star
    return PieceWiseHazardCurve(knits = mats, hazards = hazards)

def cds_bootstrap_fit_diagnostics(
    *,
    maturities: Sequence[float],
    market_spreads: Sequence[float],
    curve: PiecewiseHazardCurve,
    recovery: float,
    df: DiscountFactorFn,
    pay_freq_per_year: int = 4,
    accrual_on_default: bool = True,
) -> CDSFitDiagnostics:
    mats = np.array(maturities, dtype=float)
    mkt = np.array(market_spreads, dtype=float)
    if mats.shape != mkt.shape:
        raise ValueError("maturities and market_spreads must have same length.")
    if np.any(mats <= 0.0) or np.any(np.diff(mats) <= 0.0):
        raise ValueError("maturities must be strictly increasing and > 0.")

    mdl = np.array(
        [
            implied_par_spread(
                maturity=float(T),
                recovery=recovery,
                curve=curve,
                df=df,
                pay_freq_per_year=pay_freq_per_year,
                accrual_on_default=accrual_on_default,
            )
            for T in mats
        ],
        dtype=float,
    )

    # errors in bps
    errors_bps = (mdl - mkt) * 1e4
    max_abs = float(np.max(np.abs(errors_bps)))
    rmse = float(np.sqrt(np.mean(errors_bps**2)))

    return CDSFitDiagnostics(
        maturities=mats,
        market_spreads=mkt,
        model_spreads=mdl,
        errors_bps=errors_bps,
        max_abs_error_bps=max_abs,
        rmse_bps=rmse,
    )

def implied_par_spread(
    *,
    maturity: float,
    recorvery: float,
    curve: PieceWiseHazardCurve,
    df: DiscountFactorFn,
    pay_freq_per_year: int = 4,
    accrual_on_default: bool = True,
    ) -> float:
    """
    Compute the CDS par spread S(T) for a given maturity, hazard curve, and discount curve.

    Par spread solves:
        PV_protection(T) = S(T) * RPV01(T)

    where RPV01 is the PV of paying 1bp per year premium (including accrued-on-default if enabled).

    Returns
    -------
    spread : float
        Par spread in decimal (e.g. 0.0125 for 125 bps).

    """

    if not (0.0 <= recovery < 1.0):
        raise ValueError("recovery must be in [0,1)")
    if maturity <= 0.0:
        raise ValueError("maturity must be > 0.")

    pay_times = generate_payment_times(maturity, pay_freq_per_year=pay_freq_per_year)
    dt = 1.0/pay_freq_per_year

    # RPV01 base: sum DF(t_i) * alpha * S(t_i)
    surv = np.array([curve.survival(float(t)) for t in pay_times])
    dfs = np.array([df(float(t)) for t in pay_times])
    rpv01 = float(np.sum(dfs*dt*surv))

    if accrual_on_default:
        t_prev = 0.0
        accr = 0.0
        for t_i in pay_times:
            s_prev = curve.survival(float(t_prev))
            s_i = curve.survival(float(t_i))
            dp = s_prev - s_i
            t_mid = 0.5 * (t_prev + float(t_i))
            accr += df(t_mid) * dp
            t_prev = float(t_i)

        rpv01 += float(0.5 * dt * accr)

    if rpv01 <= 0.0:
        raise ValueError("RPV01 is non-positive; check inputs/curves")

    def integrand(x: np.ndarray) -> np.ndarray:
        knots = curve.knots
        hz = curve.hazards
        idx = np.searchsorted(knots, x, side="left")
        idx = np.clip(idx, 0, len(hz) - 1)
        lam = hz[idx]
        S = np.vectorize(curve.survival)(x)
        DF = np.vectorize(df)(x)
        return DF * lam * S

    pv_protection = float((1.0 - recovery) * _integrate_simpson(integrand, 0.0, maturity, n=60))

    spread = pv_protection / rpv01
    return float(spread)


if __name__ == "__main__":
    # Example market inputs
    curve = FlatDiscountCurve(r=0.02)
    df = curve.df

    maturities = [1, 3, 5, 7, 10]
    spreads_bps = [80, 110, 140, 155, 170]
    spreads = [x / 1e4 for x in spreads_bps]  # bps -> decimal
    recovery = 0.40

    hz_curve = bootstrap_hazard_curve_from_cds(
        maturities=maturities,
        spreads=spreads,
        recovery=recovery,
        df=df,
        pay_freq_per_year=4,
        accrual_on_default=True,
        hazard_upper=3.0,
    )

    print("Knots:", hz_curve.knots)
    print("Hazards:", hz_curve.hazards)

    for t in [0.5, 1.0, 3.0, 5.0, 10.0]:
        print(f"S({t:>4.1f}) = {hz_curve.survival(t):.6f}")
