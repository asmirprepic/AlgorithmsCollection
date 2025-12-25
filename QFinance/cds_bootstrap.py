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
