from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Tuple, Union

Number = Union[float,int]

@dataclass(frozen=True,Slots =True)
class Dual:
    """
    Forward-mode automatic differentiation via dual numbers.

    A dual number represents: x + eps * dx where eps^2 = 0
    - val: primal value
    - der: derivative w.r.t seme seed
    """

    val: float
    der: float = 0.0

    def __add__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        return Dual(self.val + o.val, self.der + o.der)

    def __radd__(self, other: Union[Dual, Number]) -> Dual:
        return self.__add__(other)

    def __sub__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        return Dual(self.val - o.val, self.der - o.der)

    def __rsub__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        return Dual(o.val - self.val, o.der - self.der)

    def __mul__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        # (u*v)' = u'*v + u*v'
        return Dual(self.val * o.val, self.der * o.val + self.val * o.der)

    def __rmul__(self, other: Union[Dual, Number]) -> Dual:
        return self.__mul__(other)

    def __truediv__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        if o.val == 0.0:
            raise ZeroDivisionError("Dual division by zero.")
        # (u/v)' = (u'*v - u*v') / v^2
        v2 = o.val * o.val
        return Dual(self.val / o.val, (self.der * o.val - self.val * o.der) / v2)

    def __rtruediv__(self, other: Union[Dual, Number]) -> Dual:
        o = _to_dual(other)
        return o.__truediv__(self)

    def __neg__(self) -> Dual:
        return Dual(-self.val, -self.der)

    def __pow__(self, power: Union[Dual, Number]) -> Dual:
        """
        Support:
        - Dual ** float
        - Dual ** Dual (general case)
        """
        p = _to_dual(power)

        # u^v = exp(v * log(u)) (general), but requires u>0
        if self.val <= 0.0:
            # For many finance cases u>0 (spot, vol, etc.).
            # Keep this strict to avoid silent complex behavior.
            raise ValueError("Dual power with non-positive base requires complex support.")

        # Use general formula:
        # d(u^v) = u^v * (v' * ln u + v * u'/u)
        u_to_v = self.val ** p.val
        der = u_to_v * (p.der * math.log(self.val) + p.val * (self.der / self.val))
        return Dual(u_to_v, der)

def _to_dual(x: Union[Dual,Number]) -> Dual:
    return x if isinstance(x,Dual) else  Dual(float(x),0.0)


def exp(x: Union[Dual, Number]) -> Dual:
    x = _to_dual(x)
    ev = math.exp(x.val)
    return Dual(ev, ev * x.der)

def log(x: Union[Dual, Number]) -> Dual:
    x = _to_dual(x)
    if x.val <= 0.0:
        raise ValueError("log requires positive input.")
    return Dual(math.log(x.val), x.der / x.val)

def sqrt(x: Union[Dual, Number]) -> Dual:
    x = _to_dual(x)

    if x.val < 0.0:
        raise ValueError("sqrt requires non-negative input.")
    sv = math.sqrt(x.val)
    der = 0.0 if sv == 0.0 else (0.5 * x.der / sv)
    return Dual(sv, der)

def sin(x: Union[Dual, Number]) -> Dual:
    x = _to_dual(x)
    return Dual(math.sin(x.val), math.cos(x.val)*x.der)

def cos(x: Union[Dual, Number]) -> Dual:
    x = _to_dual(x)
    return Dual(math.cos(x.val), -math.sin(x.val)*x.der)

def erf(x: Union[Dual, Number]) -> Dual:
    """
    Error function used in normal cdf
    d/dx erf(x) = 2/sqrt(pi)*exp(-x^2)
    """

    x = _to_dual(x)
    v = math.erf(x.val)
    der = (2.0 / math.sqrt(math.pi)) * math.exp(-(x.val * x.val)) * x.der
    return Dual(v, der)

def normal_cdf(x: Union[Dual, Number]) -> Dual:
    """
    Standard normal CDF via erf.
    """
    x = _to_dual(x)
    return 0.5 * (Dual(1.0, 0.0) + erf(x / math.sqrt(2.0)))

def normal_pdf(x: Union[Dual, Number]) -> Dual:
    """
    Standard normal PDF.
    """
    x = _to_dual(x)
    v = (1.0 / math.sqrt(2.0 * math.pi)) * math.exp(-0.5 * x.val * x.val)
    # derivative of pdf not needed often, but keep for completeness:
    der = v * (-x.val) * x.der
    return Dual(v, der)

def grad(f: Callable[[Dual], Dual], x: float) -> float:
    """
    First derivative f'(x) for scalar->scalar function.
    """
    y = f(Dual(x, 1.0))
    return float(y.der)

def value_and_grad(f: Callable[[Dual], Dual], x: float) -> Tuple[float, float]:
    """
    Return (f(x), f'(x)).
    """
    y = f(Dual(x, 1.0))
    return float(y.val), float(y.der)

def second_derivative(f: Callable[[Dual], Dual], x: float, h: float = 1e-4) -> float:
    """
    Quick second derivative using AD for first derivative + small FD around it.

    Why not "Dual of Dual"?
    - You can implement hyper-duals, but this keeps the module small.
    - For most Greeks, you need 1st derivatives (delta/vega/rho).
    - Gamma can be computed by differentiating delta with FD at a higher level.

    This is still more stable than pricing FD when your pricer is expensive.
    """
    # d/dx f'(x) approx via central diff on AD gradient
    return (grad(f, x + h) - grad(f, x - h)) / (2.0 * h)

def bs_call_price(S: Union[Dual, Number], K: float, r: float, sigma: float, T: float) -> Dual:
    """
    Black–Scholes call price with Dual-capable spot S.

    If S is Dual seeded with der=1, output.der == Delta.
    """
    S = _to_dual(S)
    if K <= 0.0 or sigma <= 0.0 or T <= 0.0:
        raise ValueError("K, sigma, T must be positive.")

    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    return S * normal_cdf(d1) - K * math.exp(-r * T) * normal_cdf(d2)


def bs_call_greeks_via_ad(S0: float, K: float, r: float, sigma: float, T: float) -> dict:
    """
    Compute Delta and Gamma via AD (Delta exact via AD; Gamma via outer FD on AD Delta),
    plus Vega and Rho via AD seeding on those parameters as well.

    This demonstrates the main pattern:
    seed one variable at a time.
    """
    # Delta: seed S
    delta = bs_call_price(Dual(S0, 1.0), K, r, sigma, T).der

    # Gamma: derivative of delta wrt S using second_derivative helper
    def price_as_fn_of_S(S: Dual) -> Dual:
        return bs_call_price(S, K, r, sigma, T)

    gamma = second_derivative(price_as_fn_of_S, S0, h=1e-3)

    # Vega: seed sigma
    def price_as_fn_of_sigma(sig: Dual) -> Dual:
        # sigma is Dual here
        sig = _to_dual(sig)
        d1 = (math.log(S0 / K) + (r + 0.5 * sig.val * sig.val) * T) / (sig.val * math.sqrt(T))
        # For vega, easiest is to implement BS in terms of Dual sigma too:
        d1d = (log(S0 / K) + (r + 0.5 * (sig * sig)) * T) / (sig * math.sqrt(T))
        d2d = d1d - sig * math.sqrt(T)
        return _to_dual(S0) * normal_cdf(d1d) - K * math.exp(-r * T) * normal_cdf(d2d)

    vega = price_as_fn_of_sigma(Dual(sigma, 1.0)).der
