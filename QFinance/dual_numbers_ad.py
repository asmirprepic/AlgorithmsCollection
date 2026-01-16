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

    def __add__(self, ohter: Union[Dual, Number]) -> Dual:
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
