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
