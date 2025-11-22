
import heapq
import itertools
import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Deque, Dict, Iterable, List, Optional, Tuple
from __future__ import annotations

import numpy as np


class Side(Enum):
    BUY = auto()
    SELL = auto()

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

class OrderType(Enum):
    LIMIT = auto()
    MARKET = auto()


@dataclass(slots = True)

class Order:
    class Order:
        """
        Representation of a single order in the LOB.

        Attributes
        ----------
        order_id : int
            Unique identifier of this order within the engine.
        side : Side
            BUY or SELL.
        type : OrderType
            LIMIT or MARKET.
        price : Optional[float]
            Limit price. Must be positive for LIMIT orders. None for MARKET.
        quantity : float
            Remaining quantity. Positive.
        timestamp : float
            Time in seconds since epoch (or simulation clock).
        owner_id : Optional[str]
            Optional identifier for the agent/trader that owns this order.
        """

        order_id: int
        side: Side
        type: OrderType
        price: Optional[float]
        quantity: float
        timestamp: float
        owner_id: Optional[str] = None

        def __post_init__(self) -> None:
            if self.type is OrderType.LIMIT:
                if self.price is None or self.price <= 0:
                    raise ValueError("Limit orders require positive price.")
            else:
                if self.price is not None:
                    raise ValueError("Market orders must have price=None.")
            if self.quantity <= 0:
                raise ValueError("Order quantity must be positive.")

@dataclass(slots=True)
class Trade:
    """
        Execution resulting from matching two orders.

        Attributes
        ----------
        trade_id : int
            Unique id of the trade within the engine.
        price : float
            Execution price.
        quantity : float
            Executed quantity.
        timestamp : float
            Execution timestamp.
        buy_order_id : int
            Id of the passive or aggressive buy order.
        sell_order_id : int
            Id of the passive or aggressive sell order.
    """

    trade_id: int
    price: float
    quantity: float
    timestamp: float
    buy_order_id: int
    sell_order_id: int


@dataclass(slots=True)
class BookLevel:
    """
    Snapshot of a price level in the LOB.
    """
    price: float
    quantity: float
    order_count: int

@dataclass(slots=True)
class BookSnapshot:
    """
    Lightweight top-of-book snapshot.

    Attributes
    ----------
    timestamp : float
        Snapshot timestamp.
    best_bid : Optional[BookLevel]
    best_ask : Optional[BookLevel]
    """

    timestamp: float
    best_bid: Optional[BookLevel]
    best_ask: Optional[BookLevel]
