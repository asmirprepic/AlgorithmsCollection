
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

class OrderBook:
    """
    Price-time priority limit order book with basic matching logic.


    Supports:
     - Limit and market orders
     - Partial fills
     - FIFO at each price level
     - Cancellation by order-id
    """

    def __init__(self)  -> None:
        self._bids = Dict[float, Deque[Order]] = {}
        self._asks = Dict[float, Deque[Order]] = {}

        self._bid_heap: List[float] = []
        self._ask_heap: List[float] = []

        #Order lookup for fast cancel
        self._order_index: Dict[int, Tuple[Side, float]] = {}

        self._order_id_gen = itertools.count(1)
        self._trade_id_gen = itertools.count(1)


    def next_order_id(self) -> int:
        return next(self._order_id_gen)

    def next_trade_id(self) -> int:
        return next(self._trade_id_gen)

    def submit_order(
        self,
        side: Side,
        order_type: OrderType,
        quantity: float,
        price: Optional[float] = None,
        timestamp: Optional[float] = None,
        owner_id: Optional[str] = None,
    ) -> Tuple[Order, List[Trade]]:

        """
        Submit a new order and perform matching if possible

        Returns
        --------
        order: Order
            The order object
        trades: List[Trades]
            List of trades executed
        """

        ts = time.time() if timestamp is None else timestamp
        order_id = self.next_order_id()
        order = Order(
            order_id = order_id,
            side = side,
            type = order_type,
            price = price,
            quantity = quantity,
            timestamp = ts,
            owner_id = owner_id
        )

        trades: List[Trade] = []
        if order.type is OrderType.Market:
            trades = self._execute_market_order(order)
        else:
            trades = self._execute_limit_order(order)

        return order,trades

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an existing order by id.

        Returns:
        ------------
        success: bool
            True if the order is cancelled

        """

        info = self._order_index.get(order_id)
        if info is None:
            return False

        side, price = info
        book_side = self._bids if side is Side.BUY else self._asks

        queue = book_side.get(price)
        if queue is None:
            # Should not really happen
            self._order_index.pop(order_id, None)
            return False

        # Scan
        found = False
        new_queue = Deque[Order] = deque()
        while queue:
            o = queue.popleft()
            if order_id == order_id and not found:
                found = True

                # Do not insert the order
                self._order_index.pop(order_id,None)
            else: new_queue.append(o)

        if found:
            if new_queue:
                book_side[price] = new_queue
            else:
                del book_side[price]
            return True
        else:
            # Not found restore queues
            book_side[price] = new_queue
            self._order_index.pop(order_id,None)
            return False

    def best_bid(self ) -> Optional[BookLevel]:
        price = self._best_bid_price()
        if price is None:
            return None
        queue = self._bids[price]
        qty = sum(o.quantity for o in queue)
        return BookLevel(price = price, quantity=qty, order_count = len(queue))

    def best_ask(self) -> Optional[BookLevel]:
        price = self._best_ask_price()
        if price is None:
            return None
        queue = self._asks[price]
        qty = sum(o.quantity for o in queue)
        return BookLevel(price=price, quantity=qty, order_count=len(queue))

    def snapshot(self) -> BookSnapshot:
        """
        Return a top  of the book snapshot
        """
        ts = time.time()
        return BookSnapshot(timestamp=ts, best_bid = self.best_bid(), best_ask = self.best_ask())

    def depth(
        self,
        levels: int = 5,
    ) -> Tuple[List[BookLevel], List[BookLevel]]:
        """
        Return book depth for both sides.

        Parameters
        ----------
        levels : int
            Maximum number of levels per side.

        Returns
        -------
        bids : list[BookLevel]
        asks : list[BookLevel]
        """
        bids = self._sorted_levels(self._bids, descending=True, max_levels=levels)
        asks = self._sorted_levels(self._asks, descending=False, max_levels=levels)
        return bids, asks

    def _sorted_levels(
        self, book_side: Dict[float, Deque[Order]],
        *,
        descending: bool,
        max_levels: int

    ) -> List[BookLevel]:
        prices = sorted(book_side.keys(), reverse=descending)
        levels = List[BookLevel] = []
        for p in prices[:max_levels]:
            q = book_side[p]
            qty = sum(o.quantity for o in q)
            levels.append(BookLevel(price=p, quantity=qty,order_count= len(q)))

        return levels

    def _best_bid_price(self) -> Optional[float]:
        while self._bid_heap:
            p = -self._bid_heap[0]
            if p in self._bids and self._bids[p]:
                return p
            heapq.heappop(self._bid_heap)
        return None

    def _best_ask_price(self) -> Optional[float]:
        while self._ask_heap:
            p = self._ask_heap[0]
            if p in self._asks and self._asks[p]:
                return p
            heapq.heappop(self._ask_heap)
        return None

class PoissonOrderFlowSimulator:
    """
        Simple Poisson order-flow simulator driving an OrderBook.

        - Arrivals of limit orders, market orders, and cancellations follow
        independent Poisson processes with given intensities.
        - Limit order prices are sampled around the current mid (or config.mid_price)
        within a configurable depth.
        - Order sizes follow an exponential distribution.

        This is not calibrated to any particular market; it's a toy but structurally
        realistic environment to test strategies and market-making logic.
    """

    def __init__(self, book: Optional[OrderBook] = None, config: Optional[OrderFlowConfig] = None):
        self.book = book if book is not None else OrderBook()
        self.config = config if config is not None else OrderFlowConfig()
        self._rng = np.random.default_rng(self.config.seed)
        self._time: float = 0.0

    @property
    def time(self) -> float:
        return self._time

    def _sample_interarrival(self) -> float:
        lam_total = (
            self.config.lambda_limit
            + self.config.lambda_market
            + self.config.lambda_cancel
        )

        return float(self._rng.exponential(1.0/lam_total))

    def sample_event_type(self) -> str:
        lam = np.array(
            [self.config.lambda_limit,
            self.config.lambda_market,
            self.config.lambda_cancel],
            dtype= float,
        )

        probs = lam/lam.sum()
        idx = int(self._rng.choice(3,p = probs))
        return ["limit", "market", "cancel"][idx]

    def _sample_side(self) -> Side:
        return Side.BUY if self._rng.random() < 0.5 else Side.SELL

    def _sample_size(self) -> float:
        return float(self._rng.exponential(self.config.mean_size))

    def _current_mid(self) -> float:
        snap = self.book.snapshot()
        if snap.best_bid and snap.best_ask:
            return 0.5 * (snap.best_bid.price + snap.best_ask.price)
        return self.config.mid_price

    def _sample_limit_price(self, side: Side) -> float:
        mid = self._current_mid()
        ticks_offset = self._rng.integers(1, self.config.max_depth_ticks + 1)
        offset = ticks_offset * self.config.tick_size

        if side is Side.BUY:
            raw_price = mid - offset
        else:
            raw_price = mid + offset

        # Round to tick grid and ensure positive
        price = max(self.config.tick_size, math.floor(raw_price / self.config.tick_size) * self.config.tick_size)
        return float(price)

    def _random_resting_order_id(self) -> Optional[int]:
        """
        Uniformly sample an existing resting order to cancel, if any
        """
        if not self.book._order_index:
            return None
        return int(self._rng.choice(list(self.book_order_index.keys())))

    def step(self) -> SimulationEvent:
        """
        Advance the simulation by one event

        Returns:
        ----------
        event: SimulationEvent
        """
        dt = self._sample_interarrival()
        self._time += dt

        etype = self.sample_event_type()
        side: Optional[Side]  = None
        trades: List[Trade] = []
        order_id: Optional[int] = None

        if etype == "limit":
            side = self._sample_side()
            price = self._sample_limit_price(side)
            qty = self._sample_size()
            order, trades = self.book.submit_order(
                side=side,
                order_type=OrderType.LIMIT,
                quantity=qty,
                price=price,
                timestamp=self._time,
            )
            order_id = order.order_id
        elif etype == "market":
            side = self._sample_side()
            qty = self._sample_size()
            order, trades = self.book.submit_order(
                side=side,
                order_type=OrderType.MARKET,
                quantity=qty,
                timestamp=self._time,
            )
            order_id = order.order_id
        else:  # cancel
            order_id = self._random_resting_order_id()
            side = None
            if order_id is not None:
                self.book.cancel_order(order_id)

        return SimulationEvent(
            time=self._time,
            event_type=etype,
            side=side,
            order_id=order_id,
            trades=trades,
        )

    def run(self, n_events: int) -> Iterable[SimulationEvent]:
        """
        Run the simulator for a fixed number of events.

        Yields
        ------
        event : SimulationEvent
        """
        for _ in range(n_events):
            yield self.step()
