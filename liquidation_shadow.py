"""Shadow observation of forced-liquidation order flow.

Deliberately has no strategy, ranking, or execution integration - same
design boundary as order_flow_shadow.py. Consumes Binance's combined
`!forceOrder@arr` stream (every forced-liquidation order across every
symbol on one connection, no per-symbol depth/book reconciliation needed)
and keeps a bounded rolling window of recent liquidations per symbol.

Whether heavy long-liquidation flow means "capitulation, fade it" or
"confirms the move, follow it" is genuinely unknown up front - that
ambiguity is exactly what this module exists to let real outcome data
answer, via the shadow-outcome registration in main.py, before any of
this touches a live trading decision.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

import config


LOGGER = logging.getLogger(__name__)

LIQUIDATION_STREAM_URL = "wss://fstream.binance.com/ws/!forceOrder@arr"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if result == result else default  # filters NaN
    except (TypeError, ValueError):
        return default


def _order_payload(message: Any) -> Optional[Mapping[str, Any]]:
    if not isinstance(message, Mapping):
        return None

    data = message.get("data")
    data = data if isinstance(data, Mapping) else message
    order = data.get("o")
    return order if isinstance(order, Mapping) else None


@dataclass(frozen=True)
class _LiquidationEvent:
    timestamp: float
    side: str  # "SELL" = a long position was force-closed, "BUY" = a short was
    notional: float


class LiquidationShadowMonitor:
    """Bounded, non-blocking shadow liquidation-flow tracker.

    ``handle_message`` never performs network or file I/O and never
    blocks - it only enqueues bounded work. A single worker thread drains
    the queue and updates per-symbol state under a lock; ``snapshot`` is
    safe to call from any thread.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        *,
        shutdown_event: Optional[threading.Event] = None,
        enabled: Optional[bool] = None,
        window_seconds: Optional[float] = None,
        max_events_per_symbol: Optional[int] = None,
        queue_size: Optional[int] = None,
        clock=time.time,
    ) -> None:
        self.enabled = bool(
            getattr(config, "LIQUIDATION_SHADOW_ENABLED", False)
            if enabled is None
            else enabled
        )
        self.symbols = {str(symbol).upper() for symbol in symbols or ()}
        self.shutdown_event = shutdown_event
        self.window_seconds = max(
            float(
                window_seconds
                if window_seconds is not None
                else getattr(config, "LIQUIDATION_SHADOW_WINDOW_SECONDS", 1800)
            ),
            1.0,
        )
        self.max_events_per_symbol = max(
            int(
                max_events_per_symbol
                if max_events_per_symbol is not None
                else getattr(
                    config, "LIQUIDATION_SHADOW_MAX_EVENTS_PER_SYMBOL", 500
                )
            ),
            1,
        )
        self.clock = clock
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.events: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.max_events_per_symbol)
        )
        self.event_queue: "queue.Queue[Mapping[str, Any]]" = queue.Queue(
            maxsize=max(
                int(
                    queue_size
                    if queue_size is not None
                    else getattr(config, "LIQUIDATION_SHADOW_QUEUE_SIZE", 2000)
                ),
                1,
            )
        )
        self.worker: Optional[threading.Thread] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.ws_generation = 0
        self._ws = None
        self.connected = False
        self.dropped_events = 0
        self.processing_errors = 0

    def start(self) -> bool:
        if not self.enabled:
            return False

        if self.worker and self.worker.is_alive():
            return True

        self.stop_event.clear()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="liquidation-shadow-worker",
            daemon=True,
        )
        self.worker.start()

        with self.lock:
            self.ws_generation += 1
            generation = self.ws_generation

        self.ws_thread = threading.Thread(
            target=self._stream_loop,
            args=(generation,),
            name="liquidation-shadow-stream",
            daemon=True,
        )
        self.ws_thread.start()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()

        with self.lock:
            self.ws_generation += 1
            ws = self._ws
            self._ws = None

        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

        if self.ws_thread and self.ws_thread.is_alive():
            self.ws_thread.join(max(float(timeout), 0.0))

        if self.worker and self.worker.is_alive():
            self.worker.join(max(float(timeout), 0.0))

        # Observation-only events are disposable on shutdown.
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.event_queue.task_done()

    def _active(self, generation: int) -> bool:
        if self.stop_event.is_set():
            return False

        if self.shutdown_event is not None and self.shutdown_event.is_set():
            return False

        with self.lock:
            return generation == self.ws_generation

    def _stream_loop(self, generation: int) -> None:
        try:
            from websockets.sync.client import connect
        except Exception as exc:
            LOGGER.warning("liquidation shadow websocket unavailable: %s", exc)
            return

        while self._active(generation):
            ws = None

            try:
                with connect(
                    LIQUIDATION_STREAM_URL,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as ws:
                    with self.lock:
                        if generation != self.ws_generation:
                            return
                        self._ws = ws
                        self.connected = True

                    while self._active(generation):
                        try:
                            message = ws.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self.handle_message(json.loads(message))

            except Exception as exc:
                if self._active(generation):
                    LOGGER.warning(
                        "liquidation shadow websocket reconnecting: %s", exc
                    )
                    self.stop_event.wait(3)
            finally:
                with self.lock:
                    self.connected = False
                    if self._ws is ws:
                        self._ws = None

    def handle_message(self, message: Any) -> bool:
        """Enqueue a supported liquidation order without blocking the caller."""

        if not self.enabled or self.stop_event.is_set():
            return False

        order = _order_payload(message)

        if not order:
            return False

        symbol = str(order.get("s") or "").upper()

        if self.symbols and symbol not in self.symbols:
            return False

        try:
            self.event_queue.put_nowait(order)
            return True
        except queue.Full:
            try:
                self.event_queue.get_nowait()
                self.event_queue.task_done()
            except queue.Empty:
                pass

            self.dropped_events += 1

            try:
                self.event_queue.put_nowait(order)
                return True
            except queue.Full:
                self.dropped_events += 1
                return False

    def process_pending(self, limit: Optional[int] = None) -> int:
        """Synchronously drain queued events; intended for tests/diagnostics."""

        processed = 0

        while limit is None or processed < limit:
            try:
                order = self.event_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self._process_order(order)
                processed += 1
            except Exception as exc:
                self.processing_errors += 1
                LOGGER.warning("liquidation shadow processing error: %s", exc)
            finally:
                self.event_queue.task_done()

        return processed

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set() or not self.event_queue.empty():
            try:
                order = self.event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_order(order)
            except Exception as exc:
                self.processing_errors += 1
                LOGGER.warning("liquidation shadow processing error: %s", exc)
            finally:
                self.event_queue.task_done()

    def _process_order(self, order: Mapping[str, Any]) -> None:
        symbol = str(order.get("s") or "").upper()
        side = str(order.get("S") or "").upper()

        if side not in ("BUY", "SELL"):
            return

        price = _safe_float(order.get("ap")) or _safe_float(order.get("p"))
        quantity = _safe_float(order.get("z")) or _safe_float(order.get("q"))
        notional = abs(price * quantity)

        if notional <= 0:
            return

        timestamp = (
            _safe_float(order.get("T") or order.get("E"), self.clock() * 1000)
            / 1000
        )
        event = _LiquidationEvent(timestamp=timestamp, side=side, notional=notional)

        with self.lock:
            self.events[symbol].append(event)

    def snapshot(self, symbol: str) -> dict[str, Any]:
        symbol = str(symbol or "").upper()
        now = self.clock()
        cutoff = now - self.window_seconds

        with self.lock:
            events = list(self.events.get(symbol, ()))

        recent = [event for event in events if event.timestamp >= cutoff]

        if not recent:
            return {
                "available": bool(self.enabled and self.connected),
                "sample_count": 0,
                "long_liquidation_notional": 0.0,
                "short_liquidation_notional": 0.0,
                "net_liquidation_notional": 0.0,
                "last_event_age_seconds": None,
            }

        # SELL-side forced orders close out longs; BUY-side close out shorts.
        long_liquidation_notional = sum(
            event.notional for event in recent if event.side == "SELL"
        )
        short_liquidation_notional = sum(
            event.notional for event in recent if event.side == "BUY"
        )
        last_event_age = now - max(event.timestamp for event in recent)

        return {
            "available": True,
            "sample_count": len(recent),
            "long_liquidation_notional": round(long_liquidation_notional, 2),
            "short_liquidation_notional": round(short_liquidation_notional, 2),
            "net_liquidation_notional": round(
                long_liquidation_notional - short_liquidation_notional, 2
            ),
            "last_event_age_seconds": round(last_event_age, 1),
        }
