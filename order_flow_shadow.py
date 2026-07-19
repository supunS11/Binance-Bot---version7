"""Independent full-depth and footprint analytics for shadow observation.

This module deliberately has no strategy, ranking, or execution integration.  Market
data callbacks only enqueue bounded work; an internal worker performs snapshot
synchronisation and analytics state updates.  Consumers may inspect ``snapshot()``
for telemetry, but the returned payload explicitly declares that it has no decision
or ranking effect.

``snapshot_provider`` is an injected callable with this contract::

    snapshot_provider(symbol, limit=1000) -> {
        "lastUpdateId": 123,
        "bids": [["100.0", "2.0"], ...],
        "asks": [["100.1", "1.5"], ...],
    }

The monitor owns observation-only diff-depth websocket threads. Trade messages are
fed through :meth:`OrderFlowShadowMonitor.handle_message`; the injected provider
owns REST authentication, rate limiting, and snapshot policy.
"""

from __future__ import annotations

import csv
import json
import logging
import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import config


LOGGER = logging.getLogger(__name__)


FUTURES_PUBLIC_STREAM_BASE = "wss://fstream.binance.com/public/stream?streams="


TELEMETRY_FIELDS = (
    "timestamp_utc",
    "symbol",
    "signal_side",
    "route",
    "rank_score",
    "available",
    "book_synced",
    "shadow_score",
    "cvd_notional",
    "cvd_ratio",
    "cumulative_cvd_notional",
    "footprint_delta_ratio",
    "buy_imbalance_buckets",
    "sell_imbalance_buckets",
    "full_depth_imbalance",
    "microprice_bps",
    "impact_skew",
    "buy_impact_bps",
    "sell_impact_bps",
    "spread_bps",
    "trade_count",
    "bid_levels",
    "ask_levels",
    "last_update_id",
    "sequence_gaps",
    "resync_count",
    "snapshot_errors",
    "dropped_events",
    "trade_age_seconds",
    "depth_age_seconds",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _clip(value: float, minimum: float = -1.0, maximum: float = 1.0) -> float:
    return min(max(float(value), minimum), maximum)


def _message_data(message: Any) -> dict[str, Any]:
    if not isinstance(message, Mapping):
        return {}

    data = message.get("data")
    return dict(data) if isinstance(data, Mapping) else dict(message)


def _event_timestamp_seconds(data: Mapping[str, Any], fallback: float) -> float:
    raw = _safe_float(data.get("T") or data.get("E"), fallback * 1000)
    return raw / 1000 if raw > 10_000_000_000 else raw


def _normalise_levels(levels: Any) -> dict[float, float]:
    result: dict[float, float] = {}

    for level in levels or ():
        if not isinstance(level, (list, tuple)) or len(level) < 2:
            continue

        price = _safe_float(level[0])
        quantity = _safe_float(level[1])

        if price > 0 and quantity > 0:
            result[price] = quantity

    return result


@dataclass(frozen=True)
class _Trade:
    timestamp: float
    price: float
    quantity: float
    notional: float
    signed_notional: float
    aggressive_buy: bool


@dataclass
class _SymbolState:
    max_trades: int
    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)
    trades: deque[_Trade] = field(init=False)
    seen_trade_ids: deque[tuple[float, str]] = field(init=False)
    seen_trade_id_set: set[str] = field(default_factory=set)
    last_update_id: int = 0
    snapshot_update_id: int = 0
    book_synced: bool = False
    awaiting_bridge: bool = True
    depth_updated_at: float = 0.0
    trade_updated_at: float = 0.0
    cumulative_cvd_notional: float = 0.0
    cumulative_buy_notional: float = 0.0
    cumulative_sell_notional: float = 0.0
    depth_updates: int = 0
    trade_updates: int = 0
    stale_depth_events: int = 0
    invalid_depth_events: int = 0
    sequence_gaps: int = 0
    resync_count: int = 0
    snapshot_errors: int = 0
    book_trim_count: int = 0
    duplicate_trades: int = 0
    dropped_events: int = 0
    last_error: str = ""
    next_snapshot_attempt_at: float = 0.0

    def __post_init__(self) -> None:
        self.trades = deque(maxlen=max(int(self.max_trades), 1))
        self.seen_trade_ids = deque(maxlen=max(int(self.max_trades), 1))


class _ShadowCsvTelemetry:
    """Bounded asynchronous CSV writer used only for shadow observability."""

    def __init__(
        self,
        enabled: bool,
        path: str,
        min_interval_seconds: float,
        queue_size: int,
        clock: Callable[[], float],
    ) -> None:
        self.enabled = bool(enabled)
        configured_path = Path(path).expanduser()

        if not configured_path.is_absolute():
            configured_path = Path(__file__).resolve().parent / configured_path

        self.path = configured_path
        self.min_interval_seconds = max(float(min_interval_seconds), 0.0)
        self.clock = clock
        self.queue: queue.Queue[dict[str, Any]] = queue.Queue(
            maxsize=max(min(int(queue_size), 4096), 1)
        )
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_emit_at: dict[str, float] = {}
        self.dropped_rows = 0
        self.write_errors = 0

    def start(self) -> bool:
        if not self.enabled:
            return False

        if self.thread and self.thread.is_alive():
            return True

        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self._run,
            name="order-flow-shadow-telemetry",
            daemon=True,
        )
        self.thread.start()
        return True

    def submit(self, payload: Mapping[str, Any]) -> bool:
        if not self.enabled:
            return False

        symbol = str(payload.get("symbol") or "").upper()
        now = self.clock()

        with self.lock:
            last_emit = self.last_emit_at.get(symbol, 0.0)

            if last_emit and now - last_emit < self.min_interval_seconds:
                return False

            self.last_emit_at[symbol] = now

        row = {field: payload.get(field) for field in TELEMETRY_FIELDS}
        row["timestamp_utc"] = datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        )

        try:
            self.queue.put_nowait(row)
            return True
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except queue.Empty:
                pass

            self.dropped_rows += 1

            try:
                self.queue.put_nowait(row)
                return True
            except queue.Full:
                self.dropped_rows += 1
                return False

    def flush_pending(self, limit: Optional[int] = None) -> int:
        written = 0

        while limit is None or written < limit:
            try:
                row = self.queue.get_nowait()
            except queue.Empty:
                break

            try:
                self._write_row(row)
                written += 1
            finally:
                self.queue.task_done()

        return written

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()

        if self.thread and self.thread.is_alive():
            self.thread.join(max(float(timeout), 0.0))

        self.flush_pending()

    def _run(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                row = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._write_row(row)
            finally:
                self.queue.task_done()

    def _write_row(self, row: Mapping[str, Any]) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)

            with self.lock:
                needs_header = not self.path.exists() or self.path.stat().st_size == 0

                with self.path.open("a", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=TELEMETRY_FIELDS)

                    if needs_header:
                        writer.writeheader()

                    writer.writerow({field: row.get(field) for field in TELEMETRY_FIELDS})

        except Exception as exc:  # Telemetry must never affect analytics.
            self.write_errors += 1
            LOGGER.warning("order-flow shadow telemetry write failed: %s", exc)


class OrderFlowShadowMonitor:
    """Bounded, non-blocking shadow order-flow state machine.

    ``handle_message`` never performs network or file I/O.  Snapshot calls happen
    only in the worker (or explicit ``process_pending`` calls in offline tests), and
    telemetry is placed on its own bounded queue.
    """

    def __init__(
        self,
        symbols: Iterable[str],
        snapshot_provider: Optional[Callable[..., Mapping[str, Any]]] = None,
        *,
        shutdown_event: Optional[threading.Event] = None,
        enabled: Optional[bool] = None,
        max_symbols: Optional[int] = None,
        queue_size: Optional[int] = None,
        max_book_levels: Optional[int] = None,
        max_trades_per_symbol: Optional[int] = None,
        window_seconds: Optional[int] = None,
        bucket_bps: Optional[float] = None,
        min_bucket_notional: Optional[float] = None,
        imbalance_ratio: Optional[float] = None,
        stale_seconds: Optional[float] = None,
        impact_notional_usdt: Optional[float] = None,
        telemetry_enabled: Optional[bool] = None,
        telemetry_path: Optional[str] = None,
        telemetry_min_interval_seconds: Optional[float] = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.enabled = bool(
            getattr(config, "ORDER_FLOW_SHADOW_ENABLED", False)
            if enabled is None
            else enabled
        )
        configured_symbols = tuple(
            str(item).upper()
            for item in getattr(config, "ORDER_FLOW_SHADOW_SYMBOLS", ())
            if str(item).strip()
        )
        candidates = configured_symbols or tuple(str(item).upper() for item in symbols)
        unique_symbols = tuple(dict.fromkeys(item for item in candidates if item))
        symbol_limit = max(
            int(
                getattr(config, "ORDER_FLOW_SHADOW_MAX_SYMBOLS", 12)
                if max_symbols is None
                else max_symbols
            ),
            0,
        )
        self.symbols = unique_symbols[:symbol_limit] if symbol_limit else ()
        self.snapshot_provider = snapshot_provider
        self.shutdown_event = shutdown_event
        self.clock = clock
        self.snapshot_limit = max(
            int(
                getattr(config, "ORDER_FLOW_SHADOW_DEPTH_SNAPSHOT_LIMIT", 1000)
                if max_book_levels is None
                else max_book_levels
            ),
            1,
        )
        self.max_trades = max(
            int(
                getattr(config, "ORDER_FLOW_SHADOW_MAX_TRADES_PER_SYMBOL", 50000)
                if max_trades_per_symbol is None
                else max_trades_per_symbol
            ),
            1,
        )
        self.window_seconds = max(
            int(
                getattr(config, "ORDER_FLOW_SHADOW_WINDOW_SECONDS", 300)
                if window_seconds is None
                else window_seconds
            ),
            1,
        )
        self.bucket_bps = max(
            float(
                getattr(config, "ORDER_FLOW_SHADOW_BUCKET_BPS", 1.0)
                if bucket_bps is None
                else bucket_bps
            ),
            0.0001,
        )
        self.min_bucket_notional = max(
            float(
                getattr(config, "ORDER_FLOW_SHADOW_MIN_BUCKET_NOTIONAL", 5000)
                if min_bucket_notional is None
                else min_bucket_notional
            ),
            0.0,
        )
        self.imbalance_ratio = max(
            float(
                getattr(config, "ORDER_FLOW_SHADOW_IMBALANCE_RATIO", 3.0)
                if imbalance_ratio is None
                else imbalance_ratio
            ),
            1.0,
        )
        self.stale_seconds = max(
            float(
                getattr(config, "ORDER_FLOW_SHADOW_STALE_SECONDS", 15)
                if stale_seconds is None
                else stale_seconds
            ),
            0.1,
        )
        self.impact_notional_usdt = max(
            float(
                getattr(config, "ORDER_FLOW_SHADOW_IMPACT_NOTIONAL_USDT", 10000)
                if impact_notional_usdt is None
                else impact_notional_usdt
            ),
            0.0,
        )
        event_queue_size = max(
            int(
                getattr(config, "ORDER_FLOW_SHADOW_QUEUE_SIZE", 20000)
                if queue_size is None
                else queue_size
            ),
            1,
        )
        self.event_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue(
            maxsize=event_queue_size
        )
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.worker: Optional[threading.Thread] = None
        self.depth_threads: list[threading.Thread] = []
        self.depth_websockets: dict[int, Any] = {}
        self.depth_generation = 0
        self.processing_errors = 0
        self.dropped_events = 0
        self.states = {
            symbol: _SymbolState(max_trades=self.max_trades)
            for symbol in self.symbols
        }
        self.telemetry = _ShadowCsvTelemetry(
            enabled=(
                bool(getattr(config, "ORDER_FLOW_SHADOW_TELEMETRY_ENABLED", False))
                if telemetry_enabled is None
                else bool(telemetry_enabled)
            ),
            path=(
                str(
                    getattr(
                        config,
                        "ORDER_FLOW_SHADOW_TELEMETRY_PATH",
                        "data/order_flow_shadow_v7.csv",
                    )
                )
                if telemetry_path is None
                else telemetry_path
            ),
            min_interval_seconds=(
                float(
                    getattr(
                        config,
                        "ORDER_FLOW_SHADOW_TELEMETRY_MIN_INTERVAL_SECONDS",
                        30,
                    )
                )
                if telemetry_min_interval_seconds is None
                else telemetry_min_interval_seconds
            ),
            queue_size=max(event_queue_size // 4, 1),
            clock=clock,
        )

    def start(self) -> bool:
        if not self.enabled or not self.symbols:
            return False

        if self.worker and self.worker.is_alive():
            return True

        self.stop_event.clear()
        self.telemetry.start()
        self.worker = threading.Thread(
            target=self._worker_loop,
            name="order-flow-shadow-worker",
            daemon=True,
        )
        self.worker.start()
        self._start_depth_streams()
        return True

    def stop(self, timeout: float = 2.0) -> None:
        self.stop_event.set()

        with self.lock:
            self.depth_generation += 1
            websockets = list(self.depth_websockets.values())
            self.depth_websockets = {}

        for websocket in websockets:
            try:
                websocket.close()
            except Exception:
                pass

        if self.worker and self.worker.is_alive():
            self.worker.join(max(float(timeout), 0.0))

        # Observation-only events are disposable on shutdown. Do not let a
        # large queue continue CPU/REST work after the bot has stopped.
        while True:
            try:
                self.event_queue.get_nowait()
            except queue.Empty:
                break
            else:
                self.event_queue.task_done()

        for thread in list(self.depth_threads):
            if thread.is_alive():
                thread.join(min(max(float(timeout), 0.0), 1.0))

        self.telemetry.stop(timeout=timeout)

    def _start_depth_streams(self) -> None:
        try:
            from websockets.sync.client import connect  # noqa: F401
        except Exception as exc:
            LOGGER.warning("order-flow shadow websocket unavailable: %s", exc)
            return

        with self.lock:
            self.depth_generation += 1
            generation = self.depth_generation

        threads = []

        for index, streams in enumerate(self.depth_stream_chunks(), start=1):
            thread = threading.Thread(
                target=self._depth_stream_loop,
                args=(streams, generation),
                name=f"order-flow-shadow-depth-{index}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with self.lock:
            if generation == self.depth_generation:
                self.depth_threads = threads

    def _depth_worker_active(self, generation: int) -> bool:
        if self.stop_event.is_set():
            return False

        if self.shutdown_event is not None and self.shutdown_event.is_set():
            return False

        with self.lock:
            return generation == self.depth_generation

    def _depth_stream_loop(
        self,
        streams: tuple[str, ...],
        generation: int,
    ) -> None:
        from websockets.sync.client import connect

        url = FUTURES_PUBLIC_STREAM_BASE + "/".join(streams)
        worker_id = threading.get_ident()

        while self._depth_worker_active(generation):
            websocket = None

            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.depth_generation:
                            return
                        self.depth_websockets[worker_id] = websocket

                    while self._depth_worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self.handle_message(json.loads(message))

            except Exception as exc:
                if self._depth_worker_active(generation):
                    LOGGER.warning(
                        "order-flow shadow depth websocket reconnecting: %s",
                        exc,
                    )
                    self.stop_event.wait(3)
            finally:
                with self.lock:
                    current = self.depth_websockets.get(worker_id)

                    if current is websocket:
                        self.depth_websockets.pop(worker_id, None)

    def handle_message(self, message: Any) -> bool:
        """Enqueue a supported Binance message without blocking the caller."""

        if not self.enabled or self.stop_event.is_set():
            return False

        data = _message_data(message)
        symbol = str(data.get("s") or "").upper()

        if symbol not in self.states:
            return False

        event_type = str(data.get("e") or "")

        if event_type == "depthUpdate":
            kind = "depth"
        elif event_type in {"aggTrade", "trade"}:
            kind = "trade"
        else:
            return False

        return self._offer_event(kind, data)

    enqueue_message = handle_message

    def process_pending(self, limit: Optional[int] = None) -> int:
        """Synchronously drain queued events; intended for tests and diagnostics."""

        processed = 0

        while limit is None or processed < limit:
            try:
                kind, data = self.event_queue.get_nowait()
            except queue.Empty:
                break

            try:
                self._process_event(kind, data)
                processed += 1
            except Exception as exc:  # Shadow failures never escape to integration.
                self.processing_errors += 1
                LOGGER.warning("order-flow shadow processing failed: %s", exc)
            finally:
                self.event_queue.task_done()

        return processed

    def flush_telemetry(self, limit: Optional[int] = None) -> int:
        return self.telemetry.flush_pending(limit=limit)

    def depth_stream_chunks(self) -> tuple[tuple[str, ...], ...]:
        """Return configured Binance diff-depth stream names for an owner to subscribe."""

        configured_update_ms = int(
            getattr(config, "ORDER_FLOW_SHADOW_DEPTH_UPDATE_MS", 100)
        )
        update_ms = (
            configured_update_ms
            if configured_update_ms in {100, 500}
            else 100
        )
        chunk_size = max(
            int(
                getattr(
                    config,
                    "ORDER_FLOW_SHADOW_DEPTH_STREAMS_PER_SOCKET",
                    12,
                )
            ),
            1,
        )
        streams = tuple(
            f"{symbol.lower()}@depth@{update_ms}ms"
            for symbol in self.symbols
        )
        return tuple(
            streams[index:index + chunk_size]
            for index in range(0, len(streams), chunk_size)
        )

    def health(self) -> dict[str, Any]:
        with self.lock:
            return {
                "shadow_only": True,
                "enabled": self.enabled,
                "running": bool(self.worker and self.worker.is_alive()),
                "symbols": len(self.symbols),
                "depth_threads_alive": sum(
                    thread.is_alive() for thread in self.depth_threads
                ),
                "depth_sockets_connected": len(self.depth_websockets),
                "queue_size": self.event_queue.qsize(),
                "queue_capacity": self.event_queue.maxsize,
                "dropped_events": self.dropped_events,
                "processing_errors": self.processing_errors,
                "telemetry_dropped_rows": self.telemetry.dropped_rows,
                "telemetry_write_errors": self.telemetry.write_errors,
            }

    def snapshot(
        self,
        symbol: str,
        *,
        emit_telemetry: bool = True,
        context: Optional[Mapping[str, Any]] = None,
    ) -> dict[str, Any]:
        symbol = str(symbol or "").upper()
        now = self.clock()

        with self.lock:
            state = self.states.get(symbol)

            if not state:
                return self._unavailable(symbol, "SYMBOL_NOT_WATCHED")

            bids = dict(state.bids)
            asks = dict(state.asks)
            trades = [
                trade
                for trade in state.trades
                if trade.timestamp >= now - self.window_seconds
            ]
            state_values = {
                "book_synced": state.book_synced,
                "last_update_id": state.last_update_id,
                "snapshot_update_id": state.snapshot_update_id,
                "depth_updated_at": state.depth_updated_at,
                "trade_updated_at": state.trade_updated_at,
                "cumulative_cvd_notional": state.cumulative_cvd_notional,
                "cumulative_buy_notional": state.cumulative_buy_notional,
                "cumulative_sell_notional": state.cumulative_sell_notional,
                "depth_updates": state.depth_updates,
                "trade_updates": state.trade_updates,
                "stale_depth_events": state.stale_depth_events,
                "invalid_depth_events": state.invalid_depth_events,
                "sequence_gaps": state.sequence_gaps,
                "resync_count": state.resync_count,
                "snapshot_errors": state.snapshot_errors,
                "book_trim_count": state.book_trim_count,
                "duplicate_trades": state.duplicate_trades,
                "dropped_events": state.dropped_events,
                "last_error": state.last_error,
            }

        depth_age = (
            now - state_values["depth_updated_at"]
            if state_values["depth_updated_at"]
            else None
        )
        trade_age = (
            now - state_values["trade_updated_at"]
            if state_values["trade_updated_at"]
            else None
        )
        book_fresh = bool(
            state_values["book_synced"]
            and depth_age is not None
            and depth_age <= self.stale_seconds
        )
        selected_trades = [
            trade
            for trade in trades
            if trade.timestamp >= now - self.window_seconds
        ]
        trade_fresh = bool(
            selected_trades
            and trade_age is not None
            and trade_age <= self.stale_seconds
        )
        book_metrics = self._book_metrics(bids, asks) if book_fresh else {}
        trade_metrics = self._trade_metrics(
            selected_trades,
            book_metrics.get("mid_price", 0.0),
        )
        components: list[tuple[float, float]] = []

        if trade_fresh:
            components.extend((
                (trade_metrics["cvd_ratio"], 0.35),
                (trade_metrics["footprint_delta_ratio"], 0.10),
                (trade_metrics["footprint_imbalance"], 0.10),
            ))

        if book_fresh:
            spread_scale = max(book_metrics.get("spread_bps", 0.0), 1.0)
            components.extend((
                (book_metrics.get("full_depth_imbalance", 0.0), 0.25),
                (_clip(book_metrics.get("microprice_bps", 0.0) / spread_scale), 0.10),
                (book_metrics.get("impact_skew", 0.0), 0.10),
            ))

        total_weight = sum(weight for _, weight in components)
        directional = (
            sum(_clip(value) * weight for value, weight in components) / total_weight
            if total_weight > 0
            else 0.0
        )
        score = round(_clip(directional) * 5, 4)
        available = bool(book_fresh or trade_fresh)
        reason = "SHADOW_DATA_AVAILABLE" if available else "SHADOW_DATA_STALE_OR_UNSYNCED"
        payload = {
            "shadow_only": True,
            "decision_effect": False,
            "ranking_effect": False,
            "available": available,
            "reason": reason,
            "symbol": symbol,
            "shadow_score": score,
            "buy_shadow_score": score,
            "sell_shadow_score": round(-score, 4),
            "book_synced": bool(state_values["book_synced"]),
            "book_fresh": book_fresh,
            "trade_fresh": trade_fresh,
            "depth_age_seconds": None if depth_age is None else round(depth_age, 3),
            "trade_age_seconds": None if trade_age is None else round(trade_age, 3),
            **book_metrics,
            **trade_metrics,
            **{
                key: value
                for key, value in state_values.items()
                if key not in {"depth_updated_at", "trade_updated_at"}
            },
            "queue_size": self.event_queue.qsize(),
            "global_dropped_events": self.dropped_events,
        }
        context = context or {}
        payload.update({
            "signal_side": str(context.get("signal_side") or "").upper(),
            "route": str(context.get("route") or ""),
            "rank_score": context.get("rank_score"),
        })

        if emit_telemetry:
            self.telemetry.submit(payload)

        return payload

    def _offer_event(self, kind: str, data: dict[str, Any]) -> bool:
        event = (kind, dict(data))

        try:
            self.event_queue.put_nowait(event)
            return True
        except queue.Full:
            dropped_symbol = ""

            try:
                _, dropped = self.event_queue.get_nowait()
                dropped_symbol = str(dropped.get("s") or "").upper()
                self.event_queue.task_done()
            except queue.Empty:
                pass

            with self.lock:
                self.dropped_events += 1

                if dropped_symbol in self.states:
                    self.states[dropped_symbol].dropped_events += 1

            try:
                self.event_queue.put_nowait(event)
                return True
            except queue.Full:
                with self.lock:
                    self.dropped_events += 1
                    symbol = str(data.get("s") or "").upper()

                    if symbol in self.states:
                        self.states[symbol].dropped_events += 1

                return False

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                kind, data = self.event_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._process_event(kind, data)
            except Exception as exc:  # Never propagate shadow failures.
                self.processing_errors += 1
                LOGGER.warning("order-flow shadow worker failed: %s", exc)
            finally:
                self.event_queue.task_done()

    def _process_event(self, kind: str, data: Mapping[str, Any]) -> None:
        if kind == "depth":
            self._process_depth(data)
        elif kind == "trade":
            self._process_trade(data)

    def _process_depth(self, data: Mapping[str, Any]) -> None:
        symbol = str(data.get("s") or "").upper()
        first_update_id = _safe_int(data.get("U"), -1)
        final_update_id = _safe_int(data.get("u"), -1)
        previous_final_id = (
            _safe_int(data.get("pu"), -1)
            if data.get("pu") not in (None, "")
            else None
        )

        if (
            symbol not in self.states
            or first_update_id < 0
            or final_update_id < first_update_id
        ):
            with self.lock:
                if symbol in self.states:
                    self.states[symbol].invalid_depth_events += 1
            return

        needs_snapshot = False

        with self.lock:
            state = self.states[symbol]

            if not state.book_synced:
                if state.bids and state.asks and state.awaiting_bridge:
                    if final_update_id < state.last_update_id:
                        state.stale_depth_events += 1
                        return

                    expected_id = state.last_update_id

                    if first_update_id <= expected_id <= final_update_id:
                        self._apply_depth_levels_locked(state, data)
                        state.last_update_id = final_update_id
                        state.book_synced = True
                        state.awaiting_bridge = False
                        state.depth_updates += 1
                        state.depth_updated_at = self.clock()
                        state.last_error = ""
                        return

                    state.sequence_gaps += 1
                    state.last_error = (
                        "DEPTH_BRIDGE_GAP "
                        f"LAST={state.last_update_id} U={first_update_id} "
                        f"u={final_update_id}"
                    )

                needs_snapshot = True
            elif final_update_id <= state.last_update_id:
                state.stale_depth_events += 1
                return
            else:
                expected_id = state.last_update_id + 1
                range_valid = first_update_id <= expected_id <= final_update_id
                previous_valid = (
                    previous_final_id == state.last_update_id
                    if previous_final_id is not None
                    else range_valid
                )

                if not range_valid or not previous_valid:
                    state.sequence_gaps += 1
                    state.resync_count += 1
                    state.book_synced = False
                    state.awaiting_bridge = True
                    state.last_error = (
                        "DEPTH_SEQUENCE_GAP "
                        f"LAST={state.last_update_id} U={first_update_id} "
                        f"u={final_update_id} pu={previous_final_id}"
                    )
                    needs_snapshot = True

        if needs_snapshot:
            self._synchronise_book(symbol, data)
            return

        with self.lock:
            state = self.states[symbol]
            self._apply_depth_levels_locked(state, data)
            state.last_update_id = final_update_id
            state.depth_updates += 1
            state.depth_updated_at = self.clock()
            state.last_error = ""

    def _synchronise_book(self, symbol: str, event: Mapping[str, Any]) -> None:
        if self.stop_event.is_set():
            return

        now = self.clock()
        retry_seconds = max(
            float(
                getattr(
                    config,
                    "ORDER_FLOW_SHADOW_SNAPSHOT_RETRY_SECONDS",
                    10,
                )
            ),
            0.1,
        )

        with self.lock:
            state = self.states[symbol]

            if now < state.next_snapshot_attempt_at:
                return

            state.next_snapshot_attempt_at = now + retry_seconds

        try:
            snapshot = self._request_snapshot(symbol)
            snapshot_id = _safe_int(snapshot.get("lastUpdateId"), -1)

            if snapshot_id < 0:
                raise ValueError("snapshot lastUpdateId missing")

            bids = _normalise_levels(snapshot.get("bids"))
            asks = _normalise_levels(snapshot.get("asks"))

            if not bids or not asks:
                raise ValueError("snapshot book is empty")

        except Exception as exc:
            with self.lock:
                state = self.states[symbol]
                state.snapshot_errors += 1
                state.book_synced = False
                state.awaiting_bridge = True
                state.last_error = f"SNAPSHOT_ERROR: {exc}"
            return

        first_update_id = _safe_int(event.get("U"), -1)
        final_update_id = _safe_int(event.get("u"), -1)

        with self.lock:
            state = self.states[symbol]
            state.bids = bids
            state.asks = asks
            state.snapshot_update_id = snapshot_id
            state.last_update_id = snapshot_id
            state.book_synced = False
            state.awaiting_bridge = True
            state.depth_updated_at = self.clock()
            self._trim_book_locked(state)

            if final_update_id < snapshot_id:
                state.stale_depth_events += 1
                state.last_error = "AWAITING_DEPTH_BRIDGE"
                return

            # Binance USD-M requires the first applied diff event to overlap
            # the snapshot's lastUpdateId (U <= lastUpdateId <= u).
            expected_id = snapshot_id

            if not first_update_id <= expected_id <= final_update_id:
                state.sequence_gaps += 1
                state.last_error = (
                    "SNAPSHOT_BRIDGE_GAP "
                    f"SNAPSHOT={snapshot_id} U={first_update_id} u={final_update_id}"
                )
                return

            self._apply_depth_levels_locked(state, event)
            state.last_update_id = final_update_id
            state.book_synced = True
            state.awaiting_bridge = False
            state.next_snapshot_attempt_at = 0.0
            state.depth_updates += 1
            state.depth_updated_at = self.clock()
            state.last_error = ""

    def _request_snapshot(self, symbol: str) -> Mapping[str, Any]:
        if self.snapshot_provider is None:
            raise RuntimeError("snapshot provider unavailable")

        try:
            result = self.snapshot_provider(symbol, limit=self.snapshot_limit)
        except TypeError:
            result = self.snapshot_provider(symbol, self.snapshot_limit)

        if not isinstance(result, Mapping):
            raise TypeError("snapshot provider returned a non-mapping value")

        return result

    def _apply_depth_levels_locked(
        self,
        state: _SymbolState,
        event: Mapping[str, Any],
    ) -> None:
        for key, book in (("b", state.bids), ("a", state.asks)):
            for level in event.get(key) or ():
                if not isinstance(level, (list, tuple)) or len(level) < 2:
                    continue

                price = _safe_float(level[0])
                quantity = _safe_float(level[1])

                if price <= 0:
                    continue

                if quantity <= 0:
                    book.pop(price, None)
                else:
                    book[price] = quantity

        self._trim_book_locked(state)

    def _trim_book_locked(self, state: _SymbolState) -> None:
        trimmed = False

        if len(state.bids) > self.snapshot_limit:
            state.bids = dict(
                sorted(state.bids.items(), reverse=True)[:self.snapshot_limit]
            )
            trimmed = True

        if len(state.asks) > self.snapshot_limit:
            state.asks = dict(
                sorted(state.asks.items())[:self.snapshot_limit]
            )
            trimmed = True

        if trimmed:
            state.book_trim_count += 1

    def _process_trade(self, data: Mapping[str, Any]) -> None:
        symbol = str(data.get("s") or "").upper()

        if symbol not in self.states:
            return

        price = _safe_float(data.get("p"))
        quantity = _safe_float(data.get("q"))

        if price <= 0 or quantity <= 0:
            return

        event_type = str(data.get("e") or "trade")
        identifier = data.get("a") if event_type == "aggTrade" else data.get("t")
        trade_key = (
            f"{event_type}:{identifier}"
            if identifier not in (None, "")
            else ""
        )
        received_at = self.clock()
        timestamp = _event_timestamp_seconds(data, received_at)
        notional = price * quantity
        aggressive_buy = not bool(data.get("m"))
        signed_notional = notional if aggressive_buy else -notional

        with self.lock:
            state = self.states[symbol]
            cutoff = received_at - self.window_seconds

            while state.trades and state.trades[0].timestamp < cutoff:
                state.trades.popleft()

            while (
                state.seen_trade_ids and
                state.seen_trade_ids[0][0] < cutoff
            ):
                _, expired = state.seen_trade_ids.popleft()
                state.seen_trade_id_set.discard(expired)

            if trade_key and trade_key in state.seen_trade_id_set:
                state.duplicate_trades += 1
                return

            if trade_key:
                if len(state.seen_trade_ids) >= state.seen_trade_ids.maxlen:
                    _, expired = state.seen_trade_ids.popleft()
                    state.seen_trade_id_set.discard(expired)

                state.seen_trade_ids.append((received_at, trade_key))
                state.seen_trade_id_set.add(trade_key)

            state.trades.append(_Trade(
                timestamp=timestamp,
                price=price,
                quantity=quantity,
                notional=notional,
                signed_notional=signed_notional,
                aggressive_buy=aggressive_buy,
            ))
            state.cumulative_cvd_notional += signed_notional

            if aggressive_buy:
                state.cumulative_buy_notional += notional
            else:
                state.cumulative_sell_notional += notional

            state.trade_updates += 1
            state.trade_updated_at = received_at

    def _book_metrics(
        self,
        bids: Mapping[float, float],
        asks: Mapping[float, float],
    ) -> dict[str, Any]:
        sorted_bids = sorted(bids.items(), reverse=True)
        sorted_asks = sorted(asks.items())

        if not sorted_bids or not sorted_asks:
            return {}

        best_bid, best_bid_qty = sorted_bids[0]
        best_ask, best_ask_qty = sorted_asks[0]
        mid = (best_bid + best_ask) / 2
        spread_bps = ((best_ask - best_bid) / mid) * 10000 if mid > 0 else 0.0
        bid_notional = sum(price * quantity for price, quantity in sorted_bids)
        ask_notional = sum(price * quantity for price, quantity in sorted_asks)
        total_notional = bid_notional + ask_notional
        full_depth_imbalance = (
            (bid_notional - ask_notional) / total_notional
            if total_notional > 0
            else 0.0
        )
        best_quantity = best_bid_qty + best_ask_qty
        microprice = (
            ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) /
            best_quantity
            if best_quantity > 0
            else mid
        )
        microprice_bps = ((microprice - mid) / mid) * 10000 if mid > 0 else 0.0
        buy_walk = self._walk_book(sorted_asks, self.impact_notional_usdt)
        sell_walk = self._walk_book(sorted_bids, self.impact_notional_usdt)
        buy_impact_bps = (
            ((buy_walk["vwap"] - mid) / mid) * 10000
            if mid > 0 and buy_walk["vwap"] > 0
            else 0.0
        )
        sell_impact_bps = (
            ((mid - sell_walk["vwap"]) / mid) * 10000
            if mid > 0 and sell_walk["vwap"] > 0
            else 0.0
        )
        impact_total = abs(buy_impact_bps) + abs(sell_impact_bps)
        impact_skew = (
            _clip((sell_impact_bps - buy_impact_bps) / impact_total)
            if impact_total > 0
            else 0.0
        )
        return {
            "best_bid": round(best_bid, 12),
            "best_ask": round(best_ask, 12),
            "mid_price": round(mid, 12),
            "spread_bps": round(spread_bps, 4),
            "bid_levels": len(sorted_bids),
            "ask_levels": len(sorted_asks),
            "bid_notional": round(bid_notional, 2),
            "ask_notional": round(ask_notional, 2),
            "full_depth_imbalance": round(full_depth_imbalance, 6),
            "microprice_bps": round(microprice_bps, 6),
            "impact_notional_usdt": self.impact_notional_usdt,
            "buy_impact_bps": round(buy_impact_bps, 6),
            "sell_impact_bps": round(sell_impact_bps, 6),
            "buy_impact_fill_pct": round(buy_walk["fill_pct"], 4),
            "sell_impact_fill_pct": round(sell_walk["fill_pct"], 4),
            "impact_skew": round(impact_skew, 6),
        }

    @staticmethod
    def _walk_book(
        levels: Iterable[tuple[float, float]],
        target_notional: float,
    ) -> dict[str, float]:
        if target_notional <= 0:
            return {"vwap": 0.0, "fill_pct": 0.0}

        remaining = float(target_notional)
        base_quantity = 0.0
        quote_notional = 0.0

        for price, available_quantity in levels:
            if remaining <= 0:
                break

            take_quantity = min(available_quantity, remaining / price)
            taken_notional = take_quantity * price
            base_quantity += take_quantity
            quote_notional += taken_notional
            remaining -= taken_notional

        return {
            "vwap": quote_notional / base_quantity if base_quantity > 0 else 0.0,
            "fill_pct": min((quote_notional / target_notional) * 100, 100.0),
        }

    def _trade_metrics(
        self,
        trades: list[_Trade],
        reference_price: float,
    ) -> dict[str, Any]:
        buy_notional = sum(trade.notional for trade in trades if trade.aggressive_buy)
        sell_notional = sum(trade.notional for trade in trades if not trade.aggressive_buy)
        total_notional = buy_notional + sell_notional
        cvd_notional = buy_notional - sell_notional
        cvd_ratio = cvd_notional / total_notional if total_notional > 0 else 0.0

        if reference_price <= 0 and trades:
            reference_price = trades[-1].price

        bucket_size = max(reference_price * self.bucket_bps / 10000, 1e-12)
        buckets: dict[int, dict[str, float]] = {}

        for trade in trades:
            bucket_index = int(round(trade.price / bucket_size))
            bucket = buckets.setdefault(
                bucket_index,
                {"buy_notional": 0.0, "sell_notional": 0.0},
            )
            key = "buy_notional" if trade.aggressive_buy else "sell_notional"
            bucket[key] += trade.notional

        buy_imbalances = 0
        sell_imbalances = 0
        active_buckets = 0
        point_of_control_price = None
        point_of_control_notional = 0.0
        output_buckets = []

        for bucket_index, values in buckets.items():
            buy = values["buy_notional"]
            sell = values["sell_notional"]
            total = buy + sell
            delta = buy - sell
            imbalance = "NONE"

            if total >= self.min_bucket_notional:
                active_buckets += 1

                if buy > 0 and buy >= sell * self.imbalance_ratio:
                    buy_imbalances += 1
                    imbalance = "BUY"
                elif sell > 0 and sell >= buy * self.imbalance_ratio:
                    sell_imbalances += 1
                    imbalance = "SELL"

            if total > point_of_control_notional:
                point_of_control_notional = total
                point_of_control_price = bucket_index * bucket_size

            output_buckets.append({
                "price": round(bucket_index * bucket_size, 12),
                "buy_notional": round(buy, 2),
                "sell_notional": round(sell, 2),
                "delta": round(delta, 2),
                "total_notional": round(total, 2),
                "imbalance": imbalance,
            })

        footprint_delta_ratio = cvd_ratio
        footprint_imbalance = (
            (buy_imbalances - sell_imbalances) / active_buckets
            if active_buckets > 0
            else 0.0
        )
        max_output_buckets = 100
        footprint_truncated = len(output_buckets) > max_output_buckets

        if footprint_truncated:
            output_buckets = sorted(
                output_buckets,
                key=lambda item: item["total_notional"],
                reverse=True,
            )[:max_output_buckets]

        output_buckets.sort(key=lambda item: item["price"])
        return {
            "trade_count": len(trades),
            "buy_trade_count": sum(trade.aggressive_buy for trade in trades),
            "sell_trade_count": sum(not trade.aggressive_buy for trade in trades),
            "buy_notional": round(buy_notional, 2),
            "sell_notional": round(sell_notional, 2),
            "trade_notional": round(total_notional, 2),
            "cvd_notional": round(cvd_notional, 2),
            "cvd_ratio": round(cvd_ratio, 6),
            "footprint_delta_ratio": round(footprint_delta_ratio, 6),
            "footprint_imbalance": round(footprint_imbalance, 6),
            "footprint_bucket_bps": self.bucket_bps,
            "footprint_bucket_size": round(bucket_size, 12),
            "footprint_bucket_count": len(buckets),
            "footprint_active_buckets": active_buckets,
            "buy_imbalance_buckets": buy_imbalances,
            "sell_imbalance_buckets": sell_imbalances,
            "point_of_control_price": (
                None
                if point_of_control_price is None
                else round(point_of_control_price, 12)
            ),
            "point_of_control_notional": round(point_of_control_notional, 2),
            "footprint_buckets": output_buckets,
            "footprint_truncated": footprint_truncated,
        }

    def _unavailable(self, symbol: str, reason: str) -> dict[str, Any]:
        return {
            "shadow_only": True,
            "decision_effect": False,
            "ranking_effect": False,
            "available": False,
            "reason": reason,
            "symbol": symbol,
            "shadow_score": 0.0,
            "buy_shadow_score": 0.0,
            "sell_shadow_score": 0.0,
        }


__all__ = ["OrderFlowShadowMonitor", "TELEMETRY_FIELDS"]

