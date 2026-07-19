from collections import deque
import json
import threading
import time

import numpy as np

import config
from logger import log_error, log_info, log_warning


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip(value, minimum=-1.0, maximum=1.0):
    return min(max(float(value), minimum), maximum)


def _message_data(message):
    if not isinstance(message, dict):
        return {}

    data = message.get("data")
    return data if isinstance(data, dict) else message


class MarketFlowMonitor:
    def __init__(self, symbols, shutdown_event=None):
        self.enabled = bool(getattr(config, "MARKET_FLOW_ENABLED", True))
        self.symbols = tuple(dict.fromkeys(symbol.upper() for symbol in symbols))
        self.shutdown_event = shutdown_event
        self.running = False
        self.resetting = False
        self.last_message_at = 0.0
        self.last_trade_message_at = 0.0
        self.last_book_message_at = 0.0
        self.last_restart_at = 0.0
        self.lock = threading.RLock()
        self.stop_event = threading.Event()
        self.watchdog_thread = None
        self.trade_threads = []
        self.trade_websockets = {}
        self.trade_generation = 0
        self.book_threads = []
        self.data = {
            symbol: {
                "trades": deque(),
                "depth_imbalance": 0.0,
                "microprice_bps": 0.0,
                "depth_samples": 0,
                "trade_updated_at": 0.0,
                "depth_updated_at": 0.0,
            }
            for symbol in self.symbols
        }

    def start(self):
        if not self.enabled or not self.symbols:
            log_info("Market flow websocket disabled")
            return

        self._start_watchdog()

        try:
            # Validate the dependency before worker threads are started.
            from websockets.sync.client import connect  # noqa: F401

            with self.lock:
                if self.running:
                    return

                self.running = True
                self.trade_generation += 1
                self.last_message_at = time.time()
                self.last_trade_message_at = self.last_message_at

            self._start_trade_streams()
            self._start_book_streams()

            log_info(
                f"Market flow websocket started | SYMBOLS={len(self.symbols)} | "
                f"TRADE_SOCKETS={len(self.trade_threads)} | "
                f"BOOK_SOCKETS={len(self.book_threads)}"
            )

        except Exception as exc:
            self.running = False
            log_error(f"Market flow websocket start error: {exc}")

    def stop(self):
        self.stop_event.set()

        with self.lock:
            self.running = False
            self.trade_generation += 1
            websockets = list(self.trade_websockets.values())
            self.trade_websockets = {}

        self._close_websockets(websockets)

    def _start_watchdog(self):
        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return

        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="market-flow-websocket-watchdog",
            daemon=True,
        )
        self.watchdog_thread.start()

    def _watchdog_loop(self):
        interval = max(
            _safe_float(
                getattr(config, "MARKET_FLOW_WATCHDOG_INTERVAL_SECONDS", 15),
                15,
            ),
            5,
        )

        while not self.stop_event.wait(interval):
            if self.shutdown_event is not None and self.shutdown_event.is_set():
                return

            stale_seconds = max(
                _safe_float(getattr(config, "MARKET_FLOW_STALE_SECONDS", 45), 45),
                15,
            )
            now = time.time()

            with self.lock:
                age = (
                    now - self.last_trade_message_at
                    if self.last_trade_message_at
                    else stale_seconds + 1
                )
                running = self.running
                resetting = self.resetting

            if not resetting and (not running or age >= stale_seconds):
                self.reset_connection(f"watchdog stale={round(age, 1)}s")

    def _start_trade_streams(self):
        chunk_size = max(
            int(getattr(config, "MARKET_FLOW_STREAMS_PER_SOCKET", 100)),
            1,
        )

        with self.lock:
            generation = self.trade_generation

        threads = []

        for start in range(0, len(self.symbols), chunk_size):
            symbols = self.symbols[start:start + chunk_size]
            thread = threading.Thread(
                target=self._trade_stream_loop,
                args=(symbols, generation),
                name=f"market-flow-trade-{(start // chunk_size) + 1}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)

        with self.lock:
            if generation == self.trade_generation:
                self.trade_threads = threads

    def _trade_stream_loop(self, symbols, generation):
        from websockets.sync.client import connect

        streams = "/".join(f"{symbol.lower()}@trade" for symbol in symbols)
        url = f"wss://fstream.binance.com/stream?streams={streams}"
        worker_id = threading.get_ident()

        while self._trade_worker_active(generation):
            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    with self.lock:
                        if generation != self.trade_generation:
                            return
                        self.trade_websockets[worker_id] = websocket

                    while self._trade_worker_active(generation):
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self.handle_message(json.loads(message))

            except Exception as exc:
                if self._trade_worker_active(generation):
                    log_warning(f"Market flow trade websocket reconnecting: {exc}")
                    self.stop_event.wait(3)
            finally:
                with self.lock:
                    current = self.trade_websockets.get(worker_id)
                    if current is locals().get("websocket"):
                        self.trade_websockets.pop(worker_id, None)

    def _trade_worker_active(self, generation):
        if self.stop_event.is_set():
            return False

        if self.shutdown_event is not None and self.shutdown_event.is_set():
            return False

        with self.lock:
            return generation == self.trade_generation

    @staticmethod
    def _close_websockets(websockets):
        for websocket in websockets:
            try:
                websocket.close()
            except Exception as exc:
                log_warning(f"Market flow websocket close warning: {exc}")

    def _start_book_streams(self):
        chunk_size = max(
            int(getattr(config, "MARKET_FLOW_BOOK_STREAMS_PER_SOCKET", 80)),
            1,
        )
        self.book_threads = []

        for start in range(0, len(self.symbols), chunk_size):
            symbols = self.symbols[start:start + chunk_size]
            thread = threading.Thread(
                target=self._book_stream_loop,
                args=(symbols,),
                name=f"market-flow-book-{(start // chunk_size) + 1}",
                daemon=True,
            )
            thread.start()
            self.book_threads.append(thread)

    def _book_stream_loop(self, symbols):
        from websockets.sync.client import connect

        streams = "/".join(
            f"{symbol.lower()}@bookTicker"
            for symbol in symbols
        )
        url = f"wss://fstream.binance.com/stream?streams={streams}"

        while not self.stop_event.is_set():
            if self.shutdown_event is not None and self.shutdown_event.is_set():
                return

            try:
                with connect(
                    url,
                    open_timeout=10,
                    close_timeout=2,
                    ping_interval=20,
                    ping_timeout=20,
                ) as websocket:
                    while not self.stop_event.is_set():
                        try:
                            message = websocket.recv(timeout=2)
                        except TimeoutError:
                            continue

                        self.handle_message(json.loads(message))

            except Exception as exc:
                if not self.stop_event.is_set():
                    log_warning(f"Market flow book websocket reconnecting: {exc}")
                    self.stop_event.wait(3)

    def reset_connection(self, reason):
        now = time.time()
        cooldown = max(
            _safe_float(
                getattr(config, "MARKET_FLOW_RESTART_COOLDOWN_SECONDS", 30),
                30,
            ),
            5,
        )

        with self.lock:
            if self.resetting or now - self.last_restart_at < cooldown:
                return

            self.resetting = True
            self.last_restart_at = now

        threading.Thread(
            target=self._reset_connection,
            args=(reason,),
            name="market-flow-websocket-reset",
            daemon=True,
        ).start()

    def _reset_connection(self, reason):
        log_warning(f"Market flow websocket resetting | REASON={reason}")

        try:
            with self.lock:
                self.running = False
                self.trade_generation += 1
                websockets = list(self.trade_websockets.values())
                self.trade_websockets = {}

            self._close_websockets(websockets)

            if self.stop_event.wait(2):
                return

            with self.lock:
                self.running = True
                self.last_message_at = time.time()
                self.last_trade_message_at = self.last_message_at

            self._start_trade_streams()

            log_info(
                f"Market flow websocket restored | "
                f"TRADE_SOCKETS={len(self.trade_threads)}"
            )

        except Exception as exc:
            log_error(f"Market flow websocket reset error: {exc}")
        finally:
            with self.lock:
                self.resetting = False

    def handle_message(self, message):
        if self.stop_event.is_set():
            return

        if not isinstance(message, dict):
            return

        data = _message_data(message)

        if data.get("e") == "error" or message.get("e") == "error":
            reason = data.get("type") or data.get("m") or "stream error"
            self.reset_connection(reason)
            return

        event_type = data.get("e")

        with self.lock:
            self.last_message_at = time.time()

        if event_type in {"aggTrade", "trade"}:
            self._handle_trade(data)
        elif event_type == "depthUpdate":
            self._handle_depth(data)
        elif (
            data.get("s") and
            all(key in data for key in ("b", "B", "a", "A"))
        ):
            self._handle_book_ticker(data)

    def _handle_trade(self, data):
        symbol = str(data.get("s") or "").upper()

        if symbol not in self.data:
            return

        price = _safe_float(data.get("p"))
        quantity = _safe_float(data.get("q"))

        if price <= 0 or quantity <= 0:
            return

        timestamp = _safe_float(data.get("T") or data.get("E"), time.time() * 1000) / 1000
        notional = price * quantity
        signed_notional = -notional if bool(data.get("m")) else notional
        maximum_window = max(
            int(getattr(config, "MARKET_FLOW_MAX_WINDOW_SECONDS", 900)),
            60,
        )

        with self.lock:
            item = self.data[symbol]
            item["trades"].append((timestamp, signed_notional, notional))
            item["trade_updated_at"] = time.time()
            self.last_trade_message_at = item["trade_updated_at"]
            cutoff = timestamp - maximum_window

            while item["trades"] and item["trades"][0][0] < cutoff:
                item["trades"].popleft()

    def _handle_depth(self, data):
        symbol = str(data.get("s") or "").upper()

        if symbol not in self.data:
            return

        bids = data.get("b") or []
        asks = data.get("a") or []

        if not bids or not asks:
            return

        bid_notional = sum(
            _safe_float(level[0]) * _safe_float(level[1])
            for level in bids
            if len(level) >= 2
        )
        ask_notional = sum(
            _safe_float(level[0]) * _safe_float(level[1])
            for level in asks
            if len(level) >= 2
        )
        total_notional = bid_notional + ask_notional

        if total_notional <= 0:
            return

        imbalance = (bid_notional - ask_notional) / total_notional
        best_bid = _safe_float(bids[0][0])
        best_bid_qty = _safe_float(bids[0][1])
        best_ask = _safe_float(asks[0][0])
        best_ask_qty = _safe_float(asks[0][1])
        mid = (best_bid + best_ask) / 2 if best_bid and best_ask else 0
        quantity_total = best_bid_qty + best_ask_qty
        microprice = (
            ((best_ask * best_bid_qty) + (best_bid * best_ask_qty)) /
            quantity_total
            if quantity_total > 0
            else mid
        )
        microprice_bps = ((microprice - mid) / mid) * 10000 if mid else 0
        alpha = _clip(
            getattr(config, "MARKET_FLOW_DEPTH_EMA_ALPHA", 0.15),
            0.01,
            1.0,
        )

        with self.lock:
            item = self.data[symbol]

            if item["depth_samples"]:
                item["depth_imbalance"] = (
                    (alpha * imbalance) +
                    ((1 - alpha) * item["depth_imbalance"])
                )
                item["microprice_bps"] = (
                    (alpha * microprice_bps) +
                    ((1 - alpha) * item["microprice_bps"])
                )
            else:
                item["depth_imbalance"] = imbalance
                item["microprice_bps"] = microprice_bps

            item["depth_samples"] += 1
            item["depth_updated_at"] = time.time()
            self.last_book_message_at = item["depth_updated_at"]

    def _handle_book_ticker(self, data):
        self._handle_depth({
            "s": data.get("s"),
            "b": [[data.get("b"), data.get("B")]],
            "a": [[data.get("a"), data.get("A")]],
        })

    def snapshot(self, symbol):
        symbol = str(symbol or "").upper()
        now = time.time()
        windows = (60, 300, 900)
        stale_seconds = max(
            _safe_float(getattr(config, "MARKET_FLOW_STALE_SECONDS", 45), 45),
            15,
        )

        with self.lock:
            item = self.data.get(symbol)

            if not item:
                return {"available": False, "reason": "SYMBOL_NOT_WATCHED"}

            trades = list(item["trades"])
            depth_imbalance = float(item["depth_imbalance"])
            microprice_bps = float(item["microprice_bps"])
            depth_samples = int(item["depth_samples"])
            trade_age = now - item["trade_updated_at"] if item["trade_updated_at"] else None
            depth_age = now - item["depth_updated_at"] if item["depth_updated_at"] else None

        ratios = {}
        notionals = {}

        for window in windows:
            cutoff = now - window
            selected = [trade for trade in trades if trade[0] >= cutoff]
            total = sum(trade[2] for trade in selected)
            signed = sum(trade[1] for trade in selected)
            ratios[window] = signed / total if total > 0 else None
            notionals[window] = total

        minimum_notional = max(
            _safe_float(
                getattr(config, "MARKET_FLOW_MIN_NOTIONAL_USDT", 25000),
                25000,
            ),
            0,
        )
        depth_fresh = depth_age is not None and depth_age <= stale_seconds
        trade_fresh = trade_age is not None and trade_age <= stale_seconds
        weighted_parts = []

        if trade_fresh:
            for window, weight in ((60, 0.5), (300, 0.3), (900, 0.2)):
                ratio = ratios[window]

                if ratio is not None and notionals[window] >= minimum_notional:
                    weighted_parts.append((ratio, weight))

        weight_total = sum(weight for _, weight in weighted_parts)
        cvd_score = (
            sum(value * weight for value, weight in weighted_parts) / weight_total
            if weight_total > 0
            else 0.0
        )
        depth_component = depth_imbalance if depth_fresh else 0.0
        micro_scale = max(
            _safe_float(getattr(config, "MARKET_FLOW_MICROPRICE_SCALE_BPS", 2), 2),
            0.1,
        )
        micro_component = _clip(microprice_bps / micro_scale) if depth_fresh else 0.0
        directional = _clip(
            (cvd_score * 0.60) +
            (depth_component * 0.30) +
            (micro_component * 0.10)
        )
        max_score = max(
            _safe_float(getattr(config, "MARKET_FLOW_MAX_SCORE", 5), 5),
            0,
        )
        buy_score = round(directional * max_score, 3)
        components = {
            "cvd_1m": ratios[60] if trade_fresh else None,
            "cvd_5m": ratios[300] if trade_fresh else None,
            "cvd_15m": ratios[900] if trade_fresh else None,
            "depth": depth_component if depth_fresh else None,
            "microprice": micro_component if depth_fresh else None,
        }
        conflict_threshold = max(
            _safe_float(
                getattr(config, "MARKET_FLOW_COMPONENT_CONFLICT_THRESHOLD", 0.12),
                0.12,
            ),
            0,
        )
        buy_conflicts = sum(
            value is not None and value <= -conflict_threshold
            for value in components.values()
        )
        sell_conflicts = sum(
            value is not None and value >= conflict_threshold
            for value in components.values()
        )

        return {
            "available": bool(trade_fresh or depth_fresh),
            "symbol": symbol,
            "buy_score": buy_score,
            "sell_score": round(-buy_score, 3),
            "cvd_1m": (
                round(ratios[60], 4)
                if trade_fresh and ratios[60] is not None
                else None
            ),
            "cvd_5m": (
                round(ratios[300], 4)
                if trade_fresh and ratios[300] is not None
                else None
            ),
            "cvd_15m": (
                round(ratios[900], 4)
                if trade_fresh and ratios[900] is not None
                else None
            ),
            "notional_1m": round(notionals[60], 2) if trade_fresh else 0,
            "depth_imbalance": round(depth_imbalance, 4) if depth_fresh else None,
            "microprice_bps": round(microprice_bps, 4) if depth_fresh else None,
            "buy_conflicts": buy_conflicts,
            "sell_conflicts": sell_conflicts,
            "trade_age_seconds": None if trade_age is None else round(trade_age, 1),
            "depth_age_seconds": None if depth_age is None else round(depth_age, 1),
            "depth_samples": depth_samples,
        }


def build_breadth_sample(symbol, trend_df):
    try:
        closed = trend_df.iloc[:-1]

        if len(closed) < 6:
            return None

        latest = closed.iloc[-1]
        previous = closed.iloc[-2]
        older = closed.iloc[-6]
        close = _safe_float(latest.get("close"))
        previous_close = _safe_float(previous.get("close"))
        older_close = _safe_float(older.get("close"))
        volume = _safe_float(latest.get("volume"))
        volume_sma = max(_safe_float(latest.get("volume_sma"), volume), 1e-10)

        return {
            "symbol": symbol,
            "above_ema20": close >= _safe_float(latest.get("ema20"), close),
            "above_ema50": close >= _safe_float(latest.get("ema50"), close),
            "advanced": close >= previous_close,
            "return_5": ((close - older_close) / older_close) if older_close > 0 else 0,
            "signed_volume": (1 if close >= previous_close else -1) * min(volume / volume_sma, 3),
        }

    except Exception:
        return None


def calculate_market_breadth(samples):
    samples = [sample for sample in (samples or []) if sample]

    if not samples:
        return {"available": False, "sample_count": 0, "buy_score": 0, "sell_score": 0}

    count = len(samples)
    above_ema20 = sum(sample["above_ema20"] for sample in samples) / count
    above_ema50 = sum(sample["above_ema50"] for sample in samples) / count
    advance_ratio = sum(sample["advanced"] for sample in samples) / count
    signed_volume = sum(sample["signed_volume"] for sample in samples)
    absolute_volume = sum(abs(sample["signed_volume"]) for sample in samples)
    volume_balance = signed_volume / absolute_volume if absolute_volume else 0
    median_return_5 = float(np.median([sample["return_5"] for sample in samples]))
    minimum_samples = max(
        int(getattr(config, "MARKET_BREADTH_MIN_SAMPLES", 20)),
        1,
    )
    directional = _clip(
        ((above_ema20 - 0.5) * 0.70) +
        ((above_ema50 - 0.5) * 0.60) +
        ((advance_ratio - 0.5) * 0.50) +
        (volume_balance * 0.20) +
        (_clip(median_return_5 / 0.03) * 0.20)
    )
    max_score = max(
        _safe_float(getattr(config, "MARKET_BREADTH_MAX_SCORE", 3), 3),
        0,
    )
    available = count >= minimum_samples
    buy_score = round(directional * max_score, 3) if available else 0

    return {
        "available": available,
        "sample_count": count,
        "above_ema20_pct": round(above_ema20 * 100, 2),
        "above_ema50_pct": round(above_ema50 * 100, 2),
        "advance_pct": round(advance_ratio * 100, 2),
        "volume_balance": round(volume_balance, 4),
        "median_return_5_pct": round(median_return_5 * 100, 3),
        "buy_score": buy_score,
        "sell_score": round(-buy_score, 3),
    }


def calculate_regime_transition(entry_df):
    try:
        closes = np.asarray(entry_df["close"].iloc[:-1], dtype=float)
        recent_window = max(
            int(getattr(config, "REGIME_TRANSITION_RECENT_CANDLES", 6)),
            3,
        )
        baseline_window = max(
            int(getattr(config, "REGIME_TRANSITION_BASELINE_CANDLES", 30)),
            recent_window * 2,
        )

        if len(closes) < baseline_window + 2:
            return {"available": False, "buy_score": 0, "sell_score": 0}

        returns = np.diff(closes) / np.maximum(closes[:-1], 1e-10)
        recent = returns[-recent_window:]
        baseline = returns[-(baseline_window + recent_window):-recent_window]
        baseline_mean = float(np.mean(baseline))
        baseline_std = max(float(np.std(baseline)), 1e-8)
        recent_mean = float(np.mean(recent))
        recent_std = float(np.std(recent))
        shift_z = (recent_mean - baseline_mean) / (baseline_std / np.sqrt(recent_window))
        volatility_ratio = recent_std / baseline_std
        directional = _clip(shift_z / 3)
        transition_threshold = max(
            _safe_float(getattr(config, "REGIME_TRANSITION_Z_THRESHOLD", 1.5), 1.5),
            0.5,
        )
        volatility_threshold = max(
            _safe_float(
                getattr(config, "REGIME_TRANSITION_VOLATILITY_RATIO", 1.6),
                1.6,
            ),
            1,
        )
        transition = (
            abs(shift_z) >= transition_threshold or
            volatility_ratio >= volatility_threshold
        )
        strength = min(
            1.0,
            (abs(shift_z) / max(transition_threshold * 2, 1)) +
            max(volatility_ratio - 1, 0) * 0.20,
        )
        max_score = max(
            _safe_float(getattr(config, "REGIME_TRANSITION_MAX_SCORE", 2), 2),
            0,
        )
        buy_score = directional * strength * max_score if transition else 0

        return {
            "available": True,
            "transition": transition,
            "direction": "UP" if directional > 0 else "DOWN" if directional < 0 else "FLAT",
            "shift_z": round(shift_z, 3),
            "volatility_ratio": round(volatility_ratio, 3),
            "buy_score": round(buy_score, 3),
            "sell_score": round(-buy_score, 3),
        }

    except Exception as exc:
        return {
            "available": False,
            "buy_score": 0,
            "sell_score": 0,
            "reason": str(exc),
        }
