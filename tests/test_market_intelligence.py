import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from market_intelligence import (
    MarketFlowMonitor,
    build_breadth_sample,
    calculate_market_breadth,
    calculate_regime_transition,
)


class MarketFlowTests(unittest.TestCase):
    def test_start_uses_independent_direct_stream_workers(self):
        monitor = MarketFlowMonitor(["BTCUSDT"])

        with (
            patch.object(monitor, "_start_watchdog"),
            patch.object(monitor, "_start_trade_streams") as start_trades,
            patch.object(monitor, "_start_book_streams") as start_books,
        ):
            monitor.start()

        self.assertTrue(monitor.running)
        self.assertEqual(monitor.trade_generation, 1)
        start_trades.assert_called_once_with()
        start_books.assert_called_once_with()

    def test_stop_closes_trade_websockets_and_invalidates_workers(self):
        monitor = MarketFlowMonitor(["BTCUSDT"])
        websocket = Mock()
        monitor.running = True
        monitor.trade_generation = 3
        monitor.trade_websockets[1] = websocket

        monitor.stop()

        self.assertFalse(monitor.running)
        self.assertEqual(monitor.trade_generation, 4)
        self.assertEqual(monitor.trade_websockets, {})
        websocket.close.assert_called_once_with()

    def test_aggressive_buy_flow_and_bid_depth_produce_buy_score(self):
        monitor = MarketFlowMonitor(["BTCUSDT"])
        now_ms = int(time.time() * 1000)

        with patch("config.MARKET_FLOW_MIN_NOTIONAL_USDT", 0):
            for offset in range(3):
                monitor.handle_message({
                    "e": "aggTrade",
                    "s": "BTCUSDT",
                    "p": "100",
                    "q": "10",
                    "m": False,
                    "T": now_ms + offset,
                })

            monitor.handle_message({
                "s": "BTCUSDT",
                "b": "99.9",
                "B": "30",
                "a": "100.1",
                "A": "10",
            })
            snapshot = monitor.snapshot("BTCUSDT")

        self.assertTrue(snapshot["available"])
        self.assertGreater(snapshot["buy_score"], 0)
        self.assertLess(snapshot["sell_score"], 0)

    def test_raw_trade_event_updates_cvd(self):
        monitor = MarketFlowMonitor(["BTCUSDT"])

        with patch("config.MARKET_FLOW_MIN_NOTIONAL_USDT", 0):
            monitor.handle_message({
                "e": "trade",
                "s": "BTCUSDT",
                "p": "100",
                "q": "10",
                "m": False,
                "T": int(time.time() * 1000),
            })
            snapshot = monitor.snapshot("BTCUSDT")

        self.assertTrue(snapshot["available"])
        self.assertEqual(snapshot["cvd_1m"], 1.0)
        self.assertGreater(snapshot["buy_score"], 0)

    def test_stale_trades_do_not_influence_fresh_book_score(self):
        monitor = MarketFlowMonitor(["BTCUSDT"])
        now = time.time()

        with monitor.lock:
            item = monitor.data["BTCUSDT"]
            item["trades"].append((now - 30, 1000, 1000))
            item["trade_updated_at"] = now - 30
            item["depth_imbalance"] = 0
            item["microprice_bps"] = 0
            item["depth_updated_at"] = now

        with patch("config.MARKET_FLOW_STALE_SECONDS", 15), patch(
            "config.MARKET_FLOW_MIN_NOTIONAL_USDT",
            0,
        ):
            snapshot = monitor.snapshot("BTCUSDT")

        self.assertTrue(snapshot["available"])
        self.assertIsNone(snapshot["cvd_1m"])
        self.assertEqual(snapshot["notional_1m"], 0)
        self.assertEqual(snapshot["buy_score"], 0)
        self.assertEqual(snapshot["sell_score"], 0)


class BreadthAndTransitionTests(unittest.TestCase):
    @staticmethod
    def frame(prices):
        rows = []

        for index, close in enumerate(prices):
            rows.append({
                "close": close,
                "ema20": close - 1,
                "ema50": close - 2,
                "volume": 120,
                "volume_sma": 100,
            })

        return pd.DataFrame(rows)

    def test_broad_participation_supports_buy_ranking(self):
        samples = [
            build_breadth_sample(f"S{index}", self.frame(range(90, 101)))
            for index in range(25)
        ]
        breadth = calculate_market_breadth(samples)

        self.assertTrue(breadth["available"])
        self.assertGreater(breadth["buy_score"], 0)
        self.assertLess(breadth["sell_score"], 0)

    def test_change_point_detects_recent_upward_shift(self):
        prices = [100 + (index * 0.01) for index in range(40)]
        prices.extend(prices[-1] + (index * 1.5) for index in range(1, 9))
        prices.append(prices[-1])
        result = calculate_regime_transition(self.frame(prices))

        self.assertTrue(result["available"])
        self.assertTrue(result["transition"])
        self.assertEqual(result["direction"], "UP")
        self.assertGreater(result["buy_score"], 0)


if __name__ == "__main__":
    unittest.main()
