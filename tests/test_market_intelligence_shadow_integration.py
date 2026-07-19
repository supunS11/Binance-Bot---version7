import inspect
import time
import unittest
from unittest.mock import Mock, patch

from market_intelligence import (
    FUTURES_MARKET_STREAM_BASE,
    FUTURES_PUBLIC_STREAM_BASE,
    MarketFlowMonitor,
)


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


class MarketFlowRouteTests(unittest.TestCase):
    def test_stream_routes_match_current_binance_futures_split(self):
        self.assertEqual(
            FUTURES_MARKET_STREAM_BASE,
            "wss://fstream.binance.com/market/stream?streams=",
        )
        self.assertEqual(
            FUTURES_PUBLIC_STREAM_BASE,
            "wss://fstream.binance.com/public/stream?streams=",
        )

        trade_source = inspect.getsource(MarketFlowMonitor._trade_stream_loop)
        book_source = inspect.getsource(MarketFlowMonitor._book_stream_loop)
        self.assertIn('@aggTrade', trade_source)
        self.assertIn('FUTURES_MARKET_STREAM_BASE', trade_source)
        self.assertIn('@bookTicker', book_source)
        self.assertIn('FUTURES_PUBLIC_STREAM_BASE', book_source)


class ShadowTradeFanoutTests(unittest.TestCase):
    @staticmethod
    def trade_message():
        return {
            "stream": "btcusdt@aggTrade",
            "data": {
                "e": "aggTrade",
                "s": "BTCUSDT",
                "p": "100",
                "q": "2",
                "m": False,
                "T": int(time.time() * 1000),
                "a": 123,
            },
        }

    def test_trade_is_fanned_out_without_changing_core_processing(self):
        shadow = Mock()
        monitor = MarketFlowMonitor(["BTCUSDT"], shadow_monitor=shadow)
        message = self.trade_message()

        with patch("config.MARKET_FLOW_MIN_NOTIONAL_USDT", 0):
            monitor.handle_message(message)
            core_snapshot = monitor.snapshot("BTCUSDT")

        shadow.handle_message.assert_called_once_with(message["data"])
        self.assertTrue(core_snapshot["available"])
        self.assertEqual(core_snapshot["cvd_1m"], 1.0)
        self.assertGreater(core_snapshot["buy_score"], 0)

    def test_shadow_failure_is_fail_open_for_core_market_flow(self):
        shadow = Mock()
        shadow.handle_message.side_effect = RuntimeError("shadow unavailable")
        monitor = MarketFlowMonitor(["BTCUSDT"], shadow_monitor=shadow)

        with (
            patch("config.MARKET_FLOW_MIN_NOTIONAL_USDT", 0),
            patch("market_intelligence.log_warning") as warning,
        ):
            monitor.handle_message(self.trade_message())
            core_snapshot = monitor.snapshot("BTCUSDT")

        shadow.handle_message.assert_called_once()
        warning.assert_called_once()
        self.assertTrue(core_snapshot["available"])
        self.assertEqual(core_snapshot["cvd_1m"], 1.0)
        self.assertGreater(core_snapshot["buy_score"], 0)


class CandidateShadowIsolationTests(unittest.TestCase):
    def test_attachment_preserves_rank_and_market_context(self):
        market_context = {
            "route": "TREND",
            "flow": {"buy_score": 0.71},
        }
        candidate = {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "rank_score": 8.25,
            "market_context": market_context,
        }
        shadow = Mock()
        shadow.snapshot.return_value = {
            "shadow_only": True,
            "decision_effect": False,
            "ranking_effect": False,
            "available": True,
            "score": 0.95,
        }

        result = main.attach_candidate_shadow_order_flow(candidate, shadow)

        self.assertIs(result, candidate)
        self.assertEqual(candidate["rank_score"], 8.25)
        self.assertIs(candidate["market_context"], market_context)
        self.assertEqual(candidate["market_context"]["flow"]["buy_score"], 0.71)
        self.assertTrue(candidate["shadow_order_flow"]["shadow_only"])
        shadow.snapshot.assert_called_once_with(
            "BTCUSDT",
            context={
                "signal_side": "BUY",
                "route": "TREND",
                "rank_score": 8.25,
            },
        )

if __name__ == "__main__":
    unittest.main()
