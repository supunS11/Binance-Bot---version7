import unittest
from unittest.mock import patch

import pandas as pd

import config
from backtest import BacktestData, simulate_trade
from strategy import evaluate_route_early_invalidation


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def timeframe_result(
    *,
    structure_break=False,
    opposes_direction=False,
    opposite_reversal=False,
    ema_wrong_side=False,
    support_score=0,
):
    return {
        "latest_close": 100,
        "structure_break": structure_break,
        "opposes_direction": opposes_direction,
        "opposite_reversal": opposite_reversal,
        "ema_wrong_side": ema_wrong_side,
        "support_score": support_score,
    }


class RouteEarlyInvalidationStrategyTests(unittest.TestCase):
    @patch("strategy._live_entry_timeframe_check")
    def test_trend_requires_confirmed_fast_and_slow_failure(self, check):
        check.side_effect = [
            timeframe_result(
                structure_break=True,
                opposes_direction=True,
                support_score=-1,
            ),
            timeframe_result(
                structure_break=True,
                opposes_direction=True,
                support_score=-1,
            ),
        ]

        result = evaluate_route_early_invalidation(
            "BUY",
            object(),
            object(),
            95,
            confirmation_type="TREND",
            reference_price=96,
        )

        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "TREND_THESIS_INVALIDATED")

    @patch("strategy._live_entry_timeframe_check")
    def test_reference_breach_alone_never_exits(self, check):
        check.side_effect = [timeframe_result(), timeframe_result()]

        result = evaluate_route_early_invalidation(
            "BUY",
            object(),
            object(),
            90,
            confirmation_type="TREND",
            reference_price=96,
        )

        self.assertTrue(result["reference_broken"])
        self.assertFalse(result["should_exit"])

    @patch("strategy._live_entry_timeframe_check")
    def test_reversal_can_exit_after_slow_failure_and_fast_opposition(self, check):
        check.side_effect = [
            timeframe_result(opposes_direction=True, support_score=-1),
            timeframe_result(
                structure_break=True,
                opposes_direction=True,
                support_score=-1,
            ),
        ]

        result = evaluate_route_early_invalidation(
            "SELL",
            object(),
            object(),
            105,
            confirmation_type="REVERSAL",
            reference_price=104,
        )

        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "REVERSAL_THESIS_INVALIDATED")

    @patch("strategy._live_entry_timeframe_check")
    def test_reversal_does_not_exit_without_slow_structure_failure(self, check):
        check.side_effect = [
            timeframe_result(
                structure_break=True,
                opposes_direction=True,
                support_score=-1,
            ),
            timeframe_result(opposes_direction=True, support_score=-1),
        ]

        result = evaluate_route_early_invalidation(
            "BUY",
            object(),
            object(),
            95,
            confirmation_type="REVERSAL",
            reference_price=96,
        )

        self.assertFalse(result["should_exit"])


class RouteEarlyInvalidationMonitorTests(unittest.TestCase):
    def test_post_dca_grace_blocks_invalidation_check(self):
        monitor = main.DcaWebsocketMonitor()
        position = {
            "managed_by_bot": True,
            "side": "BUY",
            "avg_entry": 100,
            "confirmation_type": "TREND",
            "opened_at": "2026-01-01T00:00:00",
            "last_dca_at": "2026-01-01T01:00:00",
        }

        with patch.object(config, "EARLY_FLOW_EXIT_TREND_ENABLED", True), patch.object(
            config,
            "EARLY_FLOW_EXIT_POST_DCA_GRACE_MINUTES",
            15,
        ), patch("main.seconds_since", return_value=300):
            result = monitor._route_early_invalidation_context(position, 95)

        self.assertIsNone(result)

    def test_websocket_invalidation_prevents_same_tick_dca(self):
        monitor = main.DcaWebsocketMonitor()

        with patch("main.parse_mark_price_message", return_value=("BTCUSDT", 95)), patch(
            "main.load_trade_state",
            return_value={"positions": {}},
        ), patch.object(
            monitor,
            "_handle_reversal_profit_protection",
            return_value=False,
        ), patch.object(
            monitor,
            "_handle_route_early_invalidation",
            return_value=True,
        ), patch("main.dca_tick_ready") as dca_ready, patch(
            "main.run_dca_check"
        ) as run_dca:
            monitor.handle_message({"data": {}})

        dca_ready.assert_not_called()
        run_dca.assert_not_called()

    def test_confirmed_invalidation_closes_and_persists_state(self):
        monitor = main.DcaWebsocketMonitor()
        state = {
            "positions": {
                "TESTUSDT": {
                    "managed_by_bot": True,
                    "side": "BUY",
                    "confirmation_type": "TREND",
                }
            }
        }
        context = {
            "route": "TREND",
            "side": "BUY",
            "avg_entry": 100,
            "current_roi": -25,
            "max_roi": -20,
            "reference_price": 98,
        }
        info = {
            "should_exit": True,
            "reason": "TREND_THESIS_INVALIDATED",
            "fast_failure": True,
            "slow_failure": True,
            "fast_adverse": True,
            "slow_adverse": True,
            "dual_opposition": True,
            "reference_broken": False,
            "fast": {"support_score": -1},
            "slow": {"support_score": -1},
        }

        with patch.object(config, "EARLY_FLOW_EXIT_ENABLED", True), patch.object(
            monitor,
            "_route_early_invalidation_context",
            return_value=context,
        ), patch("main.get_klines", return_value=object()), patch(
            "main.apply_indicators",
            return_value=object(),
        ), patch(
            "main.evaluate_route_early_invalidation",
            return_value=info,
        ), patch("main.load_trade_state", return_value=state), patch(
            "main.get_open_position_details",
            return_value={
                "TESTUSDT": {
                    "amount": 1,
                    "position_side": "LONG",
                }
            },
        ), patch("main.close_position_market", return_value={"orderId": 1}) as close, patch(
            "main.cancel_open_protection_orders"
        ) as cancel, patch(
            "main.update_position_runtime_fields",
            return_value=True,
        ) as update, patch("main.send_telegram_message"):
            handled = monitor._handle_route_early_invalidation(
                "TESTUSDT",
                97.5,
                state,
            )

        self.assertTrue(handled)
        close.assert_called_once_with("TESTUSDT", 1.0, position_side="LONG")
        cancel.assert_called_once_with("TESTUSDT")
        self.assertEqual(
            update.call_args.args[2]["early_invalidation_exit_status"],
            "SUBMITTED",
        )


class RouteEarlyInvalidationBacktestTests(unittest.TestCase):
    @patch("backtest.evaluate_route_early_invalidation")
    @patch("backtest.compute_stop_loss", return_value=(None, "SL_DISABLED"))
    @patch("backtest.compute_take_profit", return_value=(200, "TEST_TP"))
    def test_backtest_records_route_invalidation_exit(
        self,
        _take_profit,
        _stop_loss,
        invalidation,
    ):
        interval_ms = 3_600_000
        rows = []

        for index in range(310):
            price = 97 if index >= 301 else 100
            rows.append({
                "time": index * interval_ms,
                "close_time": (index + 1) * interval_ms - 1,
                "open": price,
                "high": price + 0.5,
                "low": price - 0.5,
                "close": price,
            })

        data = pd.DataFrame(rows)
        frame = BacktestData(raw=data, indicators=data)
        frames = {
            "trend": frame,
            "confirm": frame,
            "entry": frame,
            "exit": frame,
        }
        invalidation.return_value = {
            "should_exit": True,
            "reason": "TREND_THESIS_INVALIDATED",
        }

        with patch.object(config, "EARLY_FLOW_EXIT_ENABLED", True), patch.object(
            config,
            "EARLY_FLOW_EXIT_TREND_ENABLED",
            True,
        ), patch.object(config, "EARLY_FLOW_EXIT_MINUTES", 15), patch.object(
            config,
            "EARLY_FLOW_EXIT_TREND_MAX_ROI",
            -20,
        ), patch.object(config, "BACKTEST_USE_DCA", False), patch.object(
            config,
            "BACKTEST_MAX_HOLD_CANDLES",
            5,
        ), patch.object(
            config,
            "LEVERAGE",
            10,
        ):
            result = simulate_trade(
                "BTCUSDT",
                "BUY",
                "TREND",
                90,
                300 * interval_ms,
                100,
                300,
                frames,
            )

        self.assertEqual(result["exit_reason"], "TREND_EARLY_INVALIDATION")
        self.assertEqual(
            result["early_invalidation_reason"],
            "TREND_THESIS_INVALIDATED",
        )


if __name__ == "__main__":
    unittest.main()
