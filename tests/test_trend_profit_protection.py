import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import config
from strategy import evaluate_route_profit_protection
from trade_state import (
    load_trade_state,
    record_dca_fill,
    save_trade_state,
)

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


class TrendProfitProtectionTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "TREND_PROFIT_PROTECTION_ENABLED": True,
            "TREND_PROFIT_PROTECTION_TRIGGER_ROI": 15,
            "TREND_PROFIT_PROTECTION_LOCK_ROI": 5,
            "TREND_PROFIT_PROTECTION_RETRACE_PCT": 45,
        }
        self.config_patches = [
            patch.object(config, name, value)
            for name, value in settings.items()
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def test_trend_guard_does_not_arm_before_trigger(self):
        result = evaluate_route_profit_protection(
            "BUY",
            100,
            101.4,
            leverage=10,
            confirmation_type="TREND",
        )

        self.assertEqual(result["current_roi"], 14)
        self.assertFalse(result["armed"])
        self.assertFalse(result["should_exit"])

    def test_trend_guard_allows_normal_retest_above_floor(self):
        result = evaluate_route_profit_protection(
            "BUY",
            100,
            101.2,
            peak_roi=20,
            leverage=10,
            confirmation_type="TREND",
        )

        self.assertTrue(result["armed"])
        self.assertEqual(result["floor_roi"], 11)
        self.assertFalse(result["should_exit"])

    def test_trend_guard_exits_below_peak_retrace_floor(self):
        result = evaluate_route_profit_protection(
            "SELL",
            100,
            99,
            peak_roi=20,
            leverage=10,
            confirmation_type="TREND",
        )

        self.assertEqual(result["current_roi"], 10)
        self.assertEqual(result["floor_roi"], 11)
        self.assertTrue(result["should_exit"])
        self.assertEqual(result["reason"], "TREND_PROFIT_RETRACE_EXIT")

    def test_dca_resets_route_peaks_to_new_average_entry(self):
        with TemporaryDirectory() as temp_dir, patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(temp_dir) / "state.json"),
        ):
            state = {
                "positions": {
                    "BTCUSDT": {
                        "symbol": "BTCUSDT",
                        "managed_by_bot": True,
                        "side": "BUY",
                        "avg_entry": 100,
                        "quantity": 1,
                        "used_margin": 10,
                        "dca_count": 0,
                        "reversal_peak_roi": 18,
                        "reversal_profit_armed": True,
                        "trend_peak_roi": 25,
                        "trend_profit_armed": True,
                    }
                }
            }
            save_trade_state(state)
            updated = record_dca_fill(
                state,
                "BTCUSDT",
                avg_entry=95,
                quantity=2,
                used_margin=10,
                dca_price=90,
            )
            reloaded = load_trade_state()["positions"]["BTCUSDT"]

        self.assertTrue(updated)
        self.assertEqual(reloaded["reversal_peak_roi"], 0)
        self.assertEqual(reloaded["trend_peak_roi"], 0)
        self.assertFalse(reloaded["reversal_profit_armed"])
        self.assertFalse(reloaded["trend_profit_armed"])
        self.assertEqual(reloaded["trend_profit_basis_entry"], 95)

    def test_websocket_trend_retrace_submits_market_exit(self):
        monitor = main.DcaWebsocketMonitor()
        state = {
            "positions": {
                "BTCUSDT": {
                    "symbol": "BTCUSDT",
                    "managed_by_bot": True,
                    "side": "BUY",
                    "confirmation_type": "TREND",
                    "avg_entry": 100,
                    "trend_peak_roi": 20,
                    "trend_profit_basis_entry": 100,
                    "trend_profit_exit_status": "",
                }
            }
        }

        with patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.get_open_position_details",
            return_value={
                "BTCUSDT": {
                    "amount": 1,
                    "position_side": "LONG",
                }
            },
        ), patch(
            "main.close_position_market",
            return_value={"orderId": 1},
        ) as close, patch(
            "main.cancel_open_protection_orders",
        ) as cancel, patch(
            "main.update_position_runtime_fields",
            return_value=True,
        ) as update, patch(
            "main.send_telegram_message",
        ) as telegram:
            handled = monitor._handle_trend_profit_protection(
                "BTCUSDT",
                101,
                state,
            )

        self.assertTrue(handled)
        close.assert_called_once()
        cancel.assert_called_once_with("BTCUSDT")
        telegram.assert_called_once()
        submitted_updates = [
            call.args[2]
            for call in update.call_args_list
            if call.args[2].get("trend_profit_exit_status") == "SUBMITTED"
        ]
        self.assertEqual(len(submitted_updates), 1)


if __name__ == "__main__":
    unittest.main()
