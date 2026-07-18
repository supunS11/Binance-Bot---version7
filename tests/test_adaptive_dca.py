import unittest
from unittest.mock import patch

import pandas as pd

import config
from backtest import BacktestData, simulate_trade
from strategy import validate_dca_continuation_guard


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def market_frames(thesis_valid=True):
    if thesis_valid:
        trend_rows = [
            {
                "close": 107,
                "high": 109,
                "low": 104,
                "ema20": 106,
                "ema50": 103,
                "ema200": 100,
                "macd": 0.8,
                "macd_signal": 0.4,
                "adx": 24,
            },
            {
                "close": 110,
                "high": 112,
                "low": 106,
                "ema20": 108,
                "ema50": 105,
                "ema200": 100,
                "macd": 1.0,
                "macd_signal": 0.5,
                "adx": 25,
            },
            {
                "close": 111,
                "high": 113,
                "low": 108,
                "ema20": 109,
                "ema50": 106,
                "ema200": 101,
                "macd": 1.1,
                "macd_signal": 0.6,
                "adx": 25,
            },
        ]
    else:
        trend_rows = [
            {
                "close": 96,
                "high": 99,
                "low": 94,
                "ema20": 99,
                "ema50": 102,
                "ema200": 105,
                "macd": -0.8,
                "macd_signal": -0.2,
                "adx": 24,
            },
            {
                "close": 90,
                "high": 95,
                "low": 88,
                "ema20": 95,
                "ema50": 100,
                "ema200": 105,
                "macd": -1.0,
                "macd_signal": -0.3,
                "adx": 25,
            },
            {
                "close": 91,
                "high": 94,
                "low": 89,
                "ema20": 94,
                "ema50": 99,
                "ema200": 104,
                "macd": -0.9,
                "macd_signal": -0.4,
                "adx": 25,
            },
        ]

    confirm_rows = [
        {"atr": 5.0},
        {"atr": 5.0},
        {"atr": 5.0},
    ]
    entry_rows = [{"close": 100}, {"close": 101}, {"close": 102}]
    return (
        pd.DataFrame(trend_rows),
        pd.DataFrame(confirm_rows),
        pd.DataFrame(entry_rows),
    )


class AdaptiveDcaTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "DCA_TRIGGER_MODE": "adaptive_hybrid",
            "DCA_STRICT_GUARD_ENABLED": True,
            "DCA_ADAPTIVE_ATR_MULTIPLIERS": [1.0, 1.25, 1.5, 1.75],
            "DCA_ADAPTIVE_REQUIRE_STRUCTURE": True,
            "DCA_ADAPTIVE_REQUIRE_RECOVERY": True,
            "DCA_ADAPTIVE_MIN_RECOVERY_SCORE": 2.5,
            "DCA_ADAPTIVE_REVERSAL_MIN_RECOVERY_SCORE": 3.5,
            "DCA_ADAPTIVE_RECOVERY_STEP_PER_LEVEL": 0.25,
            "DCA_ADAPTIVE_BLOCK_TREND_THESIS_INVALIDATION": True,
            "DCA_ADAPTIVE_THESIS_MIN_ADX": 18,
            "DCA_MAX_ADVERSE_ROI": 150,
            "DCA_STRICT_GUARD_BLOCK_TREND_ON_NO_STRUCTURE": False,
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

    def evaluate(
        self,
        current_price=100,
        spacing_anchor=110,
        dca_level=1,
        adverse_roi=25,
        trigger_roi=25,
        recovery_score=3.0,
        structure_ok=True,
        confirmation_type="TREND",
        thesis_valid=True,
    ):
        trend_df, confirm_df, entry_df = market_frames(thesis_valid)
        structure = {
            "reason": (
                "DCA_STRUCTURE_LEVEL_OK"
                if structure_ok
                else "NO DCA SUPPORT LEVEL NEAR CURRENT PRICE"
            )
        }

        with patch(
            "strategy._continuation_pressure_against_side",
            return_value={"score": 1.0, "active": []},
        ), patch(
            "strategy._reversal_recovery_score",
            return_value={"score": recovery_score},
        ), patch(
            "strategy.validate_dca_structure_level",
            return_value=(structure_ok, structure),
        ):
            return validate_dca_continuation_guard(
                "BUY",
                current_price,
                110,
                trend_df,
                confirm_df,
                entry_df,
                leverage=5,
                confirmation_type=confirmation_type,
                dca_level=dca_level,
                adverse_roi=adverse_roi,
                position_adverse_roi=adverse_roi,
                trigger_roi=trigger_roi,
                spacing_anchor_price=spacing_anchor,
            )

    def test_all_adaptive_confirmations_allow_dca(self):
        allowed, info = self.evaluate()

        self.assertTrue(allowed)
        self.assertEqual(
            info["adaptive"]["reason"],
            "DCA_ADAPTIVE_TRIGGER_OK",
        )

    def test_atr_spacing_blocks_clustered_dca(self):
        allowed, info = self.evaluate(current_price=108)

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "DCA_ADAPTIVE_ATR_SPACING_NOT_REACHED",
        )

    def test_structure_is_required_for_trend_dca(self):
        allowed, info = self.evaluate(structure_ok=False)

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "NO DCA SUPPORT LEVEL NEAR CURRENT PRICE",
        )

    def test_legacy_reversal_only_switch_cannot_bypass_adaptive_trend_guard(self):
        with patch.object(
            config,
            "DCA_STRICT_GUARD_APPLY_TO_REVERSAL_ONLY",
            True,
        ):
            allowed, info = self.evaluate(structure_ok=False)

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "NO DCA SUPPORT LEVEL NEAR CURRENT PRICE",
        )

    def test_later_levels_require_stronger_recovery(self):
        allowed, info = self.evaluate(
            current_price=95,
            dca_level=2,
            adverse_roi=50,
            trigger_roi=50,
            recovery_score=2.5,
        )

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "DCA_ADAPTIVE_RECOVERY_NOT_CONFIRMED",
        )
        self.assertEqual(info["adaptive"]["required_recovery"], 2.75)

    def test_invalidated_daily_trend_thesis_blocks_dca(self):
        allowed, info = self.evaluate(thesis_valid=False)

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "DCA_ADAPTIVE_1D_THESIS_INVALIDATED",
        )

    def test_maximum_risk_boundary_blocks_additional_dca(self):
        allowed, info = self.evaluate(
            adverse_roi=151,
            trigger_roi=120,
        )

        self.assertFalse(allowed)
        self.assertEqual(
            info["reason"],
            "DCA_ADAPTIVE_MAX_RISK_EXCEEDED",
        )

    def test_static_mode_keeps_legacy_structure_policy(self):
        with patch.object(config, "DCA_TRIGGER_MODE", "static_roi"):
            allowed, info = self.evaluate(
                structure_ok=False,
                recovery_score=0,
            )

        self.assertTrue(allowed)
        self.assertEqual(info["adaptive"]["reason"], "DCA_ADAPTIVE_DISABLED")


class AdaptiveDcaBacktestTests(unittest.TestCase):
    @patch(
        "backtest.validate_dca_continuation_guard",
        return_value=(True, {"reason": "DCA_ADAPTIVE_TRIGGER_OK"}),
    )
    @patch("backtest.compute_stop_loss", return_value=(None, "SL_DISABLED"))
    @patch("backtest.compute_take_profit", return_value=(200, "TEST_TP"))
    def test_replay_uses_original_entry_for_each_roi_floor(
        self,
        _take_profit,
        _stop_loss,
        _guard,
    ):
        interval_ms = 3_600_000
        rows = []

        for index in range(305):
            low = 99.5

            if index == 300:
                low = 97.5
            elif index >= 301:
                low = 95.5

            rows.append({
                "time": index * interval_ms,
                "close_time": (index + 1) * interval_ms - 1,
                "open": 100,
                "high": 100.5,
                "low": low,
                "close": max(low + 0.5, 96),
            })

        data = pd.DataFrame(rows)
        frame = BacktestData(raw=data, indicators=data)
        frames = {
            "trend": frame,
            "confirm": frame,
            "entry": frame,
            "exit": frame,
        }
        settings = {
            "LEVERAGE": 5,
            "DCA_ENABLED": True,
            "BACKTEST_USE_DCA": True,
            "DCA_MAX_ORDERS": 2,
            "DCA_MARGIN_PCTS": [50, 50],
            "DCA_TRIGGER_ROIS": [10, 20],
            "DCA_TRIGGER_MODE": "adaptive_hybrid",
            "DCA_MAX_ADVERSE_ROI": 150,
            "DCA_MIN_SECONDS_BETWEEN_ORDERS": 3600,
            "DCA_STRICT_GUARD_ENABLED": True,
            "DCA_REPRICE_TP_AFTER_FILL": False,
            "BACKTEST_MAX_HOLD_CANDLES": 3,
            "MULTI_TP_ENABLED": False,
            "BACKTEST_MULTI_TP_ENABLED": False,
            "EARLY_FLOW_EXIT_ENABLED": False,
            "TREND_PROFIT_PROTECTION_ENABLED": False,
        }
        config_patches = [
            patch.object(config, name, value)
            for name, value in settings.items()
        ]

        for config_patch in config_patches:
            config_patch.start()

        try:
            result = simulate_trade(
                "BTCUSDT",
                "BUY",
                "TREND",
                100,
                300 * interval_ms,
                100,
                300,
                frames,
            )
        finally:
            for config_patch in reversed(config_patches):
                config_patch.stop()

        self.assertEqual(result["dca_count"], 2)


class AdaptiveDcaLiveManagerTests(unittest.TestCase):
    def test_live_guard_uses_latest_fill_for_atr_spacing(self):
        symbol = "BTCUSDT"
        state = {
            "positions": {
                symbol: {
                    "managed_by_bot": True,
                    "side": "BUY",
                    "dca_count": 1,
                    "initial_entry": 100,
                    "last_dca_price": 90,
                    "confirmation_type": "TREND",
                }
            }
        }
        position = {
            "side": "BUY",
            "entry_price": 95,
            "amount": 1,
        }

        with patch.object(config, "DCA_ENABLED", True), patch.object(
            config,
            "TP1_RUNNER_DISABLE_DCA",
            True,
        ), patch.object(config, "DCA_MAX_ADVERSE_ROI", 0), patch.object(
            config,
            "DCA_MIN_SECONDS_BETWEEN_ORDERS",
            0,
        ), patch.object(config, "DCA_STRICT_GUARD_ENABLED", True), patch.object(
            config,
            "DCA_TRIGGER_MODE",
            "adaptive_hybrid",
        ), patch(
            "main.get_dca_order_margin",
            return_value=2,
        ), patch(
            "main.get_dca_trigger_roi",
            return_value=50,
        ), patch(
            "main.get_signal_frames",
            return_value=(object(), object(), object()),
        ), patch(
            "main.validate_dca_continuation_guard",
            return_value=(False, {"reason": "TEST_STOP"}),
        ) as guard, patch("main.log_info"), patch("main.log_warning"):
            main.manage_dca_position(
                symbol,
                state,
                position,
                None,
                None,
                current_price_override=80,
                price_source="websocket",
            )

        self.assertEqual(guard.call_count, 1)
        self.assertEqual(
            guard.call_args.kwargs["spacing_anchor_price"],
            90,
        )
        self.assertEqual(guard.call_args.kwargs["trigger_roi"], 50)


if __name__ == "__main__":
    unittest.main()
