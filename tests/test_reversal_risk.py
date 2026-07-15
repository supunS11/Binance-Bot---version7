import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import config
from backtest import compute_stop_loss, compute_take_profit
from strategy import (
    _reversal_futures_confirmation_context,
    _reversal_signal_check,
    evaluate_reversal_profit_protection,
)
from trade_state import (
    load_trade_state,
    save_trade_state,
    update_position_runtime_fields,
)


class ReversalEntryQualityTests(unittest.TestCase):
    @patch("strategy.validate_reversal_invalidation")
    @patch("strategy._reversal_smc_check")
    @patch("strategy._counter_trend_context")
    def test_momentum_cannot_bypass_required_smc(
        self,
        counter_context,
        smc_check,
        invalidation,
    ):
        counter_context.return_value = (True, {"count": 3})
        smc_check.return_value = (False, {"has_support": False})
        invalidation.return_value = (
            True,
            {
                "pressure_score": 2,
                "recovery_score": 5,
                "required_recovery": 3.75,
            },
        )
        momentum = {"ok": True, "score": 7}

        with patch.object(config, "REVERSAL_REQUIRE_SMC", True), patch.object(
            config,
            "REVERSAL_ALLOW_MOMENTUM_WITHOUT_SMC",
            False,
        ), patch.object(
            config,
            "REVERSAL_MIN_CONFIRMATION_POINTS",
            6,
        ):
            allowed, reasons, _ = _reversal_signal_check(
                "BUY",
                object(),
                object(),
                object(),
                trend_score=5,
                confirm_score=10,
                entry_score=8,
                quality_score=3,
                smc_score=0,
                smc_context={},
                momentum_context=momentum,
                regime_score=1,
                confidence=95,
                confirm_ok=True,
                entry_ok=True,
                level_ok=True,
            )

        self.assertFalse(allowed)
        self.assertIn("SMC_REVERSAL_EVIDENCE_MISSING", reasons)

    def test_reversal_futures_confirmation_waits_for_live_data(self):
        result = _reversal_futures_confirmation_context(
            True,
            0,
            None,
            True,
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["active"])
        self.assertEqual(
            result["reason"],
            "REVERSAL_FUTURES_AWAITING_CONTEXT",
        )

    def test_reversal_futures_confirmation_requires_positive_score(self):
        with patch.object(config, "REVERSAL_MIN_FUTURES_SCORE", 0.5):
            blocked = _reversal_futures_confirmation_context(
                True,
                0.25,
                {"available": True},
                True,
            )
            allowed = _reversal_futures_confirmation_context(
                True,
                0.75,
                {"available": True},
                True,
            )

        self.assertFalse(blocked["active"])
        self.assertTrue(allowed["active"])


class ReversalProfitProtectionTests(unittest.TestCase):
    def setUp(self):
        self.config_patches = [
            patch.object(config, "REVERSAL_PROFIT_PROTECTION_ENABLED", True),
            patch.object(config, "REVERSAL_PROFIT_PROTECTION_TRIGGER_ROI", 12),
            patch.object(config, "REVERSAL_PROFIT_PROTECTION_LOCK_ROI", 3),
            patch.object(config, "REVERSAL_PROFIT_PROTECTION_RETRACE_PCT", 50),
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def test_profit_guard_is_not_armed_before_trigger(self):
        result = evaluate_reversal_profit_protection(
            "BUY",
            100,
            101,
            leverage=10,
        )

        self.assertEqual(result["current_roi"], 10)
        self.assertFalse(result["armed"])
        self.assertFalse(result["should_exit"])

    def test_profit_guard_holds_while_above_retrace_floor(self):
        result = evaluate_reversal_profit_protection(
            "BUY",
            100,
            101.2,
            peak_roi=20,
            leverage=10,
        )

        self.assertTrue(result["armed"])
        self.assertEqual(result["floor_roi"], 10)
        self.assertFalse(result["should_exit"])

    def test_profit_guard_exits_after_peak_retrace(self):
        result = evaluate_reversal_profit_protection(
            "SELL",
            100,
            99.4,
            peak_roi=20,
            leverage=10,
        )

        self.assertTrue(result["armed"])
        self.assertEqual(result["current_roi"], 6)
        self.assertEqual(result["floor_roi"], 10)
        self.assertTrue(result["should_exit"])

    @patch("backtest.validate_structure_take_profit")
    def test_reversal_structure_tp_is_capped(self, structure_tp):
        structure_tp.return_value = (
            True,
            {
                "target_price": 110,
                "target_roi": 100,
                "source": "TEST",
            },
        )

        with patch.object(config, "STATIC_TP_ENABLED", False), patch.object(
            config,
            "REVERSAL_TP_MAX_ROI",
            45,
        ), patch.object(config, "LEVERAGE", 10):
            price, mode = compute_take_profit(
                "BUY",
                100,
                object(),
                object(),
                confirmation_type="REVERSAL",
            )

        self.assertAlmostEqual(price, 104.5)
        self.assertEqual(mode, "REVERSAL_STRUCTURE_CAPPED_45.0%")

    @patch("backtest.structure_stop_loss_price")
    def test_reversal_structure_sl_is_capped(self, structure_sl):
        structure_sl.return_value = 85

        with patch.object(config, "REVERSAL_SL_ENABLED", True), patch.object(
            config,
            "REVERSAL_MAX_SL_ROI",
            80,
        ), patch.object(config, "LEVERAGE", 10):
            price, mode = compute_stop_loss(
                "BUY",
                100,
                object(),
                "REVERSAL",
            )

        self.assertAlmostEqual(price, 92)
        self.assertEqual(mode, "STRUCTURE_SL_CAPPED_80.0%")

    def test_reversal_peak_survives_state_reload(self):
        with TemporaryDirectory() as temp_dir, patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(temp_dir) / "state.json"),
        ):
            state = {
                "positions": {
                    "BTCUSDT": {
                        "symbol": "BTCUSDT",
                        "reversal_peak_roi": 0,
                    }
                }
            }
            save_trade_state(state)
            updated = update_position_runtime_fields(
                state,
                "BTCUSDT",
                {"reversal_peak_roi": 18.5},
            )
            reloaded = load_trade_state()

        self.assertTrue(updated)
        self.assertEqual(
            reloaded["positions"]["BTCUSDT"]["reversal_peak_roi"],
            18.5,
        )


if __name__ == "__main__":
    unittest.main()
