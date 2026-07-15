import unittest
from unittest.mock import patch

import pandas as pd

import config
from strategy import (
    _continuation_pullback_context,
    futures_context_priority,
    should_fetch_futures_context,
)


def entry_frame(
    *,
    close=100.0,
    low=99.9,
    high=100.5,
    ema20=100.2,
    ema50=99.5,
    prior_low=98.8,
):
    rows = []

    for index in range(10):
        rows.append({
            "open": 100.0,
            "high": 101.0,
            "low": prior_low,
            "close": 100.5,
            "atr": 1.0,
            "ema20": 100.2,
            "ema50": 99.5,
            "rsi": 52.0,
            "volume": 100.0,
            "volume_sma": 100.0,
        })

    rows.append({
        "open": 100.4,
        "high": high,
        "low": low,
        "close": close,
        "atr": 1.0,
        "ema20": ema20,
        "ema50": ema50,
        "rsi": 50.0,
        "volume": 100.0,
        "volume_sma": 100.0,
    })
    rows.append(dict(rows[-1]))
    return pd.DataFrame(rows)


def pullback_context(**overrides):
    values = {
        "side": "BUY",
        "entry_df": entry_frame(),
        "trend_ok": True,
        "confirm_ok": True,
        "entry_ok": False,
        "level_ok": True,
        "trend_score": 9,
        "confirm_score": 9,
        "entry_score": 3,
        "quality_score": 2,
        "regime_score": 1,
        "trend_confidence": 84,
        "entry_quality": {"late_entry_ok": True},
        "participation_score": 1,
        "participation": {"available": True},
        "futures_ok": True,
    }
    values.update(overrides)
    return _continuation_pullback_context(**values)


class ContinuationPullbackTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "CONTINUATION_PULLBACK_ENABLED": True,
            "CONTINUATION_PULLBACK_MIN_CONFIDENCE": 80,
            "CONTINUATION_PULLBACK_MIN_TREND_SCORE": 8,
            "CONTINUATION_PULLBACK_MIN_CONFIRM_SCORE": 8,
            "CONTINUATION_PULLBACK_MIN_ENTRY_SCORE": 2.5,
            "CONTINUATION_PULLBACK_MIN_QUALITY_SCORE": 1,
            "CONTINUATION_PULLBACK_MIN_REGIME_SCORE": 0,
            "CONTINUATION_PULLBACK_MAX_EMA20_DISTANCE_ATR": 0.75,
            "CONTINUATION_PULLBACK_EMA20_TOUCH_ATR": 0.20,
            "CONTINUATION_PULLBACK_EMA50_BREAK_BUFFER_ATR": 0.15,
            "CONTINUATION_PULLBACK_STRUCTURE_LOOKBACK": 8,
            "CONTINUATION_PULLBACK_STRUCTURE_BREAK_BUFFER_ATR": 0.15,
            "CONTINUATION_PULLBACK_BUY_MIN_RSI": 44,
            "CONTINUATION_PULLBACK_BUY_MAX_RSI": 70,
            "CONTINUATION_PULLBACK_MAX_VOLUME_MULT": 1.35,
            "CONTINUATION_PULLBACK_MAX_CANDLE_ATR": 1.25,
            "CONTINUATION_PULLBACK_REQUIRE_FUTURES": True,
            "CONTINUATION_PULLBACK_MIN_FUTURES_SCORE": 0.5,
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

    def test_controlled_pullback_with_futures_support_is_active(self):
        result = pullback_context()

        self.assertTrue(result["eligible"])
        self.assertTrue(result["active"])
        self.assertTrue(result["ema20_touched"])
        self.assertTrue(result["ema50_hold_ok"])
        self.assertTrue(result["structure_ok"])

    def test_controlled_sell_pullback_with_futures_support_is_active(self):
        result = pullback_context(
            side="SELL",
            entry_df=entry_frame(
                close=100.0,
                low=99.5,
                high=100.1,
                ema20=99.8,
                ema50=100.5,
                prior_low=98.8,
            ),
        )

        self.assertTrue(result["eligible"])
        self.assertTrue(result["active"])
        self.assertTrue(result["ema20_touched"])
        self.assertTrue(result["ema50_hold_ok"])
        self.assertTrue(result["structure_ok"])

    def test_eligible_pullback_waits_for_futures_context(self):
        result = pullback_context(
            participation_score=0,
            participation=None,
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["active"])
        self.assertEqual(
            result["reason"],
            "CONTINUATION_PULLBACK_AWAITING_FUTURES",
        )

    def test_normal_entry_is_not_reclassified_as_pullback(self):
        result = pullback_context(entry_ok=True)

        self.assertFalse(result["eligible"])
        self.assertFalse(result["active"])
        self.assertIn("NORMAL_ENTRY_ALREADY_VALID", result["reasons"])

    def test_broken_entry_structure_blocks_pullback(self):
        result = pullback_context(
            entry_df=entry_frame(
                close=99.6,
                low=99.5,
                high=100.1,
                ema20=100.0,
                ema50=99.5,
                prior_low=100.2,
            )
        )

        self.assertFalse(result["eligible"])
        self.assertFalse(result["active"])
        self.assertIn("ENTRY_STRUCTURE_BROKEN", result["reasons"])

    def test_futures_conflict_blocks_eligible_pullback(self):
        result = pullback_context(
            participation_score=0,
            participation={"available": True},
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["active"])
        self.assertIn("FUTURES_SCORE=0.0 < 0.5", result["reasons"])

    def test_eligible_pullback_requests_futures_context(self):
        result = pullback_context(
            participation_score=0,
            participation=None,
        )
        analysis = {
            "best_confidence": 84,
            "buy": {"continuation_pullback": result},
            "sell": {},
        }

        with patch.object(config, "FUTURES_CONTEXT_ENABLED", True), patch.object(
            config,
            "FUTURES_CONTEXT_MIN_CONFIDENCE",
            60,
        ):
            self.assertTrue(should_fetch_futures_context(analysis))

    def test_pullback_bonus_prioritizes_context_request(self):
        base_side = {
            "side": "BUY",
            "trend_confidence": 80,
            "quality_score": 1,
            "smc_score": 1,
            "regime_score": 1,
            "continuation_pullback": {"eligible": False},
            "trend_timing_rescue": {"eligible": False},
        }
        pullback_side = {
            **base_side,
            "continuation_pullback": {"eligible": True},
        }

        with patch.object(config, "FUTURES_CONTEXT_PRIORITY_PULLBACK_BONUS", 8):
            normal_priority = futures_context_priority({
                "signal": None,
                "buy": base_side,
                "sell": {},
            })
            pullback_priority = futures_context_priority({
                "signal": None,
                "buy": pullback_side,
                "sell": {},
            })

        self.assertEqual(pullback_priority - normal_priority, 8)


if __name__ == "__main__":
    unittest.main()
