import unittest
from unittest.mock import patch

import pandas as pd

import config
from strategy import _range_reversion_context


def entry_frame(*, rsi=50.0):
    rows = []

    for _ in range(10):
        rows.append({
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
            "atr": 1.0,
            "ema20": 100.0,
            "rsi": 50.0,
            "volume": 100.0,
            "volume_sma": 100.0,
        })

    rows.append({
        "open": 100.0,
        "high": 100.5,
        "low": 99.5,
        "close": 100.0,
        "atr": 1.0,
        "ema20": 100.0,
        "rsi": rsi,
        "volume": 100.0,
        "volume_sma": 100.0,
    })
    rows.append(dict(rows[-1]))
    return pd.DataFrame(rows)


def range_context(**overrides):
    values = {
        "side": "BUY",
        "entry_df": entry_frame(rsi=25.0),
        "regime_context": {"regime": "sideways"},
        "confirm_score": 0,
        "quality_score": 0.5,
        "level_ok": True,
        "level": {"level": 98.0, "score": 4.0, "adverse_roi": 3.0},
    }
    values.update(overrides)
    return _range_reversion_context(**values)


class RangeReversionTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "RANGE_REVERSION_ENABLED": True,
            "RANGE_REVERSION_SIGNAL_THRESHOLD": 70,
            "RANGE_REVERSION_CONFIDENCE_MAX_SCORE": 12,
            "RANGE_REVERSION_MIN_SIGNAL_EDGE": 0,
            "RANGE_REVERSION_MAX_LEVEL_ADVERSE_ROI": 8.0,
            "RANGE_REVERSION_MIN_LEVEL_SCORE": 3.0,
            "RANGE_REVERSION_RSI_OVERSOLD": 32,
            "RANGE_REVERSION_RSI_OVERBOUGHT": 68,
            "RANGE_REVERSION_MIN_QUALITY_SCORE": 0.0,
            "RANGE_REVERSION_MIN_CONFIRM_SCORE": -3.0,
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

    def test_disabled_by_default_flag(self):
        with patch.object(config, "RANGE_REVERSION_ENABLED", False):
            result = range_context()

        self.assertFalse(result["enabled"])
        self.assertFalse(result["active"])
        self.assertIn("RANGE_REVERSION_DISABLED", result["reasons"])

    def test_qualified_buy_fade_is_active(self):
        result = range_context()

        self.assertTrue(result["active"])
        self.assertGreater(result["confidence"], 0)
        self.assertLessEqual(result["confidence"], 100)

    def test_qualified_sell_fade_is_active(self):
        result = range_context(
            side="SELL",
            entry_df=entry_frame(rsi=75.0),
            level={"level": 102.0, "score": 4.0, "adverse_roi": 3.0},
        )

        self.assertTrue(result["active"])

    def test_trending_regime_blocks_fade(self):
        result = range_context(regime_context={"regime": "trending"})

        self.assertFalse(result["active"])
        self.assertIn("REGIME_NOT_SIDEWAYS=trending", result["reasons"])

    def test_missing_level_blocks_fade(self):
        result = range_context(level_ok=False, level={"reason": "NO LEVEL"})

        self.assertFalse(result["active"])
        self.assertIn("NO_RANGE_BOUNDARY_LEVEL", result["reasons"])

    def test_level_too_far_blocks_fade(self):
        result = range_context(
            level={"level": 90.0, "score": 4.0, "adverse_roi": 15.0}
        )

        self.assertFalse(result["active"])
        self.assertTrue(
            any(r.startswith("LEVEL_TOO_FAR") for r in result["reasons"])
        )

    def test_weak_level_score_blocks_fade(self):
        result = range_context(
            level={"level": 98.0, "score": 1.0, "adverse_roi": 3.0}
        )

        self.assertFalse(result["active"])
        self.assertTrue(
            any(r.startswith("LEVEL_SCORE_LOW") for r in result["reasons"])
        )

    def test_rsi_not_extreme_blocks_fade(self):
        result = range_context(entry_df=entry_frame(rsi=50.0))

        self.assertFalse(result["active"])
        self.assertTrue(
            any(r.startswith("RSI_NOT_EXTREME") for r in result["reasons"])
        )

    def test_weak_confirm_score_blocks_fade(self):
        result = range_context(confirm_score=-10)

        self.assertFalse(result["active"])
        self.assertTrue(
            any(
                r.startswith("CONFIRM_SCORE_TOO_WEAK")
                for r in result["reasons"]
            )
        )


if __name__ == "__main__":
    unittest.main()
