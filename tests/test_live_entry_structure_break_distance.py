import unittest
from unittest.mock import patch

import pandas as pd

import config
from strategy import _live_entry_timeframe_check


def live_entry_data(rows=20):
    # high=101.0, low=99.0 on every row -> _average_range gives atr=2.0
    return pd.DataFrame([
        {
            "open": 99.8,
            "high": 101.0,
            "low": 99.0,
            "close": 100.2,
            "ema20": 100.0,
            "macd": 0.2,
            "macd_signal": 0.1,
            "rsi": 55.0,
        }
        for _ in range(rows)
    ])


class LiveEntryStructureBreakDistanceTests(unittest.TestCase):
    """structure_break/ema_wrong_side were only ever logged as booleans,
    which made it impossible to tell a near-miss apart from a wide-margin
    failure the way chase_atr/support_score could be. This adds a signed,
    ATR-scaled distance: negative = safe margin, positive = broken by that
    much, so a future check can do the same near-miss analysis on these
    two reasons that already worked for the others."""

    def test_negative_distance_when_price_is_safely_inside_structure(self):
        with patch.object(config, "LIVE_ENTRY_STRUCTURE_BUFFER_ATR", 0.08):
            result = _live_entry_timeframe_check(
                "BUY",
                live_entry_data(),
                101.0,  # well above recent_low=99.0
                "5m",
            )

        self.assertFalse(result["structure_break"])
        self.assertLess(result["structure_break_distance_atr"], 0)

    def test_positive_distance_when_structure_is_broken(self):
        with patch.object(config, "LIVE_ENTRY_STRUCTURE_BUFFER_ATR", 0.08):
            result = _live_entry_timeframe_check(
                "BUY",
                live_entry_data(),
                98.5,  # below recent_low(99.0) - buffer(0.16)
                "5m",
            )

        self.assertTrue(result["structure_break"])
        self.assertGreater(result["structure_break_distance_atr"], 0)

    def test_sell_side_positive_distance_when_structure_is_broken(self):
        with patch.object(config, "LIVE_ENTRY_STRUCTURE_BUFFER_ATR", 0.08):
            result = _live_entry_timeframe_check(
                "SELL",
                live_entry_data(),
                101.5,  # above recent_high(101.0) + buffer(0.16)
                "5m",
            )

        self.assertTrue(result["structure_break"])
        self.assertGreater(result["structure_break_distance_atr"], 0)

    def test_insufficient_data_path_still_returns_the_new_field(self):
        result = _live_entry_timeframe_check(
            "BUY",
            live_entry_data(rows=1),
            101.0,
            "5m",
        )

        self.assertEqual(result["reason"], "INSUFFICIENT_DATA")
        self.assertEqual(result["structure_break_distance_atr"], 0)


if __name__ == "__main__":
    unittest.main()
