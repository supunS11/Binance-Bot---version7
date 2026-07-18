import unittest
from unittest.mock import patch

import pandas as pd

import config
from strategy import _live_entry_timeframe_check


def live_entry_data(rows=20):
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


class LiveEntryChaseTests(unittest.TestCase):
    def test_half_atr_distance_remains_eligible(self):
        with patch.object(config, "MAX_LIVE_ENTRY_CHASE_ATR", 0.50):
            result = _live_entry_timeframe_check(
                "BUY",
                live_entry_data(),
                101.0,
                "5m",
            )

        self.assertEqual(result["ema_chase_atr"], 0.50)
        self.assertFalse(result["ema_chase"])
        self.assertFalse(result["block"])

    def test_distance_beyond_configured_limit_is_still_blocked(self):
        with patch.object(config, "MAX_LIVE_ENTRY_CHASE_ATR", 0.49):
            result = _live_entry_timeframe_check(
                "BUY",
                live_entry_data(),
                101.0,
                "5m",
            )

        self.assertTrue(result["ema_chase"])
        self.assertTrue(result["block"])


if __name__ == "__main__":
    unittest.main()
