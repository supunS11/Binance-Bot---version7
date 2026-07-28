import unittest

import pandas as pd

from strategy import _collect_ema_levels, _collect_pivot_levels


def _row(low, high, close, ema50=100.0, ema200=100.0, volume=10, volume_sma=10):
    return {
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "atr": 1.0,
        "ema50": ema50,
        "ema200": ema200,
        "volume": volume,
        "volume_sma": volume_sma,
    }


def _closed_rows(count=20):
    # A stable oscillating pattern with a repeated swing low at 99.5 and a
    # flat EMA50/EMA200 - enough history for both collectors to find
    # something, none of it near the "live" candle values used below.
    rows = []
    for i in range(count):
        if i % 4 == 0:
            rows.append(_row(low=99.5, high=100.5, close=100.0))
        else:
            rows.append(_row(low=100.0, high=101.0, close=100.5))
    return rows


class StructureLevelCollectorsIgnoreLiveCandleTests(unittest.TestCase):
    """_collect_pivot_levels/_collect_ema_levels must be invariant to
    whatever the still-forming (live) last candle looks like - only the
    closed history before it should ever affect the returned levels. Two
    dataframes that share identical closed history but end in wildly
    different live candles must therefore produce identical output."""

    def test_pivot_levels_ignore_the_live_candle(self):
        closed = _closed_rows()
        live_low_wick = _row(low=50.0, high=100.4, close=99.9)
        live_high_wick = _row(low=99.6, high=200.0, close=100.1)

        levels_a = _collect_pivot_levels(
            pd.DataFrame(closed + [live_low_wick]), "BUY", "trend", 2.0
        )
        levels_b = _collect_pivot_levels(
            pd.DataFrame(closed + [live_high_wick]), "BUY", "trend", 2.0
        )

        self.assertEqual(levels_a, levels_b)
        # and neither run should have manufactured a level anywhere near
        # the live candle's outlier low.
        self.assertFalse(any(lvl["level"] < 90 for lvl in levels_a))

    def test_ema_levels_ignore_the_live_candle(self):
        closed = _closed_rows()
        live_break = _row(
            low=80.0, high=90.0, close=85.0, ema50=85.0, ema200=85.0
        )
        live_calm = _row(
            low=100.4, high=100.6, close=100.5, ema50=100.5, ema200=100.5
        )

        levels_a = _collect_ema_levels(
            pd.DataFrame(closed + [live_break]), "BUY", "trend", 2.0
        )
        levels_b = _collect_ema_levels(
            pd.DataFrame(closed + [live_calm]), "BUY", "trend", 2.0
        )

        self.assertEqual(levels_a, levels_b)


if __name__ == "__main__":
    unittest.main()
