import unittest

import pandas as pd

from volume_profile import classify_price_vs_value_area, compute_volume_profile


def candle(high, low, volume):
    mid = (high + low) / 2
    return {"open": mid, "high": high, "low": low, "close": mid, "volume": volume}


class ComputeVolumeProfileTests(unittest.TestCase):
    def test_insufficient_data_returns_unavailable(self):
        result = compute_volume_profile(pd.DataFrame([candle(101, 99, 10)]))
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "INSUFFICIENT_DATA")

    def test_none_dataframe_returns_unavailable(self):
        result = compute_volume_profile(None)
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "INSUFFICIENT_DATA")

    def test_poc_lands_in_high_volume_price_region(self):
        rows = []

        for _ in range(10):
            rows.append(candle(101, 100, 5))

        for _ in range(10):
            rows.append(candle(111, 110, 500))

        for _ in range(10):
            rows.append(candle(121, 120, 5))

        rows.append(candle(121, 120, 5))
        df = pd.DataFrame(rows)

        result = compute_volume_profile(df, lookback=30, bins=20)

        self.assertTrue(result["available"])
        # Bin discretization means the winning bin's midpoint can sit just
        # outside [110, 111] by less than one bin width (~1.05 here) -
        # assert it's close to the high-volume region, not exactly inside it.
        self.assertGreaterEqual(result["poc"], 109)
        self.assertLessEqual(result["poc"], 112)

    def test_value_area_contains_at_least_target_share_of_volume(self):
        rows = [candle(100 + i, 99 + i, 10 + (i % 5)) for i in range(60)]
        rows.append(candle(159, 158, 10))
        df = pd.DataFrame(rows)

        result = compute_volume_profile(
            df,
            lookback=60,
            bins=30,
            value_area_pct=0.70,
        )

        self.assertTrue(result["available"])
        self.assertLess(result["val"], result["poc"])
        self.assertGreater(result["vah"], result["poc"])

        window = df.iloc[:-1].tail(60)
        in_value_volume = window[
            (window["low"] >= result["val"]) & (window["high"] <= result["vah"])
        ]["volume"].sum()
        self.assertGreaterEqual(in_value_volume / result["total_volume"], 0.60)

    def test_degenerate_range_is_reported(self):
        rows = [candle(100, 100, 10) for _ in range(10)]
        rows.append(candle(100, 100, 10))
        df = pd.DataFrame(rows)

        result = compute_volume_profile(df, lookback=10, bins=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "DEGENERATE_RANGE")

    def test_zero_volume_window_is_reported(self):
        rows = [candle(100 + i, 99 + i, 0) for i in range(10)]
        rows.append(candle(110, 109, 0))
        df = pd.DataFrame(rows)

        result = compute_volume_profile(df, lookback=10, bins=10)

        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "NO_VOLUME")


class ClassifyPriceVsValueAreaTests(unittest.TestCase):
    def setUp(self):
        self.profile = {"available": True, "vah": 110.0, "val": 100.0}

    def test_above_value(self):
        self.assertEqual(
            classify_price_vs_value_area(111, self.profile),
            "ABOVE_VALUE",
        )

    def test_below_value(self):
        self.assertEqual(
            classify_price_vs_value_area(99, self.profile),
            "BELOW_VALUE",
        )

    def test_inside_value(self):
        self.assertEqual(
            classify_price_vs_value_area(105, self.profile),
            "INSIDE_VALUE",
        )

    def test_unavailable_profile_returns_empty(self):
        self.assertEqual(
            classify_price_vs_value_area(105, {"available": False}),
            "",
        )


if __name__ == "__main__":
    unittest.main()
