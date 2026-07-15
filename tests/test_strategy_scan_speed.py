import unittest
from unittest.mock import patch

import pandas as pd

import config
import strategy


def sample_frame(close=100.0):
    return pd.DataFrame([
        {
            "time": 1,
            "close_time": 2,
            "open": 99.0,
            "high": 101.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 80.0,
            "volume_sma": 100.0,
        },
        {
            "time": 3,
            "close_time": 4,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": close,
            "volume": 120.0,
            "volume_sma": 100.0,
        },
    ])


class VectorLevelScoringTests(unittest.TestCase):
    def test_vector_scores_match_scalar_rules(self):
        data = pd.DataFrame([
            {
                "open": 100.0,
                "high": 103.0,
                "low": 99.0,
                "close": 102.0,
                "volume": 120.0,
                "volume_sma": 100.0,
            },
            {
                "open": 101.0,
                "high": 102.0,
                "low": 99.4,
                "close": 100.0,
                "volume": 80.0,
                "volume_sma": 100.0,
            },
            {
                "open": 99.0,
                "high": 101.0,
                "low": 97.0,
                "close": 98.0,
                "volume": 150.0,
                "volume_sma": 100.0,
            },
        ])
        level = 99.0
        tolerance = 0.5
        positions = [
            index
            for index, value in enumerate(data["low"])
            if abs(float(value) - level) <= tolerance
        ]
        reactions = []

        for _, candle in data.iterrows():
            if abs(float(candle["low"]) - level) > tolerance:
                continue

            candle_range = float(candle["high"] - candle["low"])
            directional_close = float(candle["close"] - candle["low"]) / candle_range
            wick = (
                float(min(candle["open"], candle["close"]) - candle["low"])
                / candle_range
            )
            reactions.append(max(directional_close, wick, 0))

        expected_reaction = min(sum(reactions) / len(reactions), 1)
        expected_volume = min(
            sum(
                1
                for _, candle in data.iterrows()
                if abs(float(candle["low"]) - level) <= tolerance
                and float(candle["volume"]) > float(candle["volume_sma"])
            ) / 3,
            1,
        )

        self.assertEqual(
            strategy._touch_positions(data, "low", level, tolerance),
            positions,
        )
        self.assertAlmostEqual(
            strategy._level_reaction_score(data, "BUY", level, tolerance),
            expected_reaction,
        )
        self.assertAlmostEqual(
            strategy._volume_touch_score(data, "low", level, tolerance),
            expected_volume,
        )


class SignalAnalysisCacheTests(unittest.TestCase):
    def setUp(self):
        strategy.clear_signal_analysis_cache()

    def tearDown(self):
        strategy.clear_signal_analysis_cache()

    def test_reuses_unchanged_frames_and_returns_isolated_copy(self):
        frame = sample_frame()
        result = {"signal": "BUY", "buy": {"confidence": 90}}

        with patch.object(config, "SIGNAL_ANALYSIS_CACHE_ENABLED", True), patch.object(
            config,
            "SIGNAL_ANALYSIS_CACHE_MAX_ITEMS",
            10,
        ), patch.object(strategy, "analyze_signal", return_value=result) as analyze:
            first = strategy.analyze_signal_cached(
                frame,
                frame,
                frame,
                "BULLISH",
                0.5,
                1.0,
                cache_namespace="BTCUSDT",
            )
            first["buy"]["confidence"] = 1
            second = strategy.analyze_signal_cached(
                frame,
                frame,
                frame,
                "BULLISH",
                0.5,
                1.0,
                cache_namespace="BTCUSDT",
            )

        self.assertEqual(analyze.call_count, 1)
        self.assertEqual(second["buy"]["confidence"], 90)

    def test_candle_or_symbol_change_invalidates_cache(self):
        frame = sample_frame()
        changed = sample_frame(close=101.0)

        with patch.object(config, "SIGNAL_ANALYSIS_CACHE_ENABLED", True), patch.object(
            strategy,
            "analyze_signal",
            return_value={"signal": None},
        ) as analyze:
            strategy.analyze_signal_cached(
                frame, frame, frame, "BULLISH", 0.5, 1.0,
                cache_namespace="BTCUSDT",
            )
            strategy.analyze_signal_cached(
                changed, frame, frame, "BULLISH", 0.5, 1.0,
                cache_namespace="BTCUSDT",
            )
            strategy.analyze_signal_cached(
                changed, frame, frame, "BULLISH", 0.5, 1.0,
                cache_namespace="ETHUSDT",
            )

        self.assertEqual(analyze.call_count, 3)


if __name__ == "__main__":
    unittest.main()
