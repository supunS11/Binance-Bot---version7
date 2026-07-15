import unittest
from unittest.mock import patch

import config
from strategy import (
    _trend_timing_rescue_context,
    should_fetch_futures_context,
    validate_live_entry_guard,
)


def rescue_context(**overrides):
    values = {
        "trend_ok": True,
        "confirm_ok": False,
        "entry_ok": True,
        "level_ok": True,
        "trend_score": 8,
        "confirm_score": 6.75,
        "entry_score": 5,
        "quality_score": 2,
        "regime_score": 0,
        "trend_confidence": 82,
        "confirm_quality": {"direction_ok": True},
        "entry_quality": {
            "direction_ok": True,
            "late_entry_ok": True,
        },
        "participation_score": 1,
        "participation": {"available": True},
        "futures_ok": True,
    }
    values.update(overrides)
    return _trend_timing_rescue_context(**values)


class TrendTimingRescueTests(unittest.TestCase):
    def setUp(self):
        self.config_patches = [
            patch.object(config, "TREND_TIMING_RESCUE_ENABLED", True),
            patch.object(config, "TREND_TIMING_RESCUE_MIN_CONFIDENCE", 78),
            patch.object(config, "TREND_TIMING_RESCUE_SCORE_TOLERANCE", 0.5),
            patch.object(config, "TREND_TIMING_RESCUE_MIN_QUALITY_SCORE", 1),
            patch.object(config, "TREND_TIMING_RESCUE_MIN_REGIME_SCORE", -1.25),
            patch.object(config, "TREND_TIMING_RESCUE_REQUIRE_FUTURES", True),
            patch.object(config, "TREND_TIMING_RESCUE_MIN_FUTURES_SCORE", 0),
            patch.object(config, "SIGNAL_MIN_TREND_SCORE", 7.5),
            patch.object(config, "SIGNAL_MIN_CONFIRM_SCORE", 7),
            patch.object(config, "SIGNAL_MIN_ENTRY_SCORE", 4),
            patch.object(config, "LIVE_ENTRY_CONFIRMATION_ENABLED", True),
            patch.object(config, "LIVE_ENTRY_REQUIRE_DIRECTION_SUPPORT", True),
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def test_one_marginal_timing_module_with_futures_support_is_active(self):
        result = rescue_context()

        self.assertTrue(result["eligible"])
        self.assertTrue(result["active"])
        self.assertEqual(result["missed_module"], "CONFIRM")

    def test_eligible_rescue_waits_for_futures_context(self):
        result = rescue_context(
            participation_score=0,
            participation=None,
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["active"])
        self.assertEqual(
            result["reason"],
            "TREND_TIMING_RESCUE_AWAITING_FUTURES",
        )

    def test_two_timing_misses_are_blocked(self):
        result = rescue_context(
            entry_ok=False,
            entry_score=3.75,
        )

        self.assertFalse(result["eligible"])
        self.assertFalse(result["active"])
        self.assertIn("TIMING_MISSES=2 REQUIRED=1", result["reasons"])

    def test_direction_conflict_is_not_rescued(self):
        result = rescue_context(
            confirm_quality={"direction_ok": False},
        )

        self.assertFalse(result["eligible"])
        self.assertFalse(result["active"])
        self.assertIn(
            "CONFIRM_DIRECTION_HARD_CHECK_FAILED",
            result["reasons"],
        )

    def test_futures_conflict_blocks_an_eligible_rescue(self):
        result = rescue_context(
            participation_score=-1,
            futures_ok=False,
        )

        self.assertTrue(result["eligible"])
        self.assertFalse(result["active"])

    def test_eligible_rescue_requests_futures_context(self):
        result = rescue_context(
            participation_score=0,
            participation=None,
        )
        analysis = {
            "best_confidence": 82,
            "buy": {"trend_timing_rescue": result},
            "sell": {},
        }

        with patch.object(config, "FUTURES_CONTEXT_ENABLED", True), patch.object(
            config,
            "FUTURES_CONTEXT_MIN_CONFIDENCE",
            60,
        ):
            self.assertTrue(should_fetch_futures_context(analysis))

    @patch("strategy._live_entry_timeframe_check")
    def test_rescue_live_guard_requires_both_timeframes(self, timeframe_check):
        neutral = {
            "structure_break": False,
            "opposite_reversal": False,
            "ema_wrong_side": False,
            "ema_chase": False,
            "close_chase": False,
            "opposes_direction": False,
        }
        timeframe_check.side_effect = [
            {**neutral, "supports_direction": True},
            {**neutral, "supports_direction": False},
        ]

        allowed, _ = validate_live_entry_guard(
            "BUY",
            object(),
            object(),
            100,
        )
        self.assertTrue(allowed)

        timeframe_check.side_effect = [
            {**neutral, "supports_direction": True},
            {**neutral, "supports_direction": False},
        ]
        allowed, details = validate_live_entry_guard(
            "BUY",
            object(),
            object(),
            100,
            require_both_override=True,
        )

        self.assertFalse(allowed)
        self.assertEqual(
            details["reason"],
            "LIVE_DIRECTION_SUPPORT_MISSING_BOTH",
        )

    @patch("strategy._live_entry_timeframe_check")
    def test_single_timeframe_mode_still_blocks_dual_opposition(
        self,
        timeframe_check,
    ):
        opposing = {
            "structure_break": False,
            "opposite_reversal": False,
            "ema_wrong_side": False,
            "ema_chase": False,
            "close_chase": False,
            "supports_direction": False,
            "opposes_direction": True,
        }
        timeframe_check.side_effect = [opposing, opposing]

        allowed, details = validate_live_entry_guard(
            "BUY",
            object(),
            object(),
            100,
            require_both_override=False,
        )

        self.assertFalse(allowed)
        self.assertEqual(
            details["reason"],
            "DUAL_LIVE_DIRECTION_OPPOSITION",
        )

    @patch("strategy._live_entry_timeframe_check")
    def test_single_timeframe_mode_still_blocks_structure_failure(
        self,
        timeframe_check,
    ):
        neutral = {
            "structure_break": False,
            "opposite_reversal": False,
            "ema_wrong_side": False,
            "ema_chase": False,
            "close_chase": False,
            "supports_direction": True,
            "opposes_direction": False,
        }
        timeframe_check.side_effect = [
            {**neutral, "structure_break": True},
            neutral,
        ]

        allowed, details = validate_live_entry_guard(
            "BUY",
            object(),
            object(),
            100,
            require_both_override=False,
        )

        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "OPPOSITE_STRUCTURE_BREAK")


if __name__ == "__main__":
    unittest.main()
