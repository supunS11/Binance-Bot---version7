import unittest
from unittest.mock import patch

import config
from strategy import _reversal_regime_score


class ReversalRegimeScoreTests(unittest.TestCase):
    def test_trending_against_side_is_rewarded_not_penalized(self):
        # This is the actual reversal thesis: higher timeframes trending
        # against the trade direction, matching what _counter_trend_context
        # requires. Must not be penalized the way TREND's own regime score
        # would penalize a trend-following entry lacking alignment.
        score = _reversal_regime_score(
            "BUY",
            {
                "regime": "trending",
                "trend_aligned": False,
                "confirm_aligned": False,
            },
        )
        self.assertEqual(score, 0.5)

    def test_trending_with_side_is_penalized(self):
        # Trend already agrees with this side - there's no reversal
        # opportunity here, just piling onto an existing trend.
        score = _reversal_regime_score(
            "BUY",
            {
                "regime": "trending",
                "trend_aligned": True,
                "confirm_aligned": True,
            },
        )
        self.assertEqual(score, -1.25)

    def test_trending_mixed_alignment_is_neutral(self):
        score = _reversal_regime_score(
            "SELL",
            {
                "regime": "trending",
                "trend_aligned": True,
                "confirm_aligned": False,
            },
        )
        self.assertEqual(score, 0)

    def test_late_entry_still_penalized_same_as_trend(self):
        with patch.object(config, "LATE_ENTRY_SCORE_PENALTY", 2.0):
            score = _reversal_regime_score(
                "BUY",
                {"regime": "late_entry"},
            )
        self.assertEqual(score, -2.0)

    def test_sideways_rewards_breakout(self):
        self.assertEqual(
            _reversal_regime_score(
                "BUY", {"regime": "sideways", "breakout": True}
            ),
            0.5,
        )
        self.assertEqual(
            _reversal_regime_score(
                "BUY", {"regime": "sideways", "breakout": False}
            ),
            -1.0,
        )

    def test_breakout_regime_rewards_confirm_alignment(self):
        self.assertEqual(
            _reversal_regime_score(
                "BUY", {"regime": "breakout", "confirm_aligned": True}
            ),
            0.75,
        )
        self.assertEqual(
            _reversal_regime_score(
                "BUY", {"regime": "breakout", "confirm_aligned": False}
            ),
            0,
        )

    def test_transition_regime_and_missing_context_are_neutral(self):
        self.assertEqual(
            _reversal_regime_score("BUY", {"regime": "transition"}), 0
        )
        self.assertEqual(_reversal_regime_score("BUY", None), 0)
        self.assertEqual(_reversal_regime_score("BUY", {}), 0)


if __name__ == "__main__":
    unittest.main()
