import unittest

from strategy import (
    _apply_trend_exhaustion_guard,
    _reversal_warning_context,
)


def build_side(
    side,
    *,
    trend_following_ok=False,
    reversal_ok=False,
    reversal_confirmed=False,
    counter_count=0,
    smc=False,
    momentum=False,
    recovery_score=0,
    pressure_score=0,
    participation_available=False,
    participation_score=0,
):
    return {
        "side": side,
        "trend_following_ok": trend_following_ok,
        "reversal_ok": reversal_ok,
        "reversal_confirmed": reversal_confirmed,
        "trend_confidence": 82,
        "reversal_confidence": 78,
        "confirm_score": 7,
        "entry_score": 5,
        "quality_score": 1,
        "level_ok": True,
        "participation_available": participation_available,
        "participation_score": participation_score,
        "reversal_context": {
            "counter_trend": {"count": counter_count},
            "extra_confirmation": {
                "items": {
                    "smc": smc,
                    "momentum": momentum,
                }
            },
            "invalidation": {
                "recovery_score": recovery_score,
                "pressure_score": pressure_score,
            },
        },
    }


class ReversalWarningTests(unittest.TestCase):
    def test_warning_activates_before_full_reversal_confirmation(self):
        side = build_side(
            "SELL",
            counter_count=3,
            smc=True,
            momentum=True,
            recovery_score=3,
        )

        warning = _reversal_warning_context(side)

        self.assertTrue(warning["active"])
        self.assertEqual(warning["stage"], "WARNING")
        self.assertGreaterEqual(warning["points"], warning["required"])

    def test_live_futures_conflict_vetoes_warning(self):
        side = build_side(
            "SELL",
            counter_count=3,
            smc=True,
            momentum=True,
            recovery_score=3,
            participation_available=True,
            participation_score=-2,
        )

        warning = _reversal_warning_context(side)

        self.assertFalse(warning["active"])
        self.assertIn("FUTURES_CONFLICT=-2.0", warning["reasons"])

    def test_opposite_warning_blocks_only_the_new_trend_entry(self):
        buy = build_side("BUY", trend_following_ok=True)
        sell = build_side(
            "SELL",
            counter_count=3,
            smc=True,
            momentum=True,
            recovery_score=3,
        )

        buy, sell = _apply_trend_exhaustion_guard(buy, sell)

        self.assertFalse(buy["trend_following_ok"])
        self.assertFalse(buy["hard_ok"])
        self.assertEqual(buy["confirmation_type"], "NONE")
        self.assertTrue(buy["trend_exhaustion"]["blocked"])
        self.assertFalse(sell["reversal_ok"])

    def test_confirmed_reversal_is_marked_as_warning_only_when_not_orderable(self):
        side = build_side(
            "BUY",
            reversal_confirmed=True,
            counter_count=3,
            smc=True,
            momentum=True,
            recovery_score=4,
        )

        warning = _reversal_warning_context(side)

        self.assertTrue(warning["active"])
        self.assertEqual(warning["stage"], "CONFIRMED_WARNING_ONLY")

    def test_smc_without_momentum_recovery_or_futures_is_not_enough(self):
        side = build_side(
            "SELL",
            counter_count=3,
            smc=True,
        )

        warning = _reversal_warning_context(side)

        self.assertFalse(warning["active"])
        self.assertIn("DIRECTIONAL_TRIGGER_MISSING", warning["reasons"])


if __name__ == "__main__":
    unittest.main()
