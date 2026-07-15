import unittest
from unittest.mock import patch

import config
from backtest import (
    apply_position_limits,
    record_reversal_diagnostics,
    reversal_diagnostics_summary,
)


def trade(side, confirmation_type, entry_ms, exit_ms):
    return {
        "side": side,
        "confirmation_type": confirmation_type,
        "entry_ms": entry_ms,
        "exit_ms": exit_ms,
    }


class BacktestPositionLimitTests(unittest.TestCase):
    def test_trend_and_reversal_use_separate_position_pools(self):
        trades = [
            trade("BUY", "TREND", 1, 10),
            trade("SELL", "REVERSAL", 2, 10),
            trade("SELL", "TREND", 3, 10),
            trade("BUY", "REVERSAL", 4, 10),
        ]

        with (
            patch.object(config, "BACKTEST_APPLY_POSITION_LIMITS", True),
            patch.object(config, "MAX_TOTAL_POSITIONS", 1),
            patch.object(config, "MAX_BUY_POSITIONS", 1),
            patch.object(config, "MAX_SELL_POSITIONS", 1),
            patch.object(config, "REVERSAL_EXTRA_TOTAL_POSITIONS", 1),
            patch.object(config, "REVERSAL_EXTRA_BUY_POSITIONS", 1),
            patch.object(config, "REVERSAL_EXTRA_SELL_POSITIONS", 1),
        ):
            accepted, skipped = apply_position_limits(trades)

        self.assertEqual(len(accepted), 2)
        self.assertEqual(skipped, 2)
        self.assertEqual(
            {item["confirmation_type"] for item in accepted},
            {"TREND", "REVERSAL"},
        )

    def test_zero_reversal_limit_disables_reversal_pool(self):
        trades = [trade("BUY", "REVERSAL", 1, 10)]

        with (
            patch.object(config, "BACKTEST_APPLY_POSITION_LIMITS", True),
            patch.object(config, "REVERSAL_EXTRA_TOTAL_POSITIONS", 0),
            patch.object(config, "REVERSAL_EXTRA_BUY_POSITIONS", 0),
            patch.object(config, "REVERSAL_EXTRA_SELL_POSITIONS", 0),
        ):
            accepted, skipped = apply_position_limits(trades)

        self.assertEqual(accepted, [])
        self.assertEqual(skipped, 1)

    def test_reversal_diagnostics_count_unique_rejection_reasons(self):
        diagnostics = {}
        analysis = {
            "buy": {
                "reversal_confirmed": False,
                "reversal_confidence": 84,
                "reversal_reasons": [
                    "CONFIDENCE=84 < 86",
                    "CONFIDENCE=84 < 86",
                    "ENTRY=4.5 < 5",
                ],
            },
            "sell": {
                "reversal_confirmed": True,
                "reversal_confidence": 90,
                "reversal_reasons": [],
            },
        }

        record_reversal_diagnostics(diagnostics, analysis)
        summary = reversal_diagnostics_summary(diagnostics)

        self.assertEqual(summary["evaluations"], 2)
        self.assertEqual(summary["chart_confirmed"], 1)
        self.assertEqual(summary["near_misses"], 1)
        self.assertEqual(summary["max_confidence"], 90)
        self.assertEqual(
            summary["top_rejection_reasons"],
            [
                {"reason": "CONFIDENCE", "count": 1},
                {"reason": "ENTRY", "count": 1},
            ],
        )


if __name__ == "__main__":
    unittest.main()
