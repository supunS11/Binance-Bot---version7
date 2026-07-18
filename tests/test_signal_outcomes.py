import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import signal_calibration
import signal_outcomes


class SignalOutcomeTests(unittest.TestCase):
    def setUp(self):
        signal_calibration._cache = None
        signal_outcomes._state_cache = None

    def tearDown(self):
        signal_calibration._cache = None
        signal_outcomes._state_cache = None

    def test_horizon_observation_records_return_excursions_and_calibration(self):
        candidate = {
            "symbol": "BTCUSDT",
            "signal": "BUY",
            "rank_score": 112,
            "analysis": {
                "buy": {
                    "score": 42,
                    "uncapped_score_index": 100,
                    "confirmation_type": "TREND",
                }
            },
            "market_context": {
                "flow": {"buy_score": 2},
                "breadth": {"buy_score": 1},
                "transition": {"buy_score": 0.5},
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            outcome_path = f"{directory}/outcomes.csv"
            state_path = f"{directory}/pending.json"
            calibration_path = f"{directory}/calibration.json"

            with patch("config.SIGNAL_OUTCOME_PATH", outcome_path), patch(
                "config.SIGNAL_OUTCOME_STATE_PATH",
                state_path,
            ), patch(
                "config.SIGNAL_OUTCOME_HORIZON_HOURS",
                [0.01],
            ), patch(
                "config.SIGNAL_CALIBRATION_PATH",
                calibration_path,
            ), patch(
                "config.SIGNAL_CALIBRATION_MIN_SAMPLES",
                1,
            ), patch(
                "signal_outcomes.time.time",
                return_value=1000,
            ):
                signal_id = signal_outcomes.register_signal_outcome(candidate, 100)
                self.assertIn(
                    signal_id,
                    signal_outcomes._state_cache.get("pending", {}),
                )

            with patch("config.SIGNAL_OUTCOME_PATH", outcome_path), patch(
                "config.SIGNAL_OUTCOME_STATE_PATH",
                state_path,
            ), patch(
                "config.SIGNAL_OUTCOME_HORIZON_HOURS",
                [0.01],
            ), patch(
                "config.SIGNAL_CALIBRATION_PATH",
                calibration_path,
            ), patch(
                "config.SIGNAL_CALIBRATION_MIN_SAMPLES",
                1,
            ), patch(
                "signal_outcomes.time.time",
                return_value=1040,
            ):
                signal_outcomes.observe_signal_outcomes(
                    "BTCUSDT",
                    102,
                    high_price=103,
                    low_price=99,
                )
                probability = signal_calibration.calibration_probability(
                    "TREND",
                    42,
                )

            self.assertNotIn(
                signal_id,
                signal_outcomes._state_cache.get("pending", {}),
            )

            with Path(outcome_path).open(
                "r",
                newline="",
                encoding="utf-8",
            ) as handle:
                rows = list(csv.DictReader(handle))

            self.assertTrue(signal_id)
            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["directional_return_pct"]), 2.0)
            self.assertEqual(float(rows[0]["mfe_pct"]), 3.0)
            self.assertEqual(float(rows[0]["mae_pct"]), -1.0)
            self.assertTrue(probability["available"])


if __name__ == "__main__":
    unittest.main()
