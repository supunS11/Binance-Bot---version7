import tempfile
import unittest
from unittest.mock import patch

import signal_calibration


class SignalCalibrationTests(unittest.TestCase):
    def setUp(self):
        signal_calibration._cache = None

    def tearDown(self):
        signal_calibration._cache = None

    def test_probability_stays_unavailable_until_bucket_has_samples(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "config.SIGNAL_CALIBRATION_PATH",
            f"{directory}/calibration.json",
        ), patch("config.SIGNAL_CALIBRATION_MIN_SAMPLES", 3):
            signal_calibration.record_calibration_outcome("TREND", 42, True, 1)
            collecting = signal_calibration.calibration_probability("TREND", 42)
            signal_calibration.record_calibration_outcome("TREND", 42, True, 2)
            signal_calibration.record_calibration_outcome("TREND", 42, False, -1)
            ready = signal_calibration.calibration_probability("TREND", 42)

        self.assertFalse(collecting["available"])
        self.assertTrue(ready["available"])
        self.assertGreater(ready["probability"], 0.5)


if __name__ == "__main__":
    unittest.main()
