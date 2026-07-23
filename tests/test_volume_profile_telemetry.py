import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import config
import volume_profile
from volume_profile import record_volume_profile_telemetry


def candle(high, low, volume):
    mid = (high + low) / 2
    return {"open": mid, "high": high, "low": low, "close": mid, "volume": volume}


def sample_df():
    rows = [candle(100 + i * 0.1, 99 + i * 0.1, 10) for i in range(60)]
    rows.append(candle(106, 105, 10))
    return pd.DataFrame(rows)


class VolumeProfileTelemetryTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.path = Path(self.tmpdir.name) / "volume_profile_v7.csv"
        volume_profile._LAST_TELEMETRY_WRITE_AT.clear()
        self.patches = [
            patch.object(config, "VOLUME_PROFILE_TELEMETRY_ENABLED", True),
            patch.object(config, "VOLUME_PROFILE_TELEMETRY_PATH", str(self.path)),
            patch.object(
                config,
                "VOLUME_PROFILE_TELEMETRY_MIN_INTERVAL_SECONDS",
                240,
            ),
            patch.object(config, "VOLUME_PROFILE_LOOKBACK", 60),
            patch.object(config, "VOLUME_PROFILE_BINS", 12),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

        self.tmpdir.cleanup()
        volume_profile._LAST_TELEMETRY_WRITE_AT.clear()

    def _read_rows(self):
        if not self.path.exists():
            return []

        with self.path.open(encoding="utf-8") as file:
            return list(csv.DictReader(file))

    def test_disabled_writes_nothing(self):
        with patch.object(config, "VOLUME_PROFILE_TELEMETRY_ENABLED", False):
            result = record_volume_profile_telemetry(
                "BTCUSDT",
                "4h",
                sample_df(),
            )

        self.assertIsNone(result)
        self.assertFalse(self.path.exists())

    def test_first_call_writes_header_and_row(self):
        result = record_volume_profile_telemetry("BTCUSDT", "4h", sample_df())

        self.assertIsNotNone(result)
        self.assertTrue(result["available"])
        rows = self._read_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["symbol"], "BTCUSDT")
        self.assertEqual(rows[0]["timeframe"], "4h")

    def test_second_call_within_interval_is_rate_limited(self):
        record_volume_profile_telemetry("BTCUSDT", "4h", sample_df())
        result = record_volume_profile_telemetry("BTCUSDT", "4h", sample_df())

        self.assertIsNone(result)
        self.assertEqual(len(self._read_rows()), 1)

    def test_different_symbol_is_not_rate_limited_by_another_symbols_timer(self):
        record_volume_profile_telemetry("BTCUSDT", "4h", sample_df())
        result = record_volume_profile_telemetry("ETHUSDT", "4h", sample_df())

        self.assertIsNotNone(result)
        self.assertEqual(len(self._read_rows()), 2)

    def test_compute_failure_is_isolated_and_returns_none(self):
        with patch.object(
            volume_profile,
            "compute_volume_profile",
            side_effect=RuntimeError("boom"),
        ):
            result = record_volume_profile_telemetry(
                "BTCUSDT",
                "4h",
                sample_df(),
            )

        self.assertIsNone(result)
        self.assertFalse(self.path.exists())


if __name__ == "__main__":
    unittest.main()
