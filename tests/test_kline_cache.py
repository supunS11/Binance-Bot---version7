import sys
import time
import unittest
from unittest.mock import patch

import pandas as pd

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": int(time.time() * 1000)},
):
    sys.modules.pop("exchange", None)
    import exchange


def frame(close_time_ms):
    return pd.DataFrame({
        "close_time": [close_time_ms],
        "close": [100.0],
    })


class KlineCacheTests(unittest.TestCase):
    def setUp(self):
        with exchange._kline_cache_lock:
            exchange._kline_cache.clear()

    def tearDown(self):
        with exchange._kline_cache_lock:
            exchange._kline_cache.clear()

    def test_candle_aware_cache_expires_after_candle_close(self):
        now = 1_000.0
        data = frame((now + 3_600) * 1000)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch.object(
            config,
            "KLINE_CACHE_CANDLE_AWARE_ENABLED",
            True,
        ), patch.object(config, "KLINE_CACHE_CLOSE_GRACE_SECONDS", 2):
            expires_at = exchange._kline_cache_expiry(data, now=now)

        self.assertEqual(expires_at, now + 3_602)

    def test_cache_hit_survives_fixed_ttl_until_candle_close(self):
        key = ("BTCUSDT", "1h", 240)
        now = 10_000.0
        data = frame((now + 3_600) * 1000)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch.object(
            config,
            "KLINE_CACHE_CANDLE_AWARE_ENABLED",
            True,
        ), patch.object(config, "KLINE_CACHE_CLOSE_GRACE_SECONDS", 2), patch(
            "exchange.time.time",
            return_value=now,
        ):
            exchange._store_cached_kline_df(key, data)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch(
            "exchange.time.time",
            return_value=now + 600,
        ):
            cached = exchange._get_cached_kline_df(key)

        self.assertIsNotNone(cached)

    def test_cache_refreshes_after_candle_close_and_grace(self):
        key = ("BTCUSDT", "5m", 80)
        now = 20_000.0
        data = frame((now + 300) * 1000)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch.object(
            config,
            "KLINE_CACHE_CANDLE_AWARE_ENABLED",
            True,
        ), patch.object(config, "KLINE_CACHE_CLOSE_GRACE_SECONDS", 2), patch(
            "exchange.time.time",
            return_value=now,
        ):
            exchange._store_cached_kline_df(key, data)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch(
            "exchange.time.time",
            return_value=now + 303,
        ):
            cached = exchange._get_cached_kline_df(key)

        self.assertIsNone(cached)

    def test_fixed_ttl_remains_available_as_fallback(self):
        now = 30_000.0
        data = frame((now + 3_600) * 1000)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch.object(
            config,
            "KLINE_CACHE_CANDLE_AWARE_ENABLED",
            False,
        ):
            expires_at = exchange._kline_cache_expiry(data, now=now)

        self.assertEqual(expires_at, now + 45)

    def test_configured_capacity_holds_all_primary_scan_frames(self):
        now = 40_000.0
        data = frame((now + 3_600) * 1000)

        with patch.object(config, "KLINE_CACHE_SECONDS", 45), patch.object(
            config,
            "KLINE_CACHE_MAX_ITEMS",
            2400,
        ), patch("exchange.time.time", return_value=now):
            for index in range(532 * 3):
                exchange._store_cached_kline_df(
                    (f"SYMBOL{index}", "1h", 240),
                    data,
                )

        self.assertEqual(len(exchange._kline_cache), 532 * 3)
        self.assertIn(("SYMBOL0", "1h", 240), exchange._kline_cache)


if __name__ == "__main__":
    unittest.main()
