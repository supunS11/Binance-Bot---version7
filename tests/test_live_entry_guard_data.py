import unittest
from unittest.mock import call, patch

import pandas as pd

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def rising_klines(rows=80):
    data = []

    for index in range(rows):
        close = 100 + (index * 0.01)
        data.append({
            "time": index,
            "open": close - 0.005,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
        })

    return pd.DataFrame(data)


def falling_klines(rows=80):
    data = []

    for index in range(rows):
        close = 100 - (index * 0.01)
        data.append({
            "time": index,
            "open": close + 0.005,
            "high": close + 1,
            "low": close - 1,
            "close": close,
            "volume": 100,
        })

    return pd.DataFrame(data)


class LiveEntryGuardDataTests(unittest.TestCase):
    def test_live_guard_enriches_both_timeframes_before_validation(self):
        fast_raw = rising_klines()
        slow_raw = rising_klines()
        fast_enriched = fast_raw.assign(ema20=1)
        slow_enriched = slow_raw.assign(ema20=2)

        with patch(
            "main.get_klines",
            side_effect=[fast_raw, slow_raw],
        ), patch(
            "main.apply_indicators",
            side_effect=[fast_enriched, slow_enriched],
        ) as indicators, patch(
            "main.validate_live_entry_guard",
            return_value=(True, {"reason": "TEST_OK"}),
        ) as validate:
            allowed, price, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                100,
                mark_price=101,
            )

        self.assertTrue(allowed)
        self.assertEqual(price, 101)
        self.assertEqual(details["reason"], "TEST_OK")
        self.assertEqual(
            indicators.call_args_list,
            [call(fast_raw), call(slow_raw)],
        )
        self.assertIs(validate.call_args.args[1], fast_enriched)
        self.assertIs(validate.call_args.args[2], slow_enriched)

    def test_enriched_live_guard_uses_ema_and_oscillator_data(self):
        raw = rising_klines()
        mark_price = float(raw.iloc[-1]["close"])

        with patch("main.get_klines", side_effect=[raw, raw.copy()]), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DATA",
            True,
        ), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DIRECTION_SUPPORT",
            True,
        ), patch.object(config, "LIVE_ENTRY_REQUIRE_BOTH_TIMEFRAMES", False):
            allowed, _, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                mark_price,
                mark_price=mark_price,
            )

        self.assertTrue(allowed)
        self.assertIsNotNone(details["fast"]["ema20"])
        self.assertIsNotNone(details["slow"]["ema20"])
        self.assertGreater(details["fast"]["support_score"], 0)
        self.assertGreater(details["slow"]["support_score"], 0)

    def test_enriched_live_guard_blocks_opposing_live_conditions(self):
        raw = falling_klines()
        mark_price = float(raw.iloc[-1]["close"])

        with patch("main.get_klines", side_effect=[raw, raw.copy()]), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DATA",
            True,
        ), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DIRECTION_SUPPORT",
            True,
        ), patch.object(config, "LIVE_ENTRY_REQUIRE_BOTH_TIMEFRAMES", False):
            allowed, _, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                mark_price,
                mark_price=mark_price,
            )

        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "DUAL_LIVE_EMA_WRONG_SIDE")
        self.assertLess(details["fast"]["support_score"], 0)
        self.assertLess(details["slow"]["support_score"], 0)

    def test_missing_live_data_obeys_fail_closed_setting(self):
        with patch("main.get_klines", side_effect=[None, None]), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DATA",
            True,
        ):
            allowed, _, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                100,
                mark_price=100,
            )

        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE")

    def test_too_short_enriched_data_is_treated_as_unavailable(self):
        raw = rising_klines(rows=10)

        with patch("main.get_klines", side_effect=[raw, raw.copy()]), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DATA",
            True,
        ):
            allowed, _, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                100,
                mark_price=100,
            )

        self.assertFalse(allowed)
        self.assertEqual(details["reason"], "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE")

    def test_missing_live_data_can_still_fail_open_when_explicitly_configured(self):
        with patch("main.get_klines", side_effect=[None, None]), patch.object(
            config,
            "LIVE_ENTRY_REQUIRE_DATA",
            False,
        ):
            allowed, _, details = main.check_live_entry_guard(
                "BTCUSDT",
                "BUY",
                100,
                mark_price=100,
            )

        self.assertTrue(allowed)
        self.assertEqual(
            details["reason"],
            "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE_ALLOWED",
        )


if __name__ == "__main__":
    unittest.main()
