import unittest
from unittest.mock import patch

import pandas as pd

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


SYMBOL = "TESTUSDT"


def _entry_df(last_closed_close=101.5):
    return pd.DataFrame(
        [
            {"close": 100.0, "high": 100.5, "low": 99.5},
            {"close": last_closed_close, "high": 102.0, "low": 101.0},
            {"close": 103.0, "high": 103.5, "low": 102.5},
        ]
    )


class _FakeMonitor:
    def __init__(self, snapshot):
        self._snapshot = snapshot

    def snapshot(self, symbol):
        return self._snapshot


_UNSET = object()


class LiquidationShadowOutcomeTests(unittest.TestCase):
    def _run(self, snapshot, entry_df=_UNSET):
        monitor = _FakeMonitor(snapshot) if snapshot is not None else None
        resolved_entry_df = _entry_df() if entry_df is _UNSET else entry_df
        with patch.object(main, "register_signal_outcome") as mock_register:
            main._register_liquidation_shadow_outcomes(
                SYMBOL,
                resolved_entry_df,
                monitor,
            )
        return mock_register

    def test_disabled_by_config_skips_everything(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", False
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 10,
                    "net_liquidation_notional": 500000,
                }
            )
        mock_register.assert_not_called()

    def test_monitor_none_skips_everything(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ):
            mock_register = self._run(snapshot=None)
        mock_register.assert_not_called()

    def test_unavailable_snapshot_skips(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ):
            mock_register = self._run({"available": False})
        mock_register.assert_not_called()

    def test_stale_event_skips_registration(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 900,  # older than the cutoff
                    "net_liquidation_notional": 500000,
                }
            )
        mock_register.assert_not_called()

    def test_missing_last_event_age_skips(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": None,
                    "net_liquidation_notional": 500000,
                }
            )
        mock_register.assert_not_called()

    def test_below_notable_threshold_skips_registration(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ), patch.object(
            config, "LIQUIDATION_SHADOW_NOTABLE_NOTIONAL_USDT", 200000
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 30,
                    "net_liquidation_notional": 50000,  # below threshold
                }
            )
        mock_register.assert_not_called()

    def test_notable_long_flush_registers_both_sides_with_correct_direction(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ), patch.object(
            config, "LIQUIDATION_SHADOW_NOTABLE_NOTIONAL_USDT", 200000
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 30,
                    "net_liquidation_notional": 500000,  # positive = long flush
                },
                entry_df=_entry_df(last_closed_close=101.5),
            )

        self.assertEqual(mock_register.call_count, 2)
        sides = set()
        for call in mock_register.call_args_list:
            candidate, price = call.args
            self.assertEqual(candidate["symbol"], SYMBOL)
            self.assertEqual(price, 101.5)
            side = candidate["signal"]
            sides.add(side)
            self.assertEqual(
                candidate["analysis"][side.lower()]["confirmation_type"],
                "LIQUIDATION_SHADOW_LONG_FLUSH",
            )
        self.assertEqual(sides, {"BUY", "SELL"})

    def test_notable_short_squeeze_uses_opposite_direction_label(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ), patch.object(
            config, "LIQUIDATION_SHADOW_NOTABLE_NOTIONAL_USDT", 200000
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 30,
                    "net_liquidation_notional": -500000,  # negative = short squeeze
                }
            )

        for call in mock_register.call_args_list:
            candidate, _ = call.args
            side = candidate["signal"]
            self.assertEqual(
                candidate["analysis"][side.lower()]["confirmation_type"],
                "LIQUIDATION_SHADOW_SHORT_SQUEEZE",
            )

    def test_missing_entry_df_skips(self):
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ), patch.object(
            config, "LIQUIDATION_SHADOW_NOTABLE_NOTIONAL_USDT", 200000
        ):
            mock_register = self._run(
                {
                    "available": True,
                    "last_event_age_seconds": 30,
                    "net_liquidation_notional": 500000,
                },
                entry_df=None,
            )
        mock_register.assert_not_called()

    def test_registration_exception_is_swallowed(self):
        monitor = _FakeMonitor(
            {
                "available": True,
                "last_event_age_seconds": 30,
                "net_liquidation_notional": 500000,
            }
        )
        with patch.object(
            config, "LIQUIDATION_SHADOW_OUTCOME_TRACKING_ENABLED", True
        ), patch.object(
            config, "LIQUIDATION_SHADOW_RECENT_EVENT_SECONDS", 360
        ), patch.object(
            config, "LIQUIDATION_SHADOW_NOTABLE_NOTIONAL_USDT", 200000
        ), patch.object(
            main,
            "register_signal_outcome",
            side_effect=RuntimeError("disk unavailable"),
        ):
            main._register_liquidation_shadow_outcomes(
                SYMBOL, _entry_df(), monitor
            )


class FinalizeScannedSymbolLiquidationWiringTests(unittest.TestCase):
    def test_finalize_scanned_symbol_forwards_monitor_to_liquidation_registration(
        self,
    ):
        entry_df = _entry_df()
        scan_item = {
            "symbol": SYMBOL,
            "analysis": {
                "signal": None,
                "buy": {},
                "sell": {},
            },
            "participation": None,
            "trend_df": entry_df,
            "confirm_df": entry_df,
            "entry_df": entry_df,
            "btc_trend": "NEUTRAL",
            "btc_corr": 0,
            "rs": 0,
        }
        sentinel_monitor = object()

        with patch.object(
            main, "record_volume_profile_telemetry"
        ), patch.object(
            main, "_register_reversal_confirmation_shadow_outcomes"
        ), patch.object(
            main, "_register_liquidation_shadow_outcomes"
        ) as mock_liquidation, patch.object(
            main, "append_signal_journal"
        ), patch.object(
            main, "log_signal_analysis"
        ):
            main.finalize_scanned_symbol(
                scan_item,
                [],
                {},
                {},
                {},
                object(),
                None,
                sentinel_monitor,
            )

        mock_liquidation.assert_called_once_with(
            SYMBOL, entry_df, sentinel_monitor
        )


if __name__ == "__main__":
    unittest.main()
