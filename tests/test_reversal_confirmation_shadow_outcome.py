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


def _analysis(buy_reversal_confirmed=False, sell_reversal_confirmed=False):
    return {
        "buy": {
            "reversal_confirmed": buy_reversal_confirmed,
            "confirmation_type": "TREND",
            "score": 5.0,
            "reversal_uncapped_score_index": 42,
        },
        "sell": {
            "reversal_confirmed": sell_reversal_confirmed,
            "confirmation_type": None,
            "score": 3.0,
            "reversal_uncapped_score_index": 17,
        },
    }


class ReversalConfirmationShadowOutcomeTests(unittest.TestCase):
    def test_disabled_by_config_skips_everything(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", False
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL, _analysis(buy_reversal_confirmed=True), _entry_df()
            )

        mock_register.assert_not_called()

    def test_neither_side_confirmed_registers_nothing(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL, _analysis(), _entry_df()
            )

        mock_register.assert_not_called()

    def test_buy_side_confirmed_registers_with_forced_reversal_route(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL,
                _analysis(buy_reversal_confirmed=True),
                _entry_df(last_closed_close=101.5),
            )

        mock_register.assert_called_once()
        candidate, price = mock_register.call_args.args
        self.assertEqual(candidate["symbol"], SYMBOL)
        self.assertEqual(candidate["signal"], "BUY")
        self.assertEqual(
            candidate["analysis"]["buy"]["confirmation_type"], "REVERSAL"
        )
        self.assertEqual(price, 101.5)

    def test_both_sides_confirmed_registers_twice(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL,
                _analysis(buy_reversal_confirmed=True, sell_reversal_confirmed=True),
                _entry_df(),
            )

        self.assertEqual(mock_register.call_count, 2)
        signals = {call.args[0]["signal"] for call in mock_register.call_args_list}
        self.assertEqual(signals, {"BUY", "SELL"})

    def test_does_not_mutate_the_original_side_data(self):
        analysis = _analysis(buy_reversal_confirmed=True)

        with patch.object(main, "register_signal_outcome"), patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL, analysis, _entry_df()
            )

        self.assertEqual(analysis["buy"]["confirmation_type"], "TREND")

    def test_missing_entry_df_does_not_raise(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL, _analysis(buy_reversal_confirmed=True), None
            )

        mock_register.assert_not_called()

    def test_registration_exception_is_swallowed(self):
        with patch.object(
            main,
            "register_signal_outcome",
            side_effect=RuntimeError("disk unavailable"),
        ), patch.object(
            config, "REVERSAL_CONFIRMATION_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_reversal_confirmation_shadow_outcomes(
                SYMBOL, _analysis(buy_reversal_confirmed=True), _entry_df()
            )


if __name__ == "__main__":
    unittest.main()
