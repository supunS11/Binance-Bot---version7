import unittest
from unittest.mock import patch

import pandas as pd

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def _entry_df(last_closed_close=101.5):
    return pd.DataFrame(
        [
            {"close": 100.0, "high": 100.5, "low": 99.5},
            {"close": last_closed_close, "high": 102.0, "low": 101.0},
            {"close": 103.0, "high": 103.5, "low": 102.5},  # still-forming live candle
        ]
    )


def _candidate(symbol="BTCUSDT"):
    return {
        "symbol": symbol,
        "signal": "BUY",
        "rank_score": 10,
        "analysis": {"buy": {"score": 42, "confirmation_type": "TREND"}},
        "entry_df": _entry_df(),
    }


class ShadowSignalOutcomeRegistrationTests(unittest.TestCase):
    def test_registers_shadow_outcome_using_last_closed_candle_when_not_executed(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "SIGNAL_OUTCOME_SHADOW_TRACKING_ENABLED", True
        ):
            candidate = _candidate()
            main._register_shadow_signal_outcome(candidate)

        mock_register.assert_called_once_with(candidate, 101.5)

    def test_skips_registration_when_disabled_via_config(self):
        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "SIGNAL_OUTCOME_SHADOW_TRACKING_ENABLED", False
        ):
            main._register_shadow_signal_outcome(_candidate())

        mock_register.assert_not_called()

    def test_swallows_missing_entry_df_without_raising(self):
        candidate = _candidate()
        candidate["entry_df"] = None

        with patch.object(
            main, "register_signal_outcome"
        ) as mock_register, patch.object(
            config, "SIGNAL_OUTCOME_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_shadow_signal_outcome(candidate)

        mock_register.assert_not_called()

    def test_swallows_registration_exceptions_without_raising(self):
        with patch.object(
            main,
            "register_signal_outcome",
            side_effect=RuntimeError("disk unavailable"),
        ), patch.object(
            config, "SIGNAL_OUTCOME_SHADOW_TRACKING_ENABLED", True
        ):
            main._register_shadow_signal_outcome(_candidate())


class ProcessRankedEntryCandidatesShadowWiringTests(unittest.TestCase):
    def test_registers_shadow_outcome_only_for_non_executed_candidates(self):
        candidates = [
            _candidate("BTCUSDT"),
            _candidate("ETHUSDT"),
        ]
        executed_map = {"BTCUSDT": True, "ETHUSDT": False}

        def fake_execute(candidate, trade_state, position_details, open_positions, *rest):
            return position_details, open_positions, executed_map[candidate["symbol"]]

        with patch.object(
            main, "prepare_news_scan_context"
        ), patch.object(
            main, "prefetch_llm_candidate_reviews"
        ), patch.object(
            main, "execute_entry_candidate", side_effect=fake_execute
        ), patch.object(
            main, "_register_shadow_signal_outcome"
        ) as mock_shadow, patch.object(
            config, "SIGNAL_RANKING_MAX_CANDIDATES", 0
        ):
            main.process_ranked_entry_candidates(
                candidates, {}, {}, {}, object(), object()
            )

        registered_symbols = [
            call.args[0]["symbol"] for call in mock_shadow.call_args_list
        ]
        self.assertEqual(registered_symbols, ["ETHUSDT"])


if __name__ == "__main__":
    unittest.main()
