import unittest
from unittest.mock import Mock, patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def build_candidate(signal="SELL", transition_score=None, transition=None):
    market_context = {}
    if transition is not None:
        market_context["transition"] = transition
    elif transition_score is not None:
        side_key = signal.lower()
        market_context["transition"] = {f"{side_key}_score": transition_score}

    return {
        "symbol": "TESTUSDT",
        "signal": signal,
        "analysis": {},
        "market_context": market_context,
    }


class RegimeTransitionHardVetoTests(unittest.TestCase):
    """Evidence: across ~7,000 recorded outcomes, transition_score near
    zero (quiet regime) wins 58.7% of the time; a strongly OPPOSING score
    wins only 42.9%, and - counterintuitively - a strongly ALIGNED score
    wins only 18.6%. An active regime transition means the market is
    unpredictable right now in either direction, so this vetoes on
    magnitude alone, not just conflicting-direction scores."""

    def test_disabled_by_default_config_never_blocks(self):
        with patch.object(config, "REGIME_TRANSITION_HARD_VETO_ENABLED", False):
            blocked, reason = main.regime_transition_hard_veto(
                build_candidate(transition_score=-5.0)
            )

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_opposing_score_beyond_threshold_is_blocked(self):
        with patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ENABLED", True
        ), patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ABS_SCORE", 0.1
        ):
            blocked, reason = main.regime_transition_hard_veto(
                build_candidate(signal="SELL", transition_score=-0.34)
            )

        self.assertTrue(blocked)
        self.assertIn("ACTIVE_REGIME_TRANSITION", reason)

    def test_aligned_score_beyond_threshold_is_also_blocked(self):
        # The counterintuitive half of the finding: a score that SUPPORTS
        # the trade direction is blocked too, since it performed worse
        # than opposing did in the real data.
        with patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ENABLED", True
        ), patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ABS_SCORE", 0.1
        ):
            blocked, reason = main.regime_transition_hard_veto(
                build_candidate(signal="SELL", transition_score=0.5)
            )

        self.assertTrue(blocked)

    def test_near_zero_score_within_threshold_is_not_blocked(self):
        with patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ENABLED", True
        ), patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ABS_SCORE", 0.1
        ):
            blocked, reason = main.regime_transition_hard_veto(
                build_candidate(signal="SELL", transition_score=0.05)
            )

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_missing_transition_data_does_not_block(self):
        with patch.object(config, "REGIME_TRANSITION_HARD_VETO_ENABLED", True):
            blocked, reason = main.regime_transition_hard_veto(
                build_candidate(transition={})
            )

        self.assertFalse(blocked)

    def test_zero_threshold_disables_veto_even_when_enabled(self):
        with patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ENABLED", True
        ), patch.object(
            config, "REGIME_TRANSITION_HARD_VETO_ABS_SCORE", 0
        ):
            blocked, _ = main.regime_transition_hard_veto(
                build_candidate(transition_score=-99)
            )

        self.assertFalse(blocked)


class ExecuteEntryCandidateRegimeTransitionWiringTests(unittest.TestCase):
    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_transition_veto_blocks_entry_and_writes_journal(self):
        state = {"positions": {}, "pending_executions": {}}
        candidate = {
            "symbol": "TESTUSDT",
            "signal": "SELL",
            "analysis": {},
            "participation": None,
            "trend_df": None,
            "confirm_df": None,
            "entry_df": None,
            "btc_trend": None,
            "btc_corr": None,
            "rs": None,
            "news_context": None,
            "llm_context": None,
            "market_context": {"transition": {"sell_score": 0.5}},
            "rank_score": 12.5,
        }

        with patch(
            "main.market_flow_hard_veto",
            return_value=(False, ""),
        ), patch(
            "main.regime_transition_hard_veto",
            return_value=(True, "ACTIVE_REGIME_TRANSITION SCORE=0.5 ABS_LIMIT=0.1"),
        ), patch(
            "main.append_signal_journal"
        ) as journal, patch("main.log_warning"):
            result = main.execute_entry_candidate(
                candidate,
                state,
                {},
                {},
                None,
                Mock(),
            )

        self.assertEqual(result, ({}, {}, False))
        journal.assert_called_once()
        _, kwargs = journal.call_args
        self.assertEqual(kwargs["action"], "SKIPPED_REGIME_TRANSITION")
        self.assertIn("ACTIVE_REGIME_TRANSITION", kwargs["skip_reason"])


if __name__ == "__main__":
    unittest.main()
