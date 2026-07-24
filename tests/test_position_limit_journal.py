import unittest
from unittest.mock import Mock, patch

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


SYMBOL = "TESTUSDT"


def build_candidate():
    return {
        "symbol": SYMBOL,
        "signal": "BUY",
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
        "market_context": {},
        "rank_score": 12.5,
    }


class PositionLimitJournalTests(unittest.TestCase):
    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_position_limit_skip_is_written_to_journal(self):
        state = {"positions": {}, "pending_executions": {}}
        candidate = build_candidate()

        with patch(
            "main.market_flow_hard_veto",
            return_value=(False, ""),
        ), patch(
            "main.get_open_position_details",
            return_value={},
        ), patch(
            "main.get_open_position_amounts",
            return_value={},
        ), patch(
            "main.prune_and_cleanup_closed_positions",
        ), patch(
            "main.get_open_position_counts",
            return_value={"total": 0, "buy": 0, "sell": 0},
        ), patch(
            "main.get_position_pool_counts",
            return_value={},
        ), patch(
            "main.get_tp1_runner_pool_counts",
            return_value={},
        ), patch(
            "main.check_entry_position_limits",
            return_value=(
                False,
                "TREND MAX POSITIONS REACHED | TOTAL=5/5 | BUY=3 | SELL=2",
            ),
        ), patch(
            "main.append_signal_journal",
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
        self.assertEqual(kwargs["action"], "SKIPPED_POSITION_LIMIT")
        self.assertIn("MAX POSITIONS REACHED", kwargs["skip_reason"])
        self.assertEqual(kwargs["rank_score"], 12.5)


if __name__ == "__main__":
    unittest.main()
