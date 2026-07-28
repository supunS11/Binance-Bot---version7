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


class ExecuteEntryCandidateSnapshotCacheTests(unittest.TestCase):
    """execute_entry_candidate is called once per ranked candidate every
    scan cycle (often a dozen+ times back to back). Forcing a fresh,
    cache-bypassing position lookup on every single call was flooding
    Binance with redundant real REST calls and tripping a real rate-limit
    rejection, which then dropped every candidate ranked afterward for the
    remainder of the reactive backoff window - real, already-qualified
    trades were being lost to this, not filtered by strategy logic."""

    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_live_position_snapshot_check_uses_the_existing_cache(self):
        state = {"positions": {}, "pending_executions": {}}
        candidate = build_candidate()

        with patch(
            "main.market_flow_hard_veto",
            return_value=(False, ""),
        ), patch(
            "main.get_open_position_details",
            return_value=None,
        ) as get_details, patch("main.append_signal_journal"), patch(
            "main.log_warning"
        ):
            main.execute_entry_candidate(
                candidate,
                state,
                {},
                {},
                None,
                Mock(),
            )

        get_details.assert_called_once_with(force=False)


if __name__ == "__main__":
    unittest.main()
