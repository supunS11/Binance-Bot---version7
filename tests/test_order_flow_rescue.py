import unittest
from unittest.mock import Mock, patch

import config
from strategy import _module_gates_check

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


class ModuleGatesCheckRescueTests(unittest.TestCase):
    """Evidence (2026-08): confirm_score/entry_score are the two biggest
    NO_SIGNAL bottlenecks, but the candidates failing them fail by a wide
    margin (median ~3+ points below the bar), not a near miss. Rather than
    lowering the thresholds for everyone, a candidate missing ONLY one of
    these two gates by a small configured margin is let through
    provisionally, tagged for the separate order-flow confirmation check
    in main.py - not a free pass."""

    def _check(self, trend=8, confirm=8, entry=4.5, quality=1, regime=0):
        return _module_gates_check(trend, confirm, entry, quality, regime)

    def test_rescue_disabled_by_default_hard_blocks_near_miss(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", False), patch.object(
            config, "SIGNAL_MIN_CONFIRM_SCORE", 8.0
        ):
            ok, reasons, pending = self._check(confirm=7.6)

        self.assertFalse(ok)
        self.assertFalse(pending)
        self.assertTrue(any("CONFIRM" in r for r in reasons))

    def test_near_miss_confirm_only_is_marked_pending_when_rescue_enabled(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_CONFIRM_SCORE", 8.0
        ), patch.object(config, "CONFIRM_SCORE_RESCUE_MARGIN", 0.5):
            ok, reasons, pending = self._check(confirm=7.6)

        self.assertTrue(ok)
        self.assertTrue(pending)
        self.assertTrue(any("CONFIRM" in r for r in reasons))

    def test_near_miss_entry_only_is_marked_pending_when_rescue_enabled(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_ENTRY_SCORE", 4.5
        ), patch.object(config, "ENTRY_SCORE_RESCUE_MARGIN", 0.5):
            ok, reasons, pending = self._check(entry=4.1)

        self.assertTrue(ok)
        self.assertTrue(pending)
        self.assertTrue(any("ENTRY" in r for r in reasons))

    def test_wide_margin_failure_stays_hard_blocked_even_with_rescue_enabled(self):
        # This is the real-world case: confirm_score failing by 3+ points,
        # not a near miss. Rescue must not paper over a wide gap.
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_CONFIRM_SCORE", 8.0
        ), patch.object(config, "CONFIRM_SCORE_RESCUE_MARGIN", 0.5):
            ok, reasons, pending = self._check(confirm=4.75)

        self.assertFalse(ok)
        self.assertFalse(pending)

    def test_trend_score_failure_never_gets_rescued(self):
        # Only CONFIRM and ENTRY are rescue-eligible; TREND/QUALITY/REGIME
        # showed no comparable near-miss evidence and stay hard gates.
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_TREND_SCORE", 7.5
        ):
            ok, reasons, pending = self._check(trend=7.4)

        self.assertFalse(ok)
        self.assertFalse(pending)

    def test_failing_both_confirm_and_entry_within_margin_still_pending(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_CONFIRM_SCORE", 8.0
        ), patch.object(
            config, "SIGNAL_MIN_ENTRY_SCORE", 4.5
        ), patch.object(
            config, "CONFIRM_SCORE_RESCUE_MARGIN", 0.5
        ), patch.object(
            config, "ENTRY_SCORE_RESCUE_MARGIN", 0.5
        ):
            ok, reasons, pending = self._check(confirm=7.6, entry=4.1)

        self.assertTrue(ok)
        self.assertTrue(pending)

    def test_failing_confirm_within_margin_but_quality_hard_fails_blocks(self):
        # A near-miss on the rescuable gate doesn't save a candidate that
        # also hard-fails a non-rescuable gate.
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "SIGNAL_MIN_CONFIRM_SCORE", 8.0
        ), patch.object(
            config, "SIGNAL_MIN_QUALITY_SCORE", 0.25
        ), patch.object(config, "CONFIRM_SCORE_RESCUE_MARGIN", 0.5):
            ok, reasons, pending = self._check(confirm=7.6, quality=-1)

        self.assertFalse(ok)
        self.assertFalse(pending)

    def test_all_gates_pass_cleanly_no_pending_flag(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True):
            ok, reasons, pending = self._check()

        self.assertTrue(ok)
        self.assertFalse(pending)
        self.assertEqual(reasons, [])


class OrderFlowRescueVetoTests(unittest.TestCase):
    def _candidate(self, signal="SELL", rescue_pending=True, flow=None):
        side_data = {"order_flow_rescue_pending": rescue_pending}
        return {
            "symbol": "TESTUSDT",
            "signal": signal,
            "analysis": {signal.lower(): side_data},
            "market_context": {"flow": flow} if flow is not None else {},
        }

    def test_disabled_by_config_never_blocks(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", False):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(rescue_pending=True, flow={"available": True, "sell_score": -5})
            )

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_non_pending_candidate_is_never_touched(self):
        # The overwhelming majority of candidates never set the pending
        # flag at all - this veto must be a complete no-op for them.
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(rescue_pending=False, flow={"available": True, "sell_score": -5})
            )

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_pending_with_strong_confirming_flow_is_allowed(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "ORDER_FLOW_RESCUE_MIN_FLOW_SCORE", 1.5
        ):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(
                    signal="SELL",
                    rescue_pending=True,
                    flow={"available": True, "sell_score": 2.1},
                )
            )

        self.assertFalse(blocked)
        self.assertEqual(reason, "")

    def test_pending_with_weak_flow_is_blocked(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True), patch.object(
            config, "ORDER_FLOW_RESCUE_MIN_FLOW_SCORE", 1.5
        ):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(
                    signal="SELL",
                    rescue_pending=True,
                    flow={"available": True, "sell_score": 0.3},
                )
            )

        self.assertTrue(blocked)
        self.assertIn("ORDER_FLOW_RESCUE_NOT_CONFIRMED", reason)

    def test_pending_with_opposing_flow_is_blocked(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(
                    signal="SELL",
                    rescue_pending=True,
                    flow={"available": True, "sell_score": -1.2},
                )
            )

        self.assertTrue(blocked)

    def test_pending_with_unavailable_flow_data_is_blocked_fail_closed(self):
        with patch.object(config, "ORDER_FLOW_RESCUE_ENABLED", True):
            blocked, reason = main.order_flow_rescue_veto(
                self._candidate(
                    signal="SELL", rescue_pending=True, flow={"available": False}
                )
            )

        self.assertTrue(blocked)
        self.assertEqual(reason, "ORDER_FLOW_RESCUE_DATA_UNAVAILABLE")


class ExecuteEntryCandidateOrderFlowRescueWiringTests(unittest.TestCase):
    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_rescue_veto_blocks_entry_and_writes_journal(self):
        state = {"positions": {}, "pending_executions": {}}
        candidate = {
            "symbol": "TESTUSDT",
            "signal": "SELL",
            "analysis": {"sell": {"order_flow_rescue_pending": True}},
            "participation": None,
            "trend_df": None,
            "confirm_df": None,
            "entry_df": None,
            "btc_trend": None,
            "btc_corr": None,
            "rs": None,
            "news_context": None,
            "llm_context": None,
            "market_context": {"flow": {"available": True, "sell_score": 0.1}},
            "rank_score": 10,
        }

        with patch(
            "main.market_flow_hard_veto", return_value=(False, "")
        ), patch(
            "main.regime_transition_hard_veto", return_value=(False, "")
        ), patch(
            "main.order_flow_rescue_veto",
            return_value=(True, "ORDER_FLOW_RESCUE_NOT_CONFIRMED SCORE=0.1 < 1.5"),
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
        self.assertEqual(kwargs["action"], "SKIPPED_ORDER_FLOW_RESCUE")
        self.assertIn("ORDER_FLOW_RESCUE_NOT_CONFIRMED", kwargs["skip_reason"])


if __name__ == "__main__":
    unittest.main()
