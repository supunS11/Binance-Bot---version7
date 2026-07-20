import unittest
import time
from datetime import datetime
from unittest.mock import patch

import config
import main


SYMBOL = "BTCUSDT"


def position_state(**updates):
    item = {
        "symbol": SYMBOL,
        "managed_by_bot": True,
        "campaign_risk_version": 2,
        "side": "BUY",
        "confirmation_type": "TREND",
        "avg_entry": 100.0,
        "opened_at": "2026-01-01T00:00:00",
    }
    item.update(updates)
    return {"positions": {SYMBOL: item}}


def live_position():
    return {
        SYMBOL: {
            "amount": 1.0,
            "position_side": "LONG",
            "mark_price": 95.0,
            "entry_price": 100.0,
        }
    }


class CoordinatedExitStateMachineTests(unittest.TestCase):
    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_tp1_touch_blocks_recovery_but_does_not_claim_runner_exits(self):
        item = position_state(
            multi_tp_stage=main.TP1_PENDING,
            tp1_trigger_seen_at="2026-01-01T00:00:00",
        )["positions"][SYMBOL]

        self.assertFalse(main.runner_owns_position(item))
        self.assertTrue(main.tp1_transition_blocks_recovery(item))
        self.assertTrue(main.position_exit_blocks_dca(item))

        item["multi_tp_stage"] = main.RUNNER_PENDING
        self.assertTrue(main.runner_owns_position(item))

    def test_stale_tp_reprice_repair_hands_off_to_fresh_runner_state(self):
        stale_state = position_state(
            tp_reprice_status="PENDING",
            multi_tp_stage=main.TP1_PENDING,
        )
        fresh_state = position_state(
            tp_reprice_status="PENDING",
            multi_tp_stage=main.RUNNER_PENDING,
        )

        with patch(
            "main.load_trade_state",
            return_value=fresh_state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=self.persister([]),
        ), patch("main.get_open_position_details") as live_details, patch(
            "main.cancel_open_take_profit_orders"
        ) as cancel_take_profit:
            handled = main.repair_pending_dca_tp_reprice(
                SYMBOL,
                live_position()[SYMBOL],
                stale_state,
                None,
            )

        self.assertTrue(handled)
        self.assertEqual(
            fresh_state["positions"][SYMBOL]["tp_reprice_status"],
            "COMPLETE_RUNNER_OWNERSHIP",
        )
        self.assertEqual(
            stale_state["positions"][SYMBOL]["multi_tp_stage"],
            main.RUNNER_PENDING,
        )
        live_details.assert_not_called()
        cancel_take_profit.assert_not_called()

    @staticmethod
    def persister(events):
        def persist(state, symbol, updates):
            status = next(
                (
                    value
                    for key, value in updates.items()
                    if key.endswith("_exit_status")
                ),
                None,
            )
            if status:
                events.append(f"persist:{status}")
            state["positions"][symbol].update(updates)
            return True

        return persist

    def test_early_exit_persists_pending_before_close_and_retries_failed(self):
        state = position_state(
            early_invalidation_exit_status="FAILED",
            early_invalidation_exit_route="TREND",
            early_invalidation_exit_reason="TREND_THESIS_INVALIDATED",
            early_invalidation_exit_evidence={"fast_failure": True},
        )
        events = []

        def close(*args, **kwargs):
            events.append("close")
            return {"orderId": 1}

        with patch.object(config, "EARLY_FLOW_EXIT_ENABLED", False), patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=self.persister(events),
        ), patch(
            "main.get_open_position_details",
            return_value=live_position(),
        ), patch(
            "main.close_position_market",
            side_effect=close,
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ), patch("main.send_telegram_message"):
            handled = main.DcaWebsocketMonitor()._handle_route_early_invalidation(
                SYMBOL,
                95.0,
                state,
            )

        self.assertTrue(handled)
        self.assertLess(events.index("persist:PENDING"), events.index("close"))
        self.assertEqual(events[-1], "persist:SUBMITTED")

    def test_early_exit_does_not_close_when_intent_persistence_fails(self):
        state = position_state(
            early_invalidation_exit_status="FAILED",
            early_invalidation_exit_route="TREND",
        )

        with patch.object(config, "EARLY_FLOW_EXIT_ENABLED", False), patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.update_position_runtime_fields",
            return_value=False,
        ), patch("main.get_open_position_details") as details, patch(
            "main.close_position_market"
        ) as close:
            handled = main.DcaWebsocketMonitor()._handle_route_early_invalidation(
                SYMBOL,
                95.0,
                state,
            )

        self.assertTrue(handled)
        details.assert_not_called()
        close.assert_not_called()
        self.assertTrue(main.shutdown_event.is_set())

    def test_profit_failed_state_retries_even_after_feature_is_disabled(self):
        state = position_state(
            trend_profit_exit_status="FAILED",
            trend_peak_roi=20.0,
            trend_profit_basis_entry=100.0,
            trend_profit_floor_roi=11.0,
            trend_profit_exit_reason="TREND_PROFIT_RETRACE",
        )
        events = []

        def close(*args, **kwargs):
            events.append("close")
            return {"orderId": 2}

        with patch.object(
            config,
            "TREND_PROFIT_PROTECTION_ENABLED",
            False,
        ), patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=self.persister(events),
        ), patch(
            "main.get_open_position_details",
            return_value=live_position(),
        ), patch(
            "main.close_position_market",
            side_effect=close,
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ), patch("main.send_telegram_message"):
            handled = main.DcaWebsocketMonitor()._handle_trend_profit_protection(
                SYMBOL,
                101.0,
                state,
            )

        self.assertTrue(handled)
        self.assertLess(events.index("persist:PENDING"), events.index("close"))
        self.assertEqual(events[-1], "persist:SUBMITTED")

    def test_profit_exit_revalidates_after_dca_changes_average_entry(self):
        stale_state = position_state(
            trend_profit_exit_status="",
            trend_peak_roi=20.0,
            trend_profit_basis_entry=100.0,
            trend_profit_armed=True,
        )
        fresh_state = position_state(
            avg_entry=90.0,
            dca_count=1,
            trend_profit_exit_status="",
            trend_peak_roi=0.0,
            trend_profit_basis_entry=90.0,
            trend_profit_armed=False,
        )
        updates = []

        def persist(state, symbol, fields):
            updates.append(dict(fields))
            state["positions"][symbol].update(fields)
            return True

        with patch.object(
            config,
            "TREND_PROFIT_PROTECTION_ENABLED",
            True,
        ), patch(
            "main.load_trade_state",
            return_value=fresh_state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=persist,
        ), patch("main.get_open_position_details") as details, patch(
            "main.close_position_market"
        ) as close:
            handled = main.DcaWebsocketMonitor()._handle_trend_profit_protection(
                SYMBOL,
                101.0,
                stale_state,
            )

        self.assertFalse(handled)
        details.assert_not_called()
        close.assert_not_called()
        self.assertFalse(
            any(
                item.get("trend_profit_exit_status") == "PENDING"
                for item in updates
            )
        )

    def test_fresh_committed_profit_attempt_honors_durable_retry_cooldown(self):
        stale_state = position_state(
            trend_profit_exit_status="",
            trend_peak_roi=20.0,
            trend_profit_basis_entry=100.0,
            trend_profit_armed=True,
        )
        fresh_state = position_state(
            trend_profit_exit_status="FAILED",
            position_exit_owner="TREND_PROFIT",
            trend_peak_roi=20.0,
            trend_profit_basis_entry=100.0,
            trend_profit_exit_last_attempt_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )

        with patch.object(
            config,
            "TREND_PROFIT_PROTECTION_ENABLED",
            True,
        ), patch.object(
            config,
            "PROFIT_EXIT_PENDING_RETRY_SECONDS",
            60,
        ), patch(
            "main.load_trade_state",
            return_value=fresh_state,
        ), patch("main.update_position_runtime_fields") as persist, patch(
            "main.get_open_position_details"
        ) as details, patch("main.close_position_market") as close:
            handled = main.DcaWebsocketMonitor()._handle_trend_profit_protection(
                SYMBOL,
                101.0,
                stale_state,
            )

        self.assertTrue(handled)
        persist.assert_not_called()
        details.assert_not_called()
        close.assert_not_called()

    def test_time_exit_unavailable_snapshot_is_uncertain_not_flat(self):
        state = position_state(
            time_exit_status="FAILED",
            time_exit_reason="TIME_EXIT_WEAKNESS_CONFIRMED",
            time_exit_evidence=["4h_structure"],
            time_exit_weakness_score=2,
        )
        events = []

        with patch.object(config, "TIME_EXIT_ENABLED", False), patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=self.persister(events),
        ), patch(
            "main.get_open_position_details",
            return_value=None,
        ), patch("main.close_position_market") as close, patch(
            "main.cancel_open_protection_orders"
        ) as cancel:
            handled = main.DcaWebsocketMonitor()._handle_time_exit(
                SYMBOL,
                95.0,
                state,
            )

        self.assertTrue(handled)
        self.assertEqual(events, ["persist:PENDING", "persist:UNCERTAIN"])
        close.assert_not_called()
        cancel.assert_not_called()

    def test_confirmed_time_close_with_unverified_cleanup_stays_uncertain(self):
        state = position_state(
            time_exit_status="FAILED",
            time_exit_reason="TIME_EXIT_WEAKNESS_CONFIRMED",
        )
        events = []

        with patch.object(config, "TIME_EXIT_ENABLED", False), patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=self.persister(events),
        ), patch(
            "main.get_open_position_details",
            return_value=live_position(),
        ), patch(
            "main.close_position_market",
            return_value={"orderId": 3},
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=False,
        ), patch("main.send_telegram_message") as telegram:
            handled = main.DcaWebsocketMonitor()._handle_time_exit(
                SYMBOL,
                95.0,
                state,
            )

        self.assertTrue(handled)
        self.assertEqual(events[-1], "persist:UNCERTAIN")
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)
        self.assertTrue(main.shutdown_event.is_set())
        telegram.assert_not_called()


class CrashWindowSafetyTests(unittest.TestCase):
    def setUp(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def tearDown(self):
        main.shutdown_event.clear()
        main.entry_quarantined_symbols.clear()

    def test_untracked_marker_and_close_failure_remains_retryable(self):
        state = {"positions": {}, "pending_executions": {}}

        with patch.object(config, "SYMBOLS", [SYMBOL]), patch.object(
            config,
            "UNTRACKED_POSITION_FAIL_CLOSE_ENABLED",
            True,
        ), patch(
            "main.upsert_position_state",
            return_value=False,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=False,
        ) as close, patch("main.log_error"):
            attempted, unresolved = main.reconcile_untracked_open_positions(
                live_position(),
                state,
            )

        self.assertTrue(attempted)
        self.assertEqual(unresolved, {SYMBOL})
        close.assert_called_once()
        self.assertFalse(main.shutdown_event.is_set())
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_untracked_position_outside_v7_scope_is_never_closed(self):
        state = {"positions": {}, "pending_executions": {}}

        with patch.object(config, "SYMBOLS", ["ETHUSDT"]), patch.object(
            config,
            "MAX_SCAN_SYMBOLS",
            0,
        ), patch.object(
            config,
            "UNTRACKED_POSITION_FAIL_CLOSE_ENABLED",
            True,
        ), patch("main.upsert_position_state") as persist, patch(
            "main.fail_safe_close_unprotected_position",
        ) as close:
            attempted, unresolved = main.reconcile_untracked_open_positions(
                live_position(),
                state,
            )

        self.assertFalse(attempted)
        self.assertFalse(unresolved)
        persist.assert_not_called()
        close.assert_not_called()

    def test_entry_order_marker_is_persisted_before_external_submission(self):
        state = {"positions": {}, "pending_executions": {}}
        events = []

        def persist_marker(*args, **kwargs):
            events.append("marker")
            state["positions"][SYMBOL] = {
                "pending_submission": {
                    "context": "ENTRY",
                    "submission_phase": "READY_TO_SUBMIT",
                },
            }
            return True

        def submit_order(*args, **kwargs):
            events.append("order")
            return None

        with patch(
            "main.persist_entry_submission_marker",
            side_effect=persist_marker,
        ), patch(
            "main.place_market_order",
            side_effect=submit_order,
        ), patch("main.log_error"):
            order = main.submit_entry_order_with_marker(
                state,
                SYMBOL,
                "BUY",
                0.5,
                100.0,
                92.0,
                "TREND",
            )

        self.assertIsNone(order)
        self.assertEqual(events, ["marker", "order"])
        self.assertTrue(main.state_requires_urgent_safety_retry(state))
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_entry_marker_failure_prevents_order_submission(self):
        state = {"positions": {}, "pending_executions": {}}

        with patch(
            "main.persist_entry_submission_marker",
            return_value=False,
        ), patch("main.place_market_order") as place_order, patch(
            "main.log_error",
        ):
            order = main.submit_entry_order_with_marker(
                state,
                SYMBOL,
                "BUY",
                0.5,
                100.0,
                92.0,
                "TREND",
            )

        self.assertIsNone(order)
        place_order.assert_not_called()
        self.assertTrue(main.shutdown_event.is_set())

    def test_failed_entry_close_retains_original_marker_without_shutdown(self):
        state = position_state(
            pending_submission={
                "context": "ENTRY",
                "submission_phase": "READY_TO_SUBMIT",
            },
        )

        with patch(
            "main.persist_pending_execution",
            return_value=False,
        ), patch("main.log_error"):
            persisted = main.retain_entry_close_retry(
                state,
                SYMBOL,
                {"clientOrderId": "v7-entry-test"},
                "BUY",
                0.5,
                100.0,
                "TREND",
                92.0,
                "ENTRY_PROTECTION_FAILURE",
            )

        self.assertFalse(persisted)
        self.assertFalse(main.shutdown_event.is_set())
        self.assertTrue(main.state_requires_urgent_safety_retry(state))
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_interrupted_entry_marker_with_live_position_is_fail_closed(self):
        state = position_state(
            pending_submission={
                "context": "ENTRY",
                "submission_phase": "READY_TO_SUBMIT",
            },
        )

        def persist_updates(current_state, symbol, updates):
            current_state["positions"][symbol].update(updates)
            return True

        def remove_marker(current_state, symbol):
            current_state["positions"].pop(symbol, None)
            return True

        with patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.get_open_position_details",
            return_value=live_position(),
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=persist_updates,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=True,
        ) as close, patch(
            "main.remove_position_state",
            side_effect=remove_marker,
        ):
            attempted, unresolved = main.reconcile_interrupted_dca_submissions(
                live_position(),
                state,
            )

        self.assertTrue(attempted)
        self.assertFalse(unresolved)
        self.assertEqual(
            close.call_args.kwargs["context"],
            "INTERRUPTED_ENTRY_READY_TO_SUBMIT",
        )
        self.assertNotIn(SYMBOL, state["positions"])

    def test_interrupted_entry_marker_cleans_verified_flat_state(self):
        state = position_state(
            pending_submission={
                "context": "ENTRY",
                "submission_phase": "READY_TO_SUBMIT",
            },
        )

        def remove_marker(current_state, symbol):
            current_state["positions"].pop(symbol, None)
            return True

        with patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.get_open_position_details",
            return_value={},
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ) as cleanup, patch(
            "main.remove_position_state",
            side_effect=remove_marker,
        ), patch(
            "main.fail_safe_close_unprotected_position",
        ) as close:
            attempted, unresolved = main.reconcile_interrupted_dca_submissions(
                {},
                state,
            )

        self.assertTrue(attempted)
        self.assertFalse(unresolved)
        cleanup.assert_called_once_with(SYMBOL)
        close.assert_not_called()
        self.assertNotIn(SYMBOL, state["positions"])

    def test_interrupted_dca_marker_and_close_failure_remains_retryable(self):
        state = position_state(
            pending_dca={"submission_phase": "READY_TO_SUBMIT"},
        )

        with patch(
            "main.load_trade_state",
            return_value=state,
        ), patch(
            "main.get_open_position_details",
            return_value=live_position(),
        ), patch(
            "main.update_position_runtime_fields",
            return_value=False,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=False,
        ) as close, patch("main.log_error"):
            attempted, unresolved = main.reconcile_interrupted_dca_submissions(
                live_position(),
                state,
            )

        self.assertTrue(attempted)
        self.assertEqual(unresolved, {SYMBOL})
        close.assert_called_once()
        self.assertFalse(main.shutdown_event.is_set())
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_active_dca_lock_defers_interrupted_reconciliation(self):
        state = position_state(
            pending_dca={"submission_phase": "ORDER_RETURNED"},
        )
        lock = main.get_dca_lock(SYMBOL)
        self.assertTrue(lock.acquire(blocking=False))

        try:
            with patch(
                "main.fail_safe_close_unprotected_position",
            ) as close:
                attempted, unresolved = (
                    main.reconcile_interrupted_dca_submissions(
                        live_position(),
                        state,
                    )
                )
        finally:
            lock.release()

        self.assertTrue(attempted)
        self.assertEqual(unresolved, {SYMBOL})
        close.assert_not_called()

    def test_fresh_completed_dca_state_prevents_stale_fail_close(self):
        stale_state = position_state(
            pending_dca={"submission_phase": "READY_TO_SUBMIT"},
        )
        fresh_state = position_state(
            pending_dca={"submission_phase": "COMPLETE"},
        )

        with patch(
            "main.load_trade_state",
            return_value=fresh_state,
        ), patch(
            "main.get_open_position_details",
        ) as live_details, patch(
            "main.fail_safe_close_unprotected_position",
        ) as close:
            attempted, unresolved = main.reconcile_interrupted_dca_submissions(
                live_position(),
                stale_state,
            )

        self.assertTrue(attempted)
        self.assertFalse(unresolved)
        live_details.assert_not_called()
        close.assert_not_called()
        self.assertEqual(
            stale_state["positions"][SYMBOL]["pending_dca"][
                "submission_phase"
            ],
            "COMPLETE",
        )

    def test_post_dca_double_failure_keeps_all_safety_reasons_retryable(self):
        for reason in (
            "DCA_POST_FILL_TOPOLOGY",
            "DCA_POST_FILL_STOP_MISSING",
            "DCA_RISK_OVERRUN",
        ):
            with self.subTest(reason=reason):
                state = position_state(
                    pending_dca={"submission_phase": "ORDER_RETURNED"},
                )
                main.shutdown_event.clear()
                main.entry_quarantined_symbols.clear()

                with patch(
                    "main.persist_dca_fail_close_pending",
                    return_value=False,
                ), patch(
                    "main.fail_safe_close_unprotected_position",
                    return_value=False,
                ) as close, patch("main.log_error"):
                    closed = main.fail_close_post_dca_safety_violation(
                        state,
                        SYMBOL,
                        reason,
                        live_position()[SYMBOL],
                        95.0,
                    )

                self.assertFalse(closed)
                close.assert_called_once()
                self.assertFalse(main.shutdown_event.is_set())
                self.assertTrue(
                    main.state_requires_urgent_safety_retry(state)
                )
                self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_short_explicit_retry_wait_is_not_short_circuited(self):
        urgent_state = {
            "positions": {},
            "pending_executions": {SYMBOL: {"context": "ENTRY"}},
        }
        started = time.monotonic()

        with patch.object(
            config,
            "PENDING_EXECUTION_RECONCILE_SECONDS",
            0.5,
        ), patch(
            "main.load_trade_state",
            return_value=urgent_state,
        ) as load_state, patch("main.log_info"):
            completed = main.wait_for_next_scan("TEST_RETRY", 0.05)

        elapsed = time.monotonic() - started
        self.assertTrue(completed)
        self.assertGreaterEqual(elapsed, 0.04)
        load_state.assert_not_called()

    def test_normal_wait_wakes_when_urgent_state_appears(self):
        urgent_state = {
            "positions": {},
            "pending_executions": {SYMBOL: {"context": "ENTRY"}},
        }
        started = time.monotonic()

        with patch.object(
            config,
            "PENDING_EXECUTION_RECONCILE_SECONDS",
            0.5,
        ), patch(
            "main.load_trade_state",
            return_value=urgent_state,
        ) as load_state, patch("main.log_info"), patch("main.log_warning"):
            completed = main.wait_for_next_scan("TEST_NORMAL", 5)

        elapsed = time.monotonic() - started
        self.assertTrue(completed)
        load_state.assert_called_once_with()
        self.assertGreaterEqual(elapsed, 0.4)
        self.assertLess(elapsed, 1.5)


if __name__ == "__main__":
    unittest.main()
