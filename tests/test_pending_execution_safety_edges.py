import unittest
from unittest.mock import patch


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


SYMBOL = "SAFETYUSDT"


def dca_pending(**updates):
    pending = {
        "symbol": SYMBOL,
        "side": "BUY",
        "context": "DCA_LEVEL_1",
        "requested_quantity": 0.25,
        "pre_position_amount": 1.0,
        "reference_price": 100.0,
        "position_side": "BOTH",
        "signal_type": "TREND",
        "dca_level": 1,
        "hard_stop_price": 90.0,
        "client_order_ids": "v7-dca-test",
        "order_ids": "",
        "execution_mode": "SMART_IOC_MARKET_FALLBACK",
        "emergency_protection_secured": False,
    }
    pending.update(updates)
    return pending


def terminal_result(executed_quantity=0.0):
    return {
        "order_terminal": True,
        "orders": [],
        "executed_quantity": executed_quantity,
        "average_fill_price": 0.0,
        "order_ids": "",
        "client_order_ids": "v7-dca-test",
        "verification_attempts": 1,
        "error": "",
    }


def live_position(amount=1.0):
    return {
        "amount": amount,
        "side": "BUY",
        "position_side": "BOTH",
        "entry_price": 100.0,
        "mark_price": 99.0,
    }


def dca_state(pending):
    return {
        "positions": {
            SYMBOL: {
                "managed_by_bot": True,
                "side": "BUY",
                "campaign_stop_price": 90.0,
                "pending_dca": {
                    "level": 1,
                    "execution_unsettled": True,
                },
            },
        },
        "pending_executions": {SYMBOL: pending},
    }


class PendingDcaTerminalSafetyTests(unittest.TestCase):
    def setUp(self):
        main.entry_quarantined_symbols.discard(SYMBOL)

    def tearDown(self):
        main.entry_quarantined_symbols.discard(SYMBOL)

    def test_terminal_zero_fill_reverifies_stop_without_cancelling_protection(self):
        pending = dca_pending()
        state = dca_state(pending)
        position = live_position()
        events = []

        def persist(current_state, symbol, data):
            events.append("protection_verified_and_persisted")
            current_state.setdefault("pending_executions", {})[symbol] = data
            return True

        def clear_reservation(current_state, symbol, level=None):
            events.append("reservation_cleared")
            item = current_state["positions"][symbol]
            if item.get("pending_dca", {}).get("level") == level:
                item.pop("pending_dca", None)
            return True

        def remove_pending(current_state, symbol):
            events.append("pending_execution_removed")
            current_state.setdefault("pending_executions", {}).pop(symbol, None)
            return True

        def exact_stop(*args, **kwargs):
            events.append("exact_stop_verified")
            return {
                "order_id": 7001,
                "sl_price": 90.0,
                "working_type": "MARK_PRICE",
                "close_position": True,
            }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(0.0),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, position),
        ), patch(
            "main.find_matching_close_position_stop",
            side_effect=exact_stop,
        ) as find_stop, patch(
            "main.upsert_pending_execution",
            side_effect=persist,
        ), patch(
            "main.clear_dca_reservation",
            side_effect=clear_reservation,
        ) as clear_dca, patch(
            "main.remove_pending_execution",
            side_effect=remove_pending,
        ) as remove_execution, patch(
            "main.cancel_open_protection_orders",
        ) as cancel_protection, patch(
            "main.place_stop_loss_only",
        ) as restore_stop, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_warning"), patch("main.log_error"):
            main.reconcile_pending_executions(state)

        find_stop.assert_called_once_with(
            SYMBOL,
            main.SIDE_BUY,
            90.0,
            position_side="BOTH",
        )
        restore_stop.assert_not_called()
        cancel_protection.assert_not_called()
        fail_safe_close.assert_not_called()
        clear_dca.assert_called_once_with(state, SYMBOL, 1)
        remove_execution.assert_called_once_with(state, SYMBOL)
        shutdown.assert_not_called()
        self.assertEqual(
            events,
            [
                "exact_stop_verified",
                "protection_verified_and_persisted",
                "reservation_cleared",
                "pending_execution_removed",
            ],
        )
        self.assertNotIn("pending_dca", state["positions"][SYMBOL])
        self.assertNotIn(SYMBOL, state["pending_executions"])

    def test_snapshot_and_update_failure_keeps_original_pending_retry_alive(self):
        pending = dca_pending()
        state = dca_state(pending)

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(0.0),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(False, None),
        ), patch(
            "main.upsert_pending_execution",
            return_value=False,
        ), patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        shutdown.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_terminal_flat_cleanup_clears_pending_quarantine(self):
        pending = dca_pending(
            context="ENTRY",
            dca_level=None,
            pre_position_amount=0.0,
        )
        state = {"positions": {}, "pending_executions": {SYMBOL: pending}}
        main.entry_quarantined_symbols.add(SYMBOL)

        def remove_pending(current_state, symbol):
            current_state["pending_executions"].pop(symbol, None)
            return True

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(0.0),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ), patch(
            "main.remove_pending_execution",
            side_effect=remove_pending,
        ), patch("main.log_warning"):
            main.reconcile_pending_executions(state)

        self.assertNotIn(SYMBOL, state["pending_executions"])
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)

    def test_terminal_execution_or_changed_topology_uses_fail_safe_close(self):
        scenarios = (
            ("reported_execution", 0.25, 1.0),
            ("changed_live_topology", 0.0, 0.4),
        )

        for label, executed_quantity, live_amount in scenarios:
            with self.subTest(label=label):
                pending = dca_pending()
                state = dca_state(pending)
                position = live_position(live_amount)

                def remove_pending(current_state, symbol):
                    current_state.setdefault("pending_executions", {}).pop(
                        symbol,
                        None,
                    )
                    return True

                with patch(
                    "main.reconcile_execution_client_orders",
                    return_value=terminal_result(executed_quantity),
                ), patch(
                    "main._pending_execution_live_detail",
                    return_value=(True, position),
                ), patch(
                    "main.fail_safe_close_unprotected_position",
                    return_value=True,
                ) as fail_safe_close, patch(
                    "main.remove_pending_execution",
                    side_effect=remove_pending,
                ) as remove_execution, patch(
                    "main.clear_dca_reservation",
                ) as clear_dca, patch(
                    "main._secure_pending_execution_protection",
                ) as secure_protection, patch(
                    "main.cancel_open_protection_orders",
                ) as cancel_protection, patch("main.log_warning"), patch(
                    "main.log_error",
                ):
                    main.reconcile_pending_executions(state)

                fail_safe_close.assert_called_once_with(
                    SYMBOL,
                    position_side="BOTH",
                    reference_price=99.0,
                    context="DCA_LEVEL_1_LATE_RECONCILIATION",
                )
                clear_dca.assert_not_called()
                secure_protection.assert_not_called()
                cancel_protection.assert_not_called()
                remove_execution.assert_called_once_with(state, SYMBOL)
                self.assertIn("pending_dca", state["positions"][SYMBOL])
                self.assertNotIn(SYMBOL, state["pending_executions"])

    def test_nonterminal_changed_topology_secures_live_reopened_leg(self):
        pending = dca_pending(pre_average_price=100.0)
        state = dca_state(pending)
        reopened_position = live_position(amount=0.3)
        nonterminal = terminal_result(0.0)
        nonterminal["order_terminal"] = False

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=nonterminal,
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, reopened_position),
        ), patch(
            "main._secure_pending_execution_protection",
            return_value=True,
        ) as secure_protection, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main.upsert_pending_execution",
        ) as persist_without_protection, patch("main.log_warning"), patch(
            "main.log_error",
        ):
            main.reconcile_pending_executions(state)

        self.assertEqual(pending["observed_position_delta"], 0)
        secure_protection.assert_called_once_with(
            state,
            SYMBOL,
            pending,
            reopened_position,
        )
        fail_safe_close.assert_not_called()
        persist_without_protection.assert_not_called()

    def test_nonterminal_reported_fill_secures_even_when_net_amount_matches(self):
        pending = dca_pending(pre_average_price=0.0)
        state = dca_state(pending)
        net_unchanged_position = live_position(amount=1.0)
        nonterminal = terminal_result(0.25)
        nonterminal["order_terminal"] = False

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=nonterminal,
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, net_unchanged_position),
        ), patch(
            "main._secure_pending_execution_protection",
            return_value=True,
        ) as secure_protection, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        secure_protection.assert_called_once_with(
            state,
            SYMBOL,
            pending,
            net_unchanged_position,
        )
        fail_safe_close.assert_not_called()

    def test_unprotected_unsettled_close_failure_retries_when_state_is_saved(self):
        pending = dca_pending(pre_average_price=100.0)
        state = dca_state(pending)
        changed_position = live_position(amount=1.25)
        nonterminal = terminal_result(0.25)
        nonterminal["order_terminal"] = False

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=nonterminal,
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, changed_position),
        ), patch(
            "main._secure_pending_execution_protection",
            return_value=False,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=False,
        ) as fail_safe_close, patch(
            "main.upsert_pending_execution",
            return_value=True,
        ) as persist, patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        fail_safe_close.assert_called_once()
        persist.assert_called_once_with(state, SYMBOL, pending)
        shutdown.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])

    def test_unprotected_unsettled_close_and_update_failure_keeps_retrying(self):
        pending = dca_pending(pre_average_price=100.0)
        state = dca_state(pending)
        changed_position = live_position(amount=1.25)
        nonterminal = terminal_result(0.25)
        nonterminal["order_terminal"] = False

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=nonterminal,
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, changed_position),
        ), patch(
            "main._secure_pending_execution_protection",
            return_value=False,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=False,
        ), patch(
            "main.upsert_pending_execution",
            return_value=False,
        ), patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        shutdown.assert_not_called()
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)
        self.assertIn(SYMBOL, state["pending_executions"])

    def test_terminal_positive_entry_fill_with_stale_topology_stays_pending(self):
        pending = dca_pending(
            context="ENTRY",
            dca_level=None,
            pre_position_amount=0.0,
        )
        state = {"positions": {}, "pending_executions": {SYMBOL: pending}}

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(0.25),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.find_matching_close_position_stop",
            return_value={"order_id": 7002},
        ) as find_stop, patch(
            "main.upsert_pending_execution",
            return_value=True,
        ) as persist, patch(
            "main.place_stop_loss_only",
        ) as restore_stop, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main.cancel_open_protection_orders",
        ) as cancel_protection, patch(
            "main.remove_pending_execution",
        ) as remove_pending, patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        find_stop.assert_called_once_with(
            SYMBOL,
            main.SIDE_BUY,
            90.0,
            position_side="BOTH",
        )
        persist.assert_called_once_with(state, SYMBOL, pending)
        restore_stop.assert_not_called()
        fail_safe_close.assert_not_called()
        cancel_protection.assert_not_called()
        remove_pending.assert_not_called()
        shutdown.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertTrue(pending["terminal_fill_topology_pending"])
        self.assertTrue(pending["terminal_fill_protection_secured"])

    def test_terminal_positive_entry_restores_confirmed_missing_stop(self):
        pending = dca_pending(
            context="ENTRY",
            dca_level=None,
            pre_position_amount=0.0,
        )
        state = {"positions": {}, "pending_executions": {SYMBOL: pending}}

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(0.25),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.find_matching_close_position_stop",
            return_value={},
        ), patch(
            "main.place_stop_loss_only",
            return_value={"ok": True, "sl_price": 90.0},
        ) as restore_stop, patch(
            "main.upsert_pending_execution",
            return_value=True,
        ), patch.object(
            main.shutdown_event,
            "set",
        ) as shutdown, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        restore_stop.assert_called_once_with(
            SYMBOL,
            main.SIDE_BUY,
            100.0,
            None,
            signal_type="TREND",
            position_side="BOTH",
            sl_price_override=90.0,
        )
        shutdown.assert_not_called()
        self.assertTrue(pending["terminal_fill_protection_secured"])


class FailSafeAlreadyFlatCleanupTests(unittest.TestCase):
    def setUp(self):
        main.entry_quarantined_symbols.discard(SYMBOL)

    def tearDown(self):
        main.entry_quarantined_symbols.discard(SYMBOL)

    def test_already_flat_returns_success_after_verified_cleanup(self):
        main.entry_quarantined_symbols.add(SYMBOL)

        with patch(
            "main.get_open_position_details",
            return_value={},
        ) as position_snapshot, patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ) as cleanup, patch(
            "main.close_position_market",
        ) as market_close, patch("main.log_warning"), patch("main.log_error"):
            result = main.fail_safe_close_unprotected_position(
                SYMBOL,
                context="TEST_ALREADY_FLAT",
            )

        self.assertTrue(result)
        position_snapshot.assert_called_once_with(SYMBOL, force=True)
        cleanup.assert_called_once_with(SYMBOL)
        market_close.assert_not_called()
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)

    def test_already_flat_cleanup_failure_returns_false_and_quarantines(self):
        with patch(
            "main.get_open_position_details",
            return_value={},
        ), patch(
            "main.cancel_open_protection_orders",
            return_value=False,
        ) as cleanup, patch(
            "main.close_position_market",
        ) as market_close, patch("main.log_warning"), patch("main.log_error"):
            result = main.fail_safe_close_unprotected_position(
                SYMBOL,
                context="TEST_ALREADY_FLAT_CLEANUP_FAILURE",
            )

        self.assertFalse(result)
        cleanup.assert_called_once_with(SYMBOL)
        market_close.assert_not_called()
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)


if __name__ == "__main__":
    unittest.main()
