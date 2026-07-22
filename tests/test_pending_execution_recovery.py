import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main
    import trade_state


SYMBOL = "TESTUSDT"


def pending_execution(**updates):
    pending = {
        "symbol": SYMBOL,
        "side": "BUY",
        "context": "ENTRY",
        "requested_quantity": 1.0,
        "pre_position_amount": 0.0,
        "reference_price": 100.0,
        "position_side": "BOTH",
        "signal_type": "TREND",
        "dca_level": None,
        "client_order_ids": "cid-ioc,cid-market",
        "order_ids": "",
        "execution_mode": "SMART_IOC_MARKET_FALLBACK",
        "emergency_protection_secured": False,
    }
    pending.update(updates)
    return pending


def terminal_result(terminal):
    return {
        "order_terminal": terminal,
        "orders": [],
        "executed_quantity": 0.0,
        "average_fill_price": 0.0,
        "order_ids": "",
        "client_order_ids": "cid-ioc,cid-market",
        "verification_attempts": 1,
        "error": "" if terminal else "status unavailable",
    }


def definitive_absence_result():
    return {
        "order_terminal": False,
        "orders": [],
        "executed_quantity": 0.0,
        "average_fill_price": 0.0,
        "order_ids": "",
        "client_order_ids": "cid-ioc,cid-market",
        "verification_attempts": 2,
        "error": "",
        "client_order_ids_valid": True,
        "client_outcomes": [
            {
                "client_order_id": "cid-ioc",
                "outcome": "DEFINITIVELY_ABSENT",
            },
            {
                "client_order_id": "cid-market",
                "outcome": "DEFINITIVELY_ABSENT",
            },
        ],
        "all_definitively_absent": True,
        "any_order_seen": False,
        "max_executed_quantity_seen": 0.0,
        "lookup_uncertain": False,
        "absence_evidence": {
            "open_orders_sweep_ok": True,
            "order_history_sweep_ok": True,
            "definitive_count": 2,
        },
    }


def safe_entry_submission_marker():
    return {
        "symbol": SYMBOL,
        "managed_by_bot": True,
        "initial_quantity": 0.0,
        "position_management_status": "ENTRY_READY_TO_SUBMIT",
        "pending_submission": {
            "context": "ENTRY",
            "submission_phase": "READY_TO_SUBMIT",
        },
    }


def persist_in_memory(state, symbol, data):
    state.setdefault("pending_executions", {})[symbol] = data
    return True


def remove_in_memory(state, symbol):
    state.setdefault("pending_executions", {}).pop(symbol, None)
    return True


class PendingExecutionRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.state_temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.state_temp_dir.cleanup)
        state_path_patch = patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(self.state_temp_dir.name) / "open_trades_v7.json"),
        )
        state_path_patch.start()
        self.addCleanup(state_path_patch.stop)
        known_symbol_patch = patch(
            "main.is_known_futures_symbol",
            return_value=True,
        )
        known_symbol_patch.start()
        self.addCleanup(known_symbol_patch.stop)
        main.entry_quarantined_symbols.discard(SYMBOL)
        self.addCleanup(main.entry_quarantined_symbols.discard, SYMBOL)

    def test_nonterminal_no_fill_stays_pending_without_close_or_new_order(self):
        state = {
            "positions": {},
            "pending_executions": {SYMBOL: pending_execution()},
        }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(False),
        ) as reconcile, patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.upsert_pending_execution",
            side_effect=persist_in_memory,
        ) as upsert, patch(
            "main._secure_pending_execution_protection",
        ) as protect, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main.place_market_order",
        ) as new_order, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        reconcile.assert_called_once_with(
            SYMBOL,
            "cid-ioc,cid-market",
            cancel_unsettled=True,
        )
        upsert.assert_called_once()
        protect.assert_not_called()
        fail_safe_close.assert_not_called()
        new_order.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertEqual(
            state["pending_executions"][SYMBOL]["observed_position_delta"],
            0,
        )

    def test_nonterminal_visible_fill_gets_emergency_full_position_protection(self):
        state = {
            "positions": {},
            "pending_executions": {SYMBOL: pending_execution()},
        }
        live_detail = {
            "amount": 0.4,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 101.0,
            "mark_price": 102.0,
        }
        protection = {
            "ok": True,
            "tp_price": 111.0,
            "sl_price": 96.0,
        }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(False),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, live_detail),
        ), patch(
            "main.get_signal_frames",
            return_value=(Mock(), Mock(), Mock()),
        ), patch(
            "main.place_tp_sl_with_recovery",
            return_value=protection,
        ) as place_protection, patch(
            "main.upsert_pending_execution",
            side_effect=persist_in_memory,
        ) as upsert, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main.place_market_order",
        ) as new_order, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        place_protection.assert_called_once()
        args = place_protection.call_args.args
        self.assertEqual(args[:4], (SYMBOL, main.SIDE_BUY, 101.0, 0.4))
        self.assertFalse(place_protection.call_args.kwargs["enable_multi_tp"])
        self.assertEqual(
            place_protection.call_args.kwargs["position_side"],
            "BOTH",
        )
        upsert.assert_called_once()
        fail_safe_close.assert_not_called()
        new_order.assert_not_called()
        recovered = state["pending_executions"][SYMBOL]
        self.assertTrue(recovered["emergency_protection_secured"])
        self.assertEqual(
            recovered["emergency_protection_mode"],
            "FULL_POSITION_TP_SL",
        )
        self.assertEqual(recovered["emergency_tp_price"], 111.0)
        self.assertEqual(recovered["emergency_sl_price"], 96.0)

    def test_terminal_no_new_fill_clears_pending_and_dca_reservation(self):
        pending = pending_execution(
            context="DCA_LEVEL_2",
            dca_level=2,
            pre_position_amount=1.0,
        )
        state = {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "pending_dca": {
                        "level": 2,
                        "execution_unsettled": True,
                    },
                },
            },
            "pending_executions": {SYMBOL: pending},
        }
        unchanged_position = {
            "amount": 1.0,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 100.0,
            "mark_price": 100.0,
        }

        def clear_reservation(current_state, symbol, level=None):
            item = current_state["positions"][symbol]
            if level == item.get("pending_dca", {}).get("level"):
                item.pop("pending_dca", None)

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(True),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, unchanged_position),
        ), patch(
            "main.clear_dca_reservation",
            side_effect=clear_reservation,
        ) as clear_dca, patch(
            "main.remove_pending_execution",
            side_effect=remove_in_memory,
        ) as remove_pending, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main._secure_pending_execution_protection",
            return_value=True,
        ) as secure_protection, patch(
            "main.cancel_open_protection_orders",
        ) as cancel_protection, patch("main.log_warning"):
            main.reconcile_pending_executions(state)

        secure_protection.assert_called_once_with(
            state,
            SYMBOL,
            pending,
            unchanged_position,
        )
        cancel_protection.assert_not_called()
        clear_dca.assert_called_once_with(state, SYMBOL, 2)
        remove_pending.assert_called_once_with(state, SYMBOL)
        fail_safe_close.assert_not_called()
        self.assertNotIn("pending_dca", state["positions"][SYMBOL])
        self.assertNotIn(SYMBOL, state["pending_executions"])

    def test_terminal_late_fill_is_closed_before_pending_is_cleared(self):
        state = {
            "positions": {},
            "pending_executions": {SYMBOL: pending_execution()},
        }
        late_position = {
            "amount": 0.25,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 100.5,
            "mark_price": 101.0,
        }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(True),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, late_position),
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=True,
        ) as fail_safe_close, patch(
            "main.remove_pending_execution",
            side_effect=remove_in_memory,
        ) as remove_pending, patch(
            "main._secure_pending_execution_protection",
        ) as protect, patch("main.log_warning"):
            main.reconcile_pending_executions(state)

        fail_safe_close.assert_called_once_with(
            SYMBOL,
            position_side="BOTH",
            reference_price=101.0,
            context="ENTRY_LATE_RECONCILIATION",
        )
        remove_pending.assert_called_once_with(state, SYMBOL)
        protect.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])

    def test_entry_guard_blocks_symbol_with_pending_execution(self):
        state = {
            "positions": {},
            "pending_executions": {SYMBOL: pending_execution()},
        }
        candidate = {
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
        }
        position_details = {}
        open_positions = {}

        with patch.object(
            main.shutdown_event,
            "is_set",
            return_value=False,
        ), patch("main.place_market_order") as place_order, patch(
            "main.market_flow_hard_veto",
        ) as flow_veto, patch("main.log_warning"):
            result = main.execute_entry_candidate(
                candidate,
                state,
                position_details,
                open_positions,
                None,
                Mock(),
            )

        self.assertEqual(result, (position_details, open_positions, False))
        place_order.assert_not_called()
        flow_veto.assert_not_called()

    def test_dca_guard_blocks_symbol_with_pending_execution(self):
        state = {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "side": "BUY",
                    "dca_count": 0,
                },
            },
            "pending_executions": {SYMBOL: pending_execution()},
        }
        position_detail = {
            "amount": 1.0,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 100.0,
        }

        with patch.object(config, "DCA_ENABLED", True), patch.object(
            main.shutdown_event,
            "is_set",
            return_value=False,
        ), patch("main.place_market_order") as place_order, patch(
            "main.get_signal_frames",
        ) as get_frames, patch("main.log_warning"):
            main.manage_dca_position(
                SYMBOL,
                state,
                position_detail,
                None,
                None,
            )

        place_order.assert_not_called()
        get_frames.assert_not_called()

    def test_urgent_pending_execution_globally_pauses_other_symbol_dca(self):
        other_symbol = "OTHERUSDT"
        state = {
            "positions": {
                other_symbol: {
                    "managed_by_bot": True,
                    "campaign_risk_version": 2,
                    "side": "BUY",
                    "dca_count": 0,
                },
            },
            "pending_executions": {SYMBOL: pending_execution()},
        }
        position_detail = {
            "amount": 1.0,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 100.0,
        }

        with patch.object(config, "DCA_ENABLED", True), patch.object(
            main.shutdown_event,
            "is_set",
            return_value=False,
        ), patch("main.place_market_order") as place_order, patch(
            "main.reserve_dca_level",
        ) as reserve, patch("main.get_signal_frames") as get_frames, patch(
            "main.log_warning",
        ):
            main.manage_dca_position(
                other_symbol,
                state,
                position_detail,
                None,
                None,
            )

        reserve.assert_not_called()
        place_order.assert_not_called()
        get_frames.assert_not_called()

    def test_unsettled_dca_persists_fresh_pre_order_position_amount(self):
        state = {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "side": "BUY",
                    "dca_count": 0,
                    "initial_entry": 100.0,
                    "campaign_stop_price": 90.0,
                    "hard_stop_price": 90.0,
                },
            },
            "pending_executions": {},
        }
        position_detail = {
            "amount": 1.0,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 100.0,
        }
        fresh_position = {
            "amount": 1.25,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 99.5,
        }
        order = {"clientOrderId": "v7-dca-ioc"}
        reconciliation = {
            "order_terminal": False,
            "client_order_ids": "v7-dca-ioc",
        }

        config_updates = {
            "DCA_ENABLED": True,
            "DCA_STRICT_GUARD_ENABLED": False,
            "DCA_TRIGGER_MODE": "static_roi",
            "DCA_MAX_ADVERSE_ROI": 0,
            "DCA_MIN_SECONDS_BETWEEN_ORDERS": 0,
            "TP1_RUNNER_DISABLE_DCA": True,
            "DCA_FIXED_RISK_ENABLED": False,
            "DCA_RECOVERY_CONFIRMATION_ENABLED": False,
            "RISK_BASED_POSITION_SIZING_ENABLED": False,
        }
        main_updates = {
            "get_dca_order_margin": Mock(return_value=10.0),
            "get_dca_trigger_roi": Mock(return_value=1.0),
            "get_dca_trigger_entry": Mock(return_value=100.0),
            "get_position_adverse_roi": Mock(return_value=2.0),
            "get_balance": Mock(return_value=1000.0),
            "get_mark_price": Mock(return_value=98.0),
            "load_trade_state": Mock(return_value=state),
            "find_matching_close_position_stop": Mock(
                return_value={"order_id": 55, "sl_price": 90.0}
            ),
            "calculate_position_size": Mock(return_value=0.5),
            "validate_min_notional": Mock(return_value=(True, 49.0)),
            "set_margin_type": Mock(return_value=True),
            "setup_leverage": Mock(return_value=True),
            "reserve_dca_level": Mock(return_value=(True, "reserved")),
            "refresh_dca_position_before_order": Mock(
                return_value=(fresh_position, "fresh")
            ),
            "place_market_order": Mock(return_value=order),
            "get_execution_reconciliation": Mock(
                return_value=reconciliation
            ),
            "is_reconciled_execution_settled": Mock(return_value=False),
            "update_position_runtime_fields": Mock(return_value=True),
            "persist_pending_execution": Mock(return_value=True),
            "reconcile_pending_executions": Mock(),
            "log_info": Mock(),
            "log_warning": Mock(),
            "log_error": Mock(),
        }

        with patch.multiple(config, **config_updates), patch.object(
            main.shutdown_event,
            "is_set",
            return_value=False,
        ), patch.multiple(main, **main_updates):
            main.manage_dca_position(
                SYMBOL,
                state,
                position_detail,
                None,
                None,
                current_price_override=98.0,
            )

        main_updates["place_market_order"].assert_called_once_with(
            SYMBOL,
            main.SIDE_BUY,
            0.5,
            pre_position_amount=1.25,
            pre_average_price=99.5,
            reference_price=98.0,
            context="DCA_LEVEL_1",
        )
        persist_call = main_updates["persist_pending_execution"].call_args
        self.assertIsNotNone(persist_call)
        self.assertEqual(persist_call.args[5], 1.25)
        self.assertEqual(persist_call.kwargs["dca_level"], 1)
        self.assertEqual(persist_call.kwargs["pre_average_price"], 99.5)
        main_updates["reconcile_pending_executions"].assert_called_once_with(
            state
        )


class PendingEntryAbsenceRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.state_temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.state_temp_dir.cleanup)
        state_path_patch = patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(self.state_temp_dir.name) / "open_trades_v7.json"),
        )
        state_path_patch.start()
        self.addCleanup(state_path_patch.stop)
        known_symbol_patch = patch(
            "main.is_known_futures_symbol",
            return_value=True,
        )
        known_symbol_patch.start()
        self.addCleanup(known_symbol_patch.stop)
        main.entry_quarantined_symbols.discard(SYMBOL)
        self.addCleanup(main.entry_quarantined_symbols.discard, SYMBOL)

    @staticmethod
    def old_pending(**updates):
        pending = pending_execution(
            created_at=(
                datetime.now() - timedelta(seconds=120)
            ).isoformat(timespec="seconds"),
            order_seen=False,
            max_executed_quantity_seen=0.0,
            consecutive_absence_confirmations=0,
            absence_first_confirmed_at=None,
        )
        pending.update(updates)
        return pending

    def test_three_proven_absence_cycles_atomically_clear_entry_and_marker(self):
        pending = self.old_pending()
        state = {
            "positions": {SYMBOL: safe_entry_submission_marker()},
            "pending_executions": {SYMBOL: pending},
        }
        persisted_counts = []

        def persist(current_state, symbol, data):
            persisted_counts.append(
                data.get("consecutive_absence_confirmations")
            )
            return persist_in_memory(current_state, symbol, data)

        def atomic_clear(current_state, symbol, expected_ids):
            self.assertEqual(
                expected_ids,
                ("cid-ioc", "cid-market"),
            )
            self.assertEqual(
                current_state["pending_executions"][symbol][
                    "consecutive_absence_confirmations"
                ],
                3,
            )
            current_state["pending_executions"].pop(symbol, None)
            current_state["positions"].pop(symbol, None)
            return True, "CLEARED"

        with patch.object(
            config,
            "PENDING_EXECUTION_ABSENCE_GRACE_SECONDS",
            60,
        ), patch.object(
            config,
            "PENDING_EXECUTION_ABSENCE_CONFIRMATIONS",
            3,
        ), patch(
            "main.reconcile_execution_client_orders",
            return_value=definitive_absence_result(),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.get_open_position_details",
            return_value={},
        ) as global_snapshot, patch(
            "main.upsert_pending_execution",
            side_effect=persist,
        ) as upsert, patch(
            "main.clear_confirmed_absent_entry_execution",
            side_effect=atomic_clear,
        ) as atomic_cleanup, patch(
            "main.cancel_open_protection_orders",
        ) as cancel_protection, patch(
            "main.fail_safe_close_unprotected_position",
        ) as fail_safe_close, patch(
            "main.place_market_order",
        ) as new_order, patch("main.log_warning"), patch("main.log_error"):
            main.reconcile_pending_executions(state)
            main.reconcile_pending_executions(state)
            main.reconcile_pending_executions(state)

        self.assertEqual(persisted_counts, [1, 2])
        self.assertEqual(upsert.call_count, 2)
        self.assertEqual(global_snapshot.call_count, 3)
        global_snapshot.assert_called_with(force=True)
        atomic_cleanup.assert_called_once()
        cancel_protection.assert_not_called()
        fail_safe_close.assert_not_called()
        new_order.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])
        self.assertNotIn(SYMBOL, state["positions"])
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)

    def test_lookup_uncertainty_resets_consecutive_absence_proof(self):
        pending = self.old_pending(
            consecutive_absence_confirmations=2,
            absence_first_confirmed_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )
        state = {
            "positions": {SYMBOL: safe_entry_submission_marker()},
            "pending_executions": {SYMBOL: pending},
        }
        uncertain = definitive_absence_result()
        uncertain.update({
            "all_definitively_absent": False,
            "lookup_uncertain": True,
            "client_outcomes": [{
                "client_order_id": "cid-ioc",
                "outcome": "UNKNOWN",
            }],
        })

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=uncertain,
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.upsert_pending_execution",
            side_effect=persist_in_memory,
        ), patch(
            "main.get_open_position_details",
        ) as global_snapshot, patch(
            "main.clear_confirmed_absent_entry_execution",
        ) as atomic_cleanup, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        global_snapshot.assert_not_called()
        atomic_cleanup.assert_not_called()
        self.assertEqual(pending["consecutive_absence_confirmations"], 0)
        self.assertIsNone(pending["absence_first_confirmed_at"])
        self.assertEqual(
            pending["last_absence_reset_reason"],
            "ORDER_LOOKUP_UNCERTAIN",
        )

    def test_order_and_execution_observations_never_decrease(self):
        pending = self.old_pending(
            order_seen=True,
            max_executed_quantity_seen=0.25,
            consecutive_absence_confirmations=2,
            absence_first_confirmed_at=datetime.now().isoformat(
                timespec="seconds"
            ),
        )
        state = {
            "positions": {SYMBOL: safe_entry_submission_marker()},
            "pending_executions": {SYMBOL: pending},
        }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=definitive_absence_result(),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, None),
        ), patch(
            "main.upsert_pending_execution",
            side_effect=persist_in_memory,
        ), patch(
            "main.get_open_position_details",
        ) as global_snapshot, patch(
            "main.clear_confirmed_absent_entry_execution",
        ) as atomic_cleanup, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        global_snapshot.assert_not_called()
        atomic_cleanup.assert_not_called()
        self.assertTrue(pending["order_seen"])
        self.assertEqual(pending["max_executed_quantity_seen"], 0.25)
        self.assertEqual(pending["consecutive_absence_confirmations"], 0)

    def test_successful_nonterminal_fail_close_retains_origin_marker(self):
        pending = self.old_pending()
        state = {
            "positions": {SYMBOL: safe_entry_submission_marker()},
            "pending_executions": {SYMBOL: pending},
        }
        live_detail = {
            "amount": 0.4,
            "side": "BUY",
            "position_side": "BOTH",
            "entry_price": 101.0,
            "mark_price": 100.0,
        }

        with patch(
            "main.reconcile_execution_client_orders",
            return_value=terminal_result(False),
        ), patch(
            "main._pending_execution_live_detail",
            return_value=(True, live_detail),
        ), patch(
            "main._secure_pending_execution_protection",
            return_value=False,
        ), patch(
            "main.fail_safe_close_unprotected_position",
            return_value=True,
        ) as fail_safe_close, patch(
            "main.upsert_pending_execution",
            side_effect=persist_in_memory,
        ) as upsert, patch(
            "main.remove_pending_execution",
        ) as remove_pending, patch(
            "main.clear_confirmed_absent_entry_execution",
        ) as atomic_cleanup, patch("main.log_warning"), patch("main.log_error"):
            main.reconcile_pending_executions(state)

        fail_safe_close.assert_called_once()
        upsert.assert_called_once_with(state, SYMBOL, pending)
        remove_pending.assert_not_called()
        atomic_cleanup.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertTrue(pending["unsettled_exposure_close_confirmed"])
        self.assertTrue(pending["order_seen"])
        self.assertEqual(pending["max_executed_quantity_seen"], 0.4)


class InvalidPendingExecutionSymbolTests(unittest.TestCase):
    def setUp(self):
        self.state_temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.state_temp_dir.cleanup)
        state_path_patch = patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(self.state_temp_dir.name) / "open_trades_v7.json"),
        )
        state_path_patch.start()
        self.addCleanup(state_path_patch.stop)
        main.entry_quarantined_symbols.discard(SYMBOL)
        self.addCleanup(main.entry_quarantined_symbols.discard, SYMBOL)

    @staticmethod
    def state_with_pending_recovery():
        return {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "pending_dca": {
                        "level": 2,
                        "execution_unsettled": True,
                    },
                },
            },
            "pending_executions": {
                SYMBOL: pending_execution(
                    context="DCA_LEVEL_2",
                    dca_level=2,
                    pre_position_amount=1.0,
                ),
            },
        }

    def test_authoritatively_invalid_flat_symbol_clears_state_without_private_calls(self):
        state = self.state_with_pending_recovery()
        events = []

        def clear_reservation(current_state, symbol, level=None):
            events.append(("reservation", symbol, level))
            current_state["positions"][symbol].pop("pending_dca", None)
            return True

        def remove_pending(current_state, symbol):
            events.append(("pending", symbol))
            current_state["pending_executions"].pop(symbol, None)
            return True

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.get_open_position_details",
            return_value={},
        ) as all_positions, patch(
            "main.clear_dca_reservation",
            side_effect=clear_reservation,
        ), patch(
            "main.remove_pending_execution",
            side_effect=remove_pending,
        ), patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch(
            "main._pending_execution_live_detail",
        ) as symbol_position, patch(
            "main.cancel_open_protection_orders",
        ) as protection_cleanup, patch("main.log_warning"):
            main.reconcile_pending_executions(state)

        all_positions.assert_called_once_with(force=True)
        self.assertEqual(
            events,
            [("reservation", SYMBOL, 2), ("pending", SYMBOL)],
        )
        order_reconciliation.assert_not_called()
        symbol_position.assert_not_called()
        protection_cleanup.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)

    def test_current_all_position_snapshot_avoids_a_second_account_query(self):
        state = self.state_with_pending_recovery()

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.get_open_position_details",
        ) as all_positions, patch(
            "main.clear_dca_reservation",
            return_value=True,
        ), patch(
            "main.remove_pending_execution",
            side_effect=remove_in_memory,
        ), patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch("main.log_warning"):
            main.reconcile_pending_executions(state, position_details={})

        all_positions.assert_not_called()
        order_reconciliation.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])

    def test_invalid_flat_entry_clears_pending_and_submit_marker_atomically(self):
        state = {
            "positions": {SYMBOL: safe_entry_submission_marker()},
            "pending_executions": {SYMBOL: pending_execution()},
        }

        def atomic_clear(current_state, symbol, expected_ids):
            self.assertEqual(expected_ids, ("cid-ioc", "cid-market"))
            current_state["pending_executions"].pop(symbol, None)
            current_state["positions"].pop(symbol, None)
            return True, "CLEARED"

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.clear_confirmed_absent_entry_execution",
            side_effect=atomic_clear,
        ) as atomic_cleanup, patch(
            "main.remove_pending_execution",
        ) as remove_pending, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch(
            "main._pending_execution_live_detail",
        ) as symbol_position, patch("main.log_warning"):
            main.reconcile_pending_executions(state, position_details={})

        atomic_cleanup.assert_called_once()
        remove_pending.assert_not_called()
        order_reconciliation.assert_not_called()
        symbol_position.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])
        self.assertNotIn(SYMBOL, state["positions"])
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)

    def test_catalog_uncertainty_retains_and_quarantines_without_private_calls(self):
        state = self.state_with_pending_recovery()

        with patch(
            "main.is_known_futures_symbol",
            return_value=None,
        ), patch(
            "main.get_open_position_details",
        ) as all_positions, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch(
            "main._pending_execution_live_detail",
        ) as symbol_position, patch("main.log_warning"):
            main.reconcile_pending_executions(state)

        all_positions.assert_not_called()
        order_reconciliation.assert_not_called()
        symbol_position.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_invalid_symbol_snapshot_uncertainty_retains_marker(self):
        state = self.state_with_pending_recovery()

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.get_open_position_details",
            return_value=None,
        ), patch(
            "main.clear_dca_reservation",
        ) as clear_reservation, patch(
            "main.remove_pending_execution",
        ) as remove_pending, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch("main.log_error"):
            main.reconcile_pending_executions(state)

        clear_reservation.assert_not_called()
        remove_pending.assert_not_called()
        order_reconciliation.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_invalid_symbol_remove_failure_retains_and_quarantines_marker(self):
        state = self.state_with_pending_recovery()

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.clear_dca_reservation",
            return_value=True,
        ) as clear_reservation, patch(
            "main.remove_pending_execution",
            return_value=False,
        ) as remove_pending, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch("main.log_error"):
            main.reconcile_pending_executions(state, position_details={})

        clear_reservation.assert_called_once_with(state, SYMBOL, 2)
        remove_pending.assert_called_once_with(state, SYMBOL)
        order_reconciliation.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_invalid_symbol_missing_level_does_not_clear_ambiguous_reservation(self):
        state = self.state_with_pending_recovery()
        state["pending_executions"][SYMBOL]["dca_level"] = None

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.clear_dca_reservation",
        ) as clear_reservation, patch(
            "main.remove_pending_execution",
        ) as remove_pending, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch("main.log_error"):
            main.reconcile_pending_executions(state, position_details={})

        clear_reservation.assert_not_called()
        remove_pending.assert_not_called()
        order_reconciliation.assert_not_called()
        self.assertIn(SYMBOL, state["pending_executions"])
        self.assertIn(SYMBOL, main.entry_quarantined_symbols)

    def test_invalid_dca_context_without_reservation_can_clear_stale_marker(self):
        state = self.state_with_pending_recovery()
        state["positions"].pop(SYMBOL)
        state["pending_executions"][SYMBOL]["dca_level"] = None

        with patch(
            "main.is_known_futures_symbol",
            return_value=False,
        ), patch(
            "main.clear_dca_reservation",
        ) as clear_reservation, patch(
            "main.remove_pending_execution",
            side_effect=remove_in_memory,
        ) as remove_pending, patch(
            "main.reconcile_execution_client_orders",
        ) as order_reconciliation, patch("main.log_warning"):
            main.reconcile_pending_executions(state, position_details={})

        clear_reservation.assert_not_called()
        remove_pending.assert_called_once_with(state, SYMBOL)
        order_reconciliation.assert_not_called()
        self.assertNotIn(SYMBOL, state["pending_executions"])
        self.assertNotIn(SYMBOL, main.entry_quarantined_symbols)


class PendingExecutionStatePersistenceTests(unittest.TestCase):
    def test_pending_execution_upsert_and_remove_are_durable(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "open_trades_v7.json"
            state = {"positions": {}, "pending_executions": {}}
            pending = pending_execution()

            with patch.object(
                config,
                "DCA_STATE_PATH",
                str(state_path),
            ), patch.object(
                config,
                "STATE_UPSERT_RETRY_ATTEMPTS",
                1,
                create=True,
            ), patch.object(
                config,
                "STATE_UPSERT_RETRY_DELAY_SECONDS",
                0,
                create=True,
            ):
                self.assertTrue(
                    trade_state.upsert_pending_execution(
                        state,
                        SYMBOL,
                        pending,
                    )
                )
                persisted = trade_state.load_trade_state()
                self.assertEqual(
                    persisted["pending_executions"][SYMBOL][
                        "client_order_ids"
                    ],
                    "cid-ioc,cid-market",
                )

                self.assertTrue(
                    trade_state.remove_pending_execution(state, SYMBOL)
                )
                persisted = trade_state.load_trade_state()
                self.assertNotIn(SYMBOL, persisted["pending_executions"])

    def test_confirmed_absent_entry_clear_removes_both_records_atomically(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "open_trades_v7.json"
            state = {
                "positions": {SYMBOL: safe_entry_submission_marker()},
                "pending_executions": {SYMBOL: pending_execution()},
            }

            with patch.object(
                config,
                "DCA_STATE_PATH",
                str(state_path),
            ), patch.object(
                config,
                "STATE_UPSERT_RETRY_ATTEMPTS",
                1,
                create=True,
            ), patch.object(
                config,
                "STATE_UPSERT_RETRY_DELAY_SECONDS",
                0,
                create=True,
            ):
                self.assertTrue(trade_state.save_trade_state(state))
                cleared, reason = (
                    trade_state.clear_confirmed_absent_entry_execution(
                        state,
                        SYMBOL,
                        ("cid-ioc", "cid-market"),
                    )
                )

                self.assertTrue(cleared)
                self.assertEqual(reason, "CLEARED")
                persisted = trade_state.load_trade_state()
                self.assertNotIn(SYMBOL, persisted["pending_executions"])
                self.assertNotIn(SYMBOL, persisted["positions"])
                self.assertNotIn(SYMBOL, state["pending_executions"])
                self.assertNotIn(SYMBOL, state["positions"])

    def test_confirmed_absent_entry_clear_fails_closed_on_compare_or_marker(self):
        scenarios = (
            (
                "client_ids_changed",
                safe_entry_submission_marker(),
                ("different-client-id",),
                "PENDING_CLIENT_ORDER_IDS_CHANGED",
            ),
            (
                "unsafe_marker",
                {
                    **safe_entry_submission_marker(),
                    "initial_quantity": 1.0,
                },
                ("cid-ioc", "cid-market"),
                "POSITION_MARKER_NOT_SAFE_TO_CLEAR",
            ),
        )

        for label, marker, expected_ids, expected_reason in scenarios:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temp_dir:
                state_path = Path(temp_dir) / "open_trades_v7.json"
                state = {
                    "positions": {SYMBOL: marker},
                    "pending_executions": {SYMBOL: pending_execution()},
                }

                with patch.object(
                    config,
                    "DCA_STATE_PATH",
                    str(state_path),
                ), patch.object(
                    config,
                    "STATE_UPSERT_RETRY_ATTEMPTS",
                    1,
                    create=True,
                ), patch.object(
                    config,
                    "STATE_UPSERT_RETRY_DELAY_SECONDS",
                    0,
                    create=True,
                ):
                    self.assertTrue(trade_state.save_trade_state(state))
                    cleared, reason = (
                        trade_state.clear_confirmed_absent_entry_execution(
                            state,
                            SYMBOL,
                            expected_ids,
                        )
                    )

                    self.assertFalse(cleared)
                    self.assertEqual(reason, expected_reason)
                    persisted = trade_state.load_trade_state()
                    self.assertIn(SYMBOL, persisted["pending_executions"])
                    self.assertIn(SYMBOL, persisted["positions"])


if __name__ == "__main__":
    unittest.main()
