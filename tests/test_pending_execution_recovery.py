import tempfile
import unittest
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


def persist_in_memory(state, symbol, data):
    state.setdefault("pending_executions", {})[symbol] = data
    return True


def remove_in_memory(state, symbol):
    state.setdefault("pending_executions", {}).pop(symbol, None)
    return True


class PendingExecutionRecoveryTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
