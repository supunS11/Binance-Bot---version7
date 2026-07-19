import unittest
from unittest.mock import patch

import config
from multi_tp import RUNNER_ACTIVE, RUNNER_PENDING, TP1_PENDING


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import exchange
    import main


SYMBOL = "BTCUSDT"


def apply_updates(state, symbol, updates):
    state["positions"][symbol].update(updates)
    return True


def new_monitor():
    with patch(
        "main.load_trade_state",
        return_value={"positions": {}, "pending_executions": {}},
    ):
        return main.DcaWebsocketMonitor()


def runner_position(**updates):
    position = {
        "managed_by_bot": True,
        "multi_tp_active": True,
        "multi_tp_stage": RUNNER_PENDING,
        "side": "BUY",
        "avg_entry": 100,
        "runner_basis_price": 105,
        "runner_tp_price": 108,
        "runner_tp_mode": "TP2_STRUCTURE_TEST",
        "runner_tp_context": {},
        "runner_sl_price": 101,
        "runner_sl_mode": "RUNNER_STRUCTURE",
        "runner_sl_order_id": "",
        "runner_tp_order_id": "",
        "initial_sl_order_id": 11,
        "sl_enabled": True,
    }
    position.update(updates)
    return position


class MultiTpExchangeHardeningTests(unittest.TestCase):
    def test_partial_tp_rejects_material_lot_rounding_deviation(self):
        with patch.object(
            exchange,
            "normalize_order_quantity",
            side_effect=[10.0, 5.0, 5.0],
        ), patch.object(
            config,
            "TP1_MAX_CLOSE_PCT_DEVIATION",
            5,
            create=True,
        ), patch.object(exchange, "place_algo_order") as place:
            order, quantity = exchange.place_partial_take_profit(
                SYMBOL,
                exchange.SIDE_BUY,
                10,
                75,
                105,
            )

        self.assertIsNone(order)
        self.assertEqual(quantity, 0)
        place.assert_not_called()

    def test_triggered_child_query_failure_is_ambiguous(self):
        algo_order = {
            "_query_ok": True,
            "_found": True,
            "algoId": 77,
            "algoStatus": "FINISHED",
            "actualOrderId": 88,
        }

        with patch.object(
            exchange,
            "get_algo_order_info",
            return_value=algo_order,
        ), patch.object(
            exchange,
            "_private_rest_call",
            side_effect=RuntimeError("child temporarily unavailable"),
        ):
            result = exchange.get_algo_order_execution(SYMBOL, 77)

        self.assertFalse(result["query_ok"])
        self.assertFalse(result["child_query_ok"])
        self.assertTrue(result["ambiguous"])
        self.assertFalse(result["open"])
        self.assertFalse(result["terminal"])

    def test_partially_filled_triggered_child_remains_open(self):
        algo_order = {
            "_query_ok": True,
            "_found": True,
            "algoId": 77,
            "algoStatus": "FINISHED",
            "actualOrderId": 88,
        }

        with patch.object(
            exchange,
            "get_algo_order_info",
            return_value=algo_order,
        ), patch.object(
            exchange,
            "_private_rest_call",
            return_value={
                "orderId": 88,
                "status": "PARTIALLY_FILLED",
                "executedQty": "0.25",
            },
        ):
            result = exchange.get_algo_order_execution(SYMBOL, 77)

        self.assertTrue(result["query_ok"])
        self.assertTrue(result["open"])
        self.assertFalse(result["terminal"])
        self.assertEqual(result["executed_quantity"], 0.25)


class MultiTpMonitorHardeningTests(unittest.TestCase):
    @staticmethod
    def _repair_state(live_quantity=1.0):
        return {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                    "multi_tp_stage": TP1_PENDING,
                    "side": "BUY",
                    "avg_entry": 100,
                    "tp1_price": 105,
                    "tp1_original_price": 105,
                    "tp1_base_quantity": 1,
                    "tp1_quantity": 0.75,
                    "tp1_order_quantity": 0.75,
                    "tp1_order_id": 77,
                    "tp1_accounted_order_ids": [],
                    "tp1_executed_quantity": 0,
                    "tp1_executed_quote": 0,
                    "tp1_repair_count": 0,
                    "live_quantity": live_quantity,
                }
            }
        }

    @staticmethod
    def _terminal_tp1(executed_quantity=0):
        return {
            "query_ok": True,
            "found": True,
            "open": False,
            "terminal": True,
            "filled": False,
            "executed_quantity": executed_quantity,
            "algo_status": "CANCELED",
            "algo_order": {
                "orderType": "TAKE_PROFIT_MARKET",
                "side": "SELL",
                "positionSide": "BOTH",
                "closePosition": False,
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
                "quantity": "0.75",
                "triggerPrice": "105",
            },
            "actual_order": {
                "status": "CANCELED",
                "executedQty": str(executed_quantity),
                "avgPrice": "105",
            },
        }

    def test_confirmed_exact_tp1_fill_moves_state_to_runner_pending(self):
        monitor = new_monitor()
        state = {
            "positions": {
                SYMBOL: {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                    "multi_tp_stage": TP1_PENDING,
                    "side": "BUY",
                    "avg_entry": 100,
                    "tp1_price": 105,
                    "tp1_base_quantity": 1,
                    "tp1_quantity": 0.5,
                    "tp1_order_quantity": 0.5,
                    "tp1_order_id": 77,
                    "tp1_accounted_order_ids": [],
                    "tp1_executed_quantity": 0,
                    "tp1_executed_quote": 0,
                }
            }
        }
        monitor.synced_position_details = {SYMBOL: {"quantity": 0.5}}

        execution = {
            "query_ok": True,
            "found": True,
            "filled": True,
            "triggered": True,
            "terminal": True,
            "open": False,
            "executed_quantity": 0.5,
            "actual_status": "FILLED",
            "algo_status": "FINISHED",
            "actual_order": {"status": "FILLED", "avgPrice": "105"},
            "algo_order": {
                "orderType": "TAKE_PROFIT_MARKET",
                "side": "SELL",
                "positionSide": "BOTH",
                "closePosition": False,
                "reduceOnly": True,
                "workingType": "MARK_PRICE",
                "quantity": "0.5",
                "triggerPrice": "105",
                "actualPrice": "105",
            },
        }

        with patch("main.load_trade_state", return_value=state), patch(
            "main.get_open_position_details",
            return_value={
                SYMBOL: {
                    "quantity": 0.5,
                    "amount": 0.5,
                    "mark_price": 105,
                    "position_side": "BOTH",
                }
            },
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ), patch(
            "main.cancel_algo_order",
            return_value=True,
        ), patch(
            "main.get_algo_order_execution",
            return_value=execution,
        ), patch.object(
            monitor,
            "_configure_multi_tp_runner",
            return_value=True,
        ) as configure:
            handled = monitor._handle_multi_tp_runner(SYMBOL, 105, state)

        self.assertTrue(handled)
        self.assertEqual(
            state["positions"][SYMBOL]["multi_tp_stage"],
            RUNNER_PENDING,
        )
        self.assertEqual(
            state["positions"][SYMBOL]["tp1_executed_quantity"],
            0.5,
        )
        configure.assert_called_once()

    def test_runner_places_stop_before_tp2_and_then_becomes_active(self):
        monitor = new_monitor()
        position = runner_position()
        state = {"positions": {SYMBOL: position}}

        with patch(
            "main.normalize_trigger_price",
            side_effect=lambda symbol, side, order_type, price: float(price),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": True},
        ), patch(
            "main.place_close_position_protection",
            side_effect=[{"algoId": 22}, {"algoId": 33}],
        ) as place, patch(
            "main.cancel_algo_order",
            return_value=True,
        ) as cancel, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ), patch(
            "main.send_telegram_message",
        ), patch.object(
            monitor,
            "_validate_runner_order_id",
            return_value="OPEN",
        ):
            handled = monitor._configure_multi_tp_runner(
                SYMBOL,
                105,
                state,
                position,
                {"quantity": 0.5, "position_side": "BOTH"},
            )

        self.assertTrue(handled)
        self.assertEqual(place.call_args_list[0].args[2], "STOP_MARKET")
        self.assertEqual(
            place.call_args_list[1].args[2],
            "TAKE_PROFIT_MARKET",
        )
        self.assertEqual(position["multi_tp_stage"], RUNNER_ACTIVE)
        cancel.assert_called_once_with(SYMBOL, 11)

    def test_runner_sl_lookup_failure_never_submits_duplicate(self):
        monitor = new_monitor()
        position = runner_position()
        state = {"positions": {SYMBOL: position}}

        with patch(
            "main.normalize_trigger_price",
            side_effect=lambda symbol, side, order_type, price: float(price),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": False, "error": "temporary failure"},
        ), patch(
            "main.place_close_position_protection",
        ) as place, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ):
            handled = monitor._configure_multi_tp_runner(
                SYMBOL,
                105,
                state,
                position,
                {"quantity": 0.5, "position_side": "BOTH"},
            )

        self.assertFalse(handled)
        place.assert_not_called()
        self.assertEqual(
            position["runner_protection_error"],
            "RUNNER_SL_OPEN_ORDER_QUERY_UNAVAILABLE",
        )

    def test_runner_tp_lookup_failure_never_submits_duplicate(self):
        monitor = new_monitor()
        position = runner_position(runner_sl_order_id=22)
        state = {"positions": {SYMBOL: position}}

        with patch(
            "main.normalize_trigger_price",
            side_effect=lambda symbol, side, order_type, price: float(price),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": False, "error": "temporary failure"},
        ), patch(
            "main.place_close_position_protection",
        ) as place, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ), patch.object(
            monitor,
            "_validate_runner_order_id",
            return_value="OPEN",
        ):
            handled = monitor._configure_multi_tp_runner(
                SYMBOL,
                105,
                state,
                position,
                {"quantity": 0.5, "position_side": "BOTH"},
            )

        self.assertFalse(handled)
        place.assert_not_called()
        self.assertEqual(
            position["runner_protection_error"],
            "RUNNER_TP_OPEN_ORDER_QUERY_UNAVAILABLE",
        )

    def test_dca_tick_is_blocked_after_tp1_trigger_or_fill(self):
        for stage, trigger_seen_at in (
            (TP1_PENDING, "2026-07-19T12:00:00"),
            (RUNNER_PENDING, None),
            (RUNNER_ACTIVE, None),
        ):
            state = {
                "positions": {
                    SYMBOL: {
                        "managed_by_bot": True,
                        "multi_tp_stage": stage,
                        "tp1_trigger_seen_at": trigger_seen_at,
                        "side": "BUY",
                    }
                }
            }

            with self.subTest(stage=stage), patch.object(
                config,
                "TP1_RUNNER_DISABLE_DCA",
                True,
            ):
                self.assertFalse(main.dca_tick_ready(SYMBOL, 90, state=state))

    def test_terminal_unfilled_tp1_is_replaced_for_exact_target(self):
        monitor = new_monitor()
        state = self._repair_state()

        with patch(
            "main.get_algo_order_execution",
            return_value=self._terminal_tp1(),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": True},
        ), patch(
            "main.place_partial_take_profit_quantity",
            return_value=({"algoId": 88}, 0.75),
        ) as place, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ):
            confirmed, _ = monitor._resolve_exact_tp1_fill(
                SYMBOL,
                state,
                state["positions"][SYMBOL],
                {
                    "quantity": 1.0,
                    "mark_price": 100,
                    "position_side": "BOTH",
                },
            )

        self.assertFalse(confirmed)
        position = state["positions"][SYMBOL]
        self.assertEqual(position["tp1_order_id"], 88)
        self.assertEqual(position["tp1_order_quantity"], 0.75)
        self.assertEqual(position["tp1_repair_count"], 1)
        place.assert_called_once()

    def test_terminal_unfilled_tp1_rearms_ahead_of_current_market(self):
        monitor = new_monitor()
        state = self._repair_state()

        with patch(
            "main.get_algo_order_execution",
            return_value=self._terminal_tp1(),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": True},
        ), patch(
            "main.get_symbol_price_rules",
            return_value={"tick_size": "0.1"},
        ), patch(
            "main.normalize_trigger_price",
            side_effect=lambda symbol, side, order_type, price: float(price),
        ), patch(
            "main.place_partial_take_profit_quantity",
            return_value=({"algoId": 88}, 0.75),
        ) as place, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ):
            monitor._resolve_exact_tp1_fill(
                SYMBOL,
                state,
                state["positions"][SYMBOL],
                {
                    "quantity": 1.0,
                    "mark_price": 106,
                    "position_side": "BOTH",
                },
            )

        repaired_trigger = place.call_args.args[3]
        self.assertGreater(repaired_trigger, 106)
        self.assertEqual(
            state["positions"][SYMBOL]["tp1_price"],
            repaired_trigger,
        )

    def test_terminal_partial_tp1_repairs_only_exact_residual(self):
        monitor = new_monitor()
        state = self._repair_state(live_quantity=0.7)

        with patch(
            "main.get_algo_order_execution",
            return_value=self._terminal_tp1(0.3),
        ), patch(
            "main.find_matching_open_algo_order",
            return_value={"query_ok": True},
        ), patch(
            "main.place_partial_take_profit_quantity",
            return_value=({"algoId": 88}, 0.45),
        ) as place, patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ):
            confirmed, _ = monitor._resolve_exact_tp1_fill(
                SYMBOL,
                state,
                state["positions"][SYMBOL],
                {
                    "quantity": 0.7,
                    "mark_price": 100,
                    "position_side": "BOTH",
                },
            )

        self.assertFalse(confirmed)
        position = state["positions"][SYMBOL]
        self.assertAlmostEqual(position["tp1_executed_quantity"], 0.3)
        self.assertAlmostEqual(position["tp1_executed_quote"], 31.5)
        self.assertAlmostEqual(position["tp1_order_quantity"], 0.45)
        self.assertAlmostEqual(place.call_args.args[2], 0.45)

    def test_tp1_health_is_checked_before_price_reaches_trigger(self):
        monitor = new_monitor()
        state = self._repair_state()
        monitor.synced_position_details = {SYMBOL: {"quantity": 1.0}}

        with patch.object(
            monitor,
            "_multi_tp_retry_ready",
            return_value=True,
        ), patch("main.load_trade_state", return_value=state), patch(
            "main.get_open_position_details",
            return_value={
                SYMBOL: {
                    "quantity": 1.0,
                    "amount": 1.0,
                    "mark_price": 100,
                    "position_side": "BOTH",
                }
            },
        ), patch.object(
            monitor,
            "_resolve_exact_tp1_fill",
            return_value=(False, {"open": True}),
        ) as resolve:
            handled = monitor._handle_multi_tp_runner(SYMBOL, 100, state)

        self.assertFalse(handled)
        resolve.assert_called_once()

    def test_runner_child_query_ambiguity_never_clears_order_id(self):
        monitor = new_monitor()

        with patch(
            "main.get_algo_order_execution",
            return_value={
                "query_ok": False,
                "ambiguous": True,
                "found": True,
            },
        ):
            status = monitor._validate_runner_order_id(
                SYMBOL,
                88,
                "TAKE_PROFIT_MARKET",
                "SELL",
                "BOTH",
                trigger_price=108,
            )

        self.assertEqual(status, "UNAVAILABLE")

    def test_scan_reconciliation_invokes_runner_recovery_after_restart(self):
        monitor = new_monitor()
        state = self._repair_state()

        with patch.object(
            monitor,
            "_handle_multi_tp_runner",
            return_value=False,
        ) as handle:
            monitor.reconcile_multi_tp_positions(
                {SYMBOL: {"mark_price": 100, "quantity": 1}},
                state,
            )

        handle.assert_called_once_with(SYMBOL, 100.0, state)


if __name__ == "__main__":
    unittest.main()
