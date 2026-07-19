import csv
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import ANY, call, patch

import config
import exchange
import execution_telemetry
import trade_state


def position(amount, entry_price=100, mark_price=100, position_side="BOTH"):
    return {
        "symbol": "BTCUSDT",
        "amount": amount,
        "quantity": abs(amount),
        "side": "BUY" if amount > 0 else "SELL",
        "position_side": position_side,
        "entry_price": entry_price,
        "mark_price": mark_price,
    }


class EntryReconciliationTests(unittest.TestCase):
    def setUp(self):
        smart_execution = patch.object(
            config,
            "SMART_EXECUTION_ENABLED",
            False,
            create=True,
        )
        smart_execution.start()
        self.addCleanup(smart_execution.stop)

    def common_patches(self, residual_retries=1):
        return (
            patch.object(config, "EXECUTION_RECONCILIATION_ENABLED", True),
            patch.object(
                config,
                "EXECUTION_RESIDUAL_RETRY_ATTEMPTS",
                residual_retries,
            ),
            patch.object(
                exchange,
                "normalize_order_quantity",
                side_effect=lambda _, qty: qty,
            ),
            patch.object(exchange, "append_execution_telemetry"),
        )

    def test_full_entry_uses_verified_position_and_records_slippage(self):
        patches = self.common_patches()

        with patches[0], patches[1], patches[2], patches[3] as journal, patch.object(
            exchange,
            "_submit_entry_market_order",
            return_value={
                "orderId": 11,
                "status": "FILLED",
                "executedQty": "1",
                "cumQuote": "101",
            },
        ), patch.object(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, position(1, entry_price=101), 1),
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
                context="ENTRY",
            )

        reconciliation = order["_execution_reconciliation"]
        self.assertTrue(reconciliation["fully_filled"])
        self.assertTrue(reconciliation["position_verified"])
        self.assertEqual(reconciliation["executed_quantity"], 1)
        self.assertEqual(float(order["avgPrice"]), 101)
        self.assertEqual(journal.call_args.args[0]["slippage_bps"], 100)

    def test_partial_entry_retries_only_verified_residual(self):
        patches = self.common_patches(residual_retries=1)
        first = {
            "orderId": 21,
            "status": "EXPIRED",
            "executedQty": "0.4",
            "cumQuote": "40",
        }
        second = {
            "orderId": 22,
            "status": "FILLED",
            "executedQty": "0.6",
            "cumQuote": "60.6",
        }

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            side_effect=[first, second],
        ) as submit, patch.object(
            exchange,
            "_wait_for_position_reconciliation",
            side_effect=[
                (True, position(0.4, entry_price=100), 1),
                (True, position(1, entry_price=100.6), 1),
            ],
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        self.assertEqual(
            submit.call_args_list,
            [
                call(
                    "BTCUSDT",
                    "BUY",
                    1,
                    client_order_id=ANY,
                ),
                call(
                    "BTCUSDT",
                    "BUY",
                    0.6,
                    client_order_id=ANY,
                ),
            ],
        )
        reconciliation = order["_execution_reconciliation"]
        self.assertTrue(reconciliation["fully_filled"])
        self.assertEqual(reconciliation["submission_attempts"], 2)
        self.assertAlmostEqual(float(order["avgPrice"]), 100.6)

    def test_ambiguous_submission_is_not_blindly_duplicated(self):
        patches = self.common_patches(residual_retries=2)

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            side_effect=TimeoutError("response lost"),
        ) as submit, patch.object(
            exchange,
            "_resolve_entry_order",
            return_value=(None, False, 4, ["order status unavailable"]),
        ), patch.object(
            exchange,
            "_cancel_unsettled_entry_order",
            return_value=(None, "cancel status unavailable"),
        ), patch.object(
            exchange,
            "_execution_position_detail",
            return_value=(True, None),
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        self.assertIsNotNone(order)
        self.assertEqual(
            order["_execution_reconciliation"]["status"],
            "PENDING",
        )
        self.assertFalse(
            order["_execution_reconciliation"]["order_terminal"]
        )
        submit.assert_called_once_with(
            "BTCUSDT",
            "BUY",
            1,
            client_order_id=ANY,
        )

    def test_nonterminal_order_response_is_not_retried(self):
        patches = self.common_patches(residual_retries=2)
        working_order = {
            "orderId": 23,
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.4",
            "cumQuote": "40",
        }

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            return_value=working_order,
        ) as submit, patch.object(
            exchange,
            "_resolve_entry_order",
            return_value=(working_order, False, 4, []),
        ), patch.object(
            exchange,
            "_cancel_unsettled_entry_order",
            return_value=(None, "cancel status unavailable"),
        ), patch.object(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, position(0.4), 1),
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        self.assertEqual(submit.call_count, 1)
        self.assertEqual(
            order["_execution_reconciliation"]["executed_quantity"],
            0.4,
        )
        self.assertEqual(
            order["_execution_reconciliation"][
                "observed_position_increase_quantity"
            ],
            0.4,
        )

    def test_terminal_cancel_allows_safe_residual_retry(self):
        patches = self.common_patches(residual_retries=1)
        working_order = {
            "orderId": 26,
            "status": "PARTIALLY_FILLED",
            "executedQty": "0.4",
            "cumQuote": "40",
        }
        canceled_order = {**working_order, "status": "CANCELED"}
        residual_order = {
            "orderId": 27,
            "status": "FILLED",
            "executedQty": "0.6",
            "cumQuote": "60.6",
        }

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            side_effect=[working_order, residual_order],
        ) as submit, patch.object(
            exchange,
            "_resolve_entry_order",
            side_effect=[
                (working_order, False, 1, []),
                (canceled_order, True, 1, []),
                (residual_order, True, 0, []),
            ],
        ), patch.object(
            exchange,
            "_cancel_unsettled_entry_order",
            return_value=(canceled_order, ""),
        ) as cancel, patch.object(
            exchange,
            "_wait_for_position_reconciliation",
            side_effect=[
                (True, position(0.4), 1),
                (True, position(1), 1),
            ],
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        self.assertEqual(submit.call_count, 2)
        cancel.assert_called_once()
        self.assertTrue(order["_execution_reconciliation"]["fully_filled"])
        self.assertAlmostEqual(float(order["avgPrice"]), 100.6)

    def test_ambiguous_partial_position_delta_is_not_retried(self):
        patches = self.common_patches(residual_retries=2)

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            side_effect=TimeoutError("response lost"),
        ) as submit, patch.object(
            exchange,
            "_resolve_entry_order",
            return_value=(None, False, 4, ["order status unavailable"]),
        ), patch.object(
            exchange,
            "_cancel_unsettled_entry_order",
            return_value=(None, "cancel status unavailable"),
        ), patch.object(
            exchange,
            "_execution_position_detail",
            return_value=(True, position(0.4)),
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        self.assertEqual(submit.call_count, 1)
        self.assertEqual(
            order["_execution_reconciliation"]["executed_quantity"],
            0,
        )
        self.assertEqual(
            order["_execution_reconciliation"][
                "observed_position_increase_quantity"
            ],
            0.4,
        )

    def test_stale_snapshot_is_not_labeled_position_verified(self):
        patches = self.common_patches(residual_retries=0)
        filled_order = {
            "orderId": 24,
            "status": "FILLED",
            "executedQty": "1",
            "cumQuote": "100",
        }

        with patches[0], patches[1], patches[2], patches[3], patch.object(
            exchange,
            "_submit_entry_market_order",
            return_value=filled_order,
        ), patch.object(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(False, position(0), 4),
        ):
            order = exchange.place_market_order(
                "BTCUSDT",
                "BUY",
                1,
                pre_position_amount=0,
                reference_price=100,
            )

        reconciliation = order["_execution_reconciliation"]
        self.assertTrue(reconciliation["fully_filled"])
        self.assertFalse(reconciliation["position_verified"])

    def test_exact_client_order_lookup_resolves_ambiguous_submission(self):
        resolved = {
            "orderId": 25,
            "clientOrderId": "v7-test-order",
            "status": "FILLED",
            "executedQty": "1",
            "cumQuote": "100",
        }

        with patch.object(
            config,
            "EXECUTION_VERIFY_ATTEMPTS",
            2,
        ), patch.object(
            config,
            "EXECUTION_VERIFY_DELAY_SECONDS",
            0,
        ), patch.object(
            exchange,
            "_private_rest_call",
            return_value=resolved,
        ) as private_call:
            order, terminal, attempts, errors = exchange._resolve_entry_order(
                "BTCUSDT",
                "v7-test-order",
            )

        self.assertEqual(order, resolved)
        self.assertTrue(terminal)
        self.assertEqual(attempts, 1)
        self.assertEqual(errors, [])
        self.assertEqual(
            private_call.call_args.kwargs["origClientOrderId"],
            "v7-test-order",
        )

    def test_explicit_zero_execution_does_not_use_requested_fallback(self):
        self.assertEqual(
            exchange.get_reconciled_executed_quantity(
                {"status": "EXPIRED", "executedQty": "0"},
                fallback=1,
            ),
            0,
        )


class CloseReconciliationTests(unittest.TestCase):
    def setUp(self):
        quantity_normalizer = patch.object(
            exchange,
            "normalize_order_quantity",
            side_effect=lambda _symbol, quantity, **_kwargs: abs(
                float(quantity)
            ),
        )
        quantity_normalizer.start()
        self.addCleanup(quantity_normalizer.stop)

    def test_partial_close_retries_residual_and_requires_flat_position(self):
        with patch.object(
            config,
            "EXECUTION_RECONCILIATION_ENABLED",
            True,
        ), patch.object(
            config,
            "EXECUTION_RESIDUAL_RETRY_ATTEMPTS",
            1,
        ), patch.object(
            exchange,
            "_close_position_snapshot",
            return_value=(
                True,
                position(1, position_side="LONG"),
                "",
            ),
        ), patch.object(
            exchange,
            "_submit_position_close_once",
            side_effect=[
                (
                    {
                        "orderId": 31,
                        "status": "FILLED",
                        "executedQty": "0.4",
                        "cumQuote": "40",
                    },
                    "SELL",
                    "LONG",
                ),
                (
                    {
                        "orderId": 32,
                        "status": "FILLED",
                        "executedQty": "0.6",
                        "cumQuote": "59.4",
                    },
                    "SELL",
                    "LONG",
                ),
            ],
        ) as submit, patch.object(
            exchange,
            "_wait_for_close_position_reconciliation",
            side_effect=[
                (True, position(0.6, position_side="LONG"), 1, ""),
                (True, None, 1, ""),
            ],
        ), patch.object(exchange, "append_execution_telemetry"):
            order = exchange.close_position_market(
                "BTCUSDT",
                1,
                position_side="LONG",
                reference_price=100,
                context="TEST_EXIT",
            )

        self.assertTrue(order["_execution_reconciliation"]["position_closed"])
        self.assertEqual(submit.call_count, 2)
        self.assertEqual(submit.call_args_list[1].args[1], 0.6)

    def test_unconfirmed_residual_returns_failure_to_keep_protection(self):
        with patch.object(
            config,
            "EXECUTION_RECONCILIATION_ENABLED",
            True,
        ), patch.object(
            config,
            "EXECUTION_RESIDUAL_RETRY_ATTEMPTS",
            0,
        ), patch.object(
            exchange,
            "_close_position_snapshot",
            return_value=(True, position(1), ""),
        ), patch.object(
            exchange,
            "_submit_position_close_once",
            return_value=(
                {
                    "orderId": 41,
                    "status": "FILLED",
                    "executedQty": "0.4",
                    "cumQuote": "40",
                },
                "SELL",
                "BOTH",
            ),
        ), patch.object(
            exchange,
            "_wait_for_close_position_reconciliation",
            return_value=(True, position(0.6), 1, ""),
        ), patch.object(exchange, "append_execution_telemetry") as journal:
            order = exchange.close_position_market(
                "BTCUSDT",
                1,
                reference_price=100,
            )

        self.assertIsNone(order)
        self.assertEqual(journal.call_args.args[0]["status"], "RESIDUAL_OPEN")


class ExecutionTelemetryTests(unittest.TestCase):
    def test_mixed_order_price_sources_use_only_matched_quantities(self):
        aggregate = execution_telemetry.aggregate_order_execution([
            {
                "executedQty": "0.4",
                "cumQuote": "40",
                "avgPrice": "100",
            },
            {
                "executedQty": "0.6",
                "avgPrice": "101",
            },
        ])

        self.assertAlmostEqual(aggregate["average_fill_price"], 100.6)
        self.assertIsNone(aggregate["commission"])

    def test_csv_journal_and_sell_slippage_direction(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.csv"

            with patch.object(config, "EXECUTION_TELEMETRY_ENABLED", True), patch.object(
                config,
                "EXECUTION_TELEMETRY_PATH",
                str(path),
            ):
                written = execution_telemetry.append_execution_telemetry({
                    "context": "ENTRY",
                    "symbol": "BTCUSDT",
                    "order_side": "SELL",
                    "reference_price": 100,
                    "average_fill_price": 99,
                    "slippage_bps": execution_telemetry.calculate_slippage_bps(
                        "SELL",
                        100,
                        99,
                    ),
                })
                execution_telemetry.flush_execution_telemetry()

            self.assertTrue(written)

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(len(rows), 1)
            self.assertEqual(float(rows[0]["slippage_bps"]), 100)


class TradeStatePersistenceTests(unittest.TestCase):
    def test_upsert_retries_and_reports_success(self):
        state = {"positions": {}}

        with patch.object(
            config,
            "STATE_UPSERT_RETRY_ATTEMPTS",
            2,
        ), patch.object(
            config,
            "STATE_UPSERT_RETRY_DELAY_SECONDS",
            0,
        ), patch.object(
            trade_state,
            "_state_file_lock",
            return_value=nullcontext(),
        ), patch.object(
            trade_state,
            "_load_trade_state_unlocked",
            return_value={"positions": {}},
        ), patch.object(
            trade_state,
            "_save_trade_state_unlocked",
            side_effect=[OSError("busy"), None],
        ) as save:
            saved = trade_state.upsert_position_state(
                state,
                "BTCUSDT",
                {"managed_by_bot": True},
            )

        self.assertTrue(saved)
        self.assertEqual(save.call_count, 2)
        self.assertIn("BTCUSDT", state["positions"])

    def test_upsert_reports_terminal_failure(self):
        with patch.object(
            config,
            "STATE_UPSERT_RETRY_ATTEMPTS",
            2,
        ), patch.object(
            config,
            "STATE_UPSERT_RETRY_DELAY_SECONDS",
            0,
        ), patch.object(
            trade_state,
            "_state_file_lock",
            return_value=nullcontext(),
        ), patch.object(
            trade_state,
            "_load_trade_state_unlocked",
            return_value={"positions": {}},
        ), patch.object(
            trade_state,
            "_save_trade_state_unlocked",
            side_effect=OSError("disk unavailable"),
        ):
            saved = trade_state.upsert_position_state(
                {"positions": {}},
                "BTCUSDT",
                {"managed_by_bot": True},
            )

        self.assertFalse(saved)


if __name__ == "__main__":
    unittest.main()
