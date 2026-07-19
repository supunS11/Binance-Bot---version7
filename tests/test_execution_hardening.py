import csv
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import ANY, patch

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import exchange
    import execution_telemetry


SYMBOL = "TESTUSDT"


def execution_order(
    status,
    executed_quantity,
    *,
    average_price=100,
    order_id="ioc-1",
    client_order_id="cid-ioc",
):
    return {
        "status": status,
        "executedQty": str(executed_quantity),
        "avgPrice": str(average_price if executed_quantity else 0),
        "cumQuote": str(float(executed_quantity) * float(average_price)),
        "orderId": order_id,
        "clientOrderId": client_order_id,
    }


def reconciled_market_order(
    executed_quantity,
    *,
    requested_quantity=None,
    terminal=True,
    position_verified=True,
    average_price=100.1,
):
    requested_quantity = (
        executed_quantity
        if requested_quantity is None
        else requested_quantity
    )
    return {
        "orderId": "market-1",
        "status": "FILLED" if terminal else "NEW",
        "_execution_reconciliation": {
            "executed_quantity": executed_quantity,
            "requested_quantity": requested_quantity,
            "residual_quantity": max(
                float(requested_quantity) - float(executed_quantity),
                0,
            ),
            "average_fill_price": average_price,
            "order_terminal": terminal,
            "position_verified": position_verified,
            "post_position_amount": executed_quantity,
            "submission_attempts": 1,
            "verification_attempts": 1,
            "order_ids": "market-1",
            "client_order_ids": "cid-market",
            "status": "FILLED" if terminal else "PENDING",
            "error": "",
        },
    }


class PositionModeVerificationTests(unittest.TestCase):
    def test_one_way_mode_requires_an_explicit_false_value(self):
        responses = (
            ({"dualSidePosition": False}, True),
            ({"dualSidePosition": "false"}, True),
            ({"dualSidePosition": True}, False),
            ({"dualSidePosition": "true"}, False),
            ({}, False),
            ({"dualSidePosition": "unknown"}, False),
        )

        for response, expected in responses:
            with self.subTest(response=response), patch.object(
                exchange,
                "_private_rest_call",
                return_value=response,
            ):
                self.assertIs(exchange.is_one_way_position_mode(), expected)

    def test_position_mode_query_failure_is_not_treated_as_one_way(self):
        with patch.object(
            exchange,
            "_private_rest_call",
            side_effect=RuntimeError("offline"),
        ):
            self.assertFalse(exchange.is_one_way_position_mode())


class OfflineExecutionCase(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self._set_config("EXECUTION_RECONCILIATION_ENABLED", True)
        self._set_config("EXECUTION_VERIFY_ATTEMPTS", 1)
        self._set_config("EXECUTION_VERIFY_DELAY_SECONDS", 0)
        self._set_config("EXECUTION_RESIDUAL_RETRY_ATTEMPTS", 0)
        self._set_config("SMART_EXECUTION_ENABLED", True)
        self._set_config("SMART_EXECUTION_CONTEXTS", {"ENTRY", "DCA"})
        self._set_config("SMART_EXECUTION_TIME_IN_FORCE", "IOC")
        self._set_config("SMART_EXECUTION_MAX_CROSS_BPS", 2.0)
        self._set_config("SMART_EXECUTION_MARKET_FALLBACK_ENABLED", True)
        self.telemetry = self._patch(
            exchange,
            "append_execution_telemetry",
        )
        self._patch(exchange, "log_info")
        self._patch(exchange, "log_warning")
        self._patch(exchange, "log_error")
        self._patch(exchange.time, "sleep")

    def _set_config(self, name, value):
        return self.stack.enter_context(
            patch.object(config, name, value, create=True)
        )

    def _patch(self, owner, name, **kwargs):
        return self.stack.enter_context(patch.object(owner, name, **kwargs))

    def _smart_harness(
        self,
        submitted_order,
        position_detail,
        *,
        fallback_result=None,
    ):
        self._patch(
            exchange,
            "get_book_ticker",
            return_value={
                "bid": 99.99,
                "ask": 100.0,
                "mid": 99.995,
                "spread_bps": 1.0,
            },
        )
        self._patch(exchange, "normalize_order_price", return_value=100.01)
        self._patch(
            exchange,
            "normalize_order_quantity",
            side_effect=lambda _symbol, quantity, **_kwargs: abs(float(quantity)),
        )
        self._patch(
            exchange,
            "_new_execution_client_order_id",
            return_value="cid-ioc",
        )
        submit = self._patch(
            exchange,
            "_submit_entry_ioc_limit_order",
            return_value=submitted_order,
        )
        self._patch(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, position_detail, 1),
        )
        fallback = self._patch(
            exchange,
            "_place_reconciled_market_order",
            return_value=fallback_result,
        )
        return submit, fallback


class SmartIocReconciliationTests(OfflineExecutionCase):
    def test_full_ioc_fill_never_uses_market_fallback(self):
        submit, fallback = self._smart_harness(
            execution_order("FILLED", 1),
            {"amount": 1, "entry_price": 100},
        )

        result = exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            reference_price=100,
            context="ENTRY",
        )
        reconciliation = exchange.get_execution_reconciliation(result)

        submit.assert_called_once()
        fallback.assert_not_called()
        self.assertEqual(reconciliation["executed_quantity"], 1)
        self.assertEqual(reconciliation["residual_quantity"], 0)
        self.assertTrue(reconciliation["fully_filled"])
        self.assertTrue(reconciliation["order_terminal"])
        self.assertFalse(reconciliation["fallback_used"])

    def test_partial_terminal_ioc_falls_back_for_only_the_residual(self):
        fallback_result = reconciled_market_order(
            0.6,
            requested_quantity=0.6,
        )
        _, fallback = self._smart_harness(
            execution_order("EXPIRED", 0.4),
            {"amount": 0.4, "entry_price": 100},
            fallback_result=fallback_result,
        )

        result = exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            reference_price=100,
            context="ENTRY",
        )
        reconciliation = exchange.get_execution_reconciliation(result)

        fallback.assert_called_once()
        self.assertAlmostEqual(fallback.call_args.args[2], 0.6)
        self.assertEqual(reconciliation["fallback_quantity"], 0.6)
        self.assertEqual(reconciliation["executed_quantity"], 1)
        self.assertTrue(reconciliation["fully_filled"])
        self.assertTrue(reconciliation["fallback_used"])

    def test_zero_fill_terminal_ioc_falls_back_for_the_full_quantity(self):
        fallback_result = reconciled_market_order(1)
        _, fallback = self._smart_harness(
            execution_order("EXPIRED", 0),
            None,
            fallback_result=fallback_result,
        )

        result = exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            reference_price=100,
            context="ENTRY",
        )
        reconciliation = exchange.get_execution_reconciliation(result)

        fallback.assert_called_once()
        self.assertEqual(fallback.call_args.args[2], 1)
        self.assertEqual(reconciliation["executed_quantity"], 1)
        self.assertTrue(reconciliation["fallback_used"])

    def test_nonterminal_ioc_never_submits_duplicate_market_fallback(self):
        self._patch(
            exchange,
            "get_book_ticker",
            return_value={
                "bid": 99.99,
                "ask": 100,
                "mid": 99.995,
                "spread_bps": 1,
            },
        )
        self._patch(exchange, "normalize_order_price", return_value=100.01)
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        self._patch(
            exchange,
            "_new_execution_client_order_id",
            return_value="cid-ioc",
        )
        unresolved = execution_order("NEW", 0)
        self._patch(
            exchange,
            "_submit_entry_ioc_limit_order",
            return_value=unresolved,
        )
        resolve = self._patch(
            exchange,
            "_resolve_entry_order",
            side_effect=[
                (unresolved, False, 1, ["query timeout"]),
                (unresolved, False, 1, ["still ambiguous"]),
            ],
        )
        cancel = self._patch(
            exchange,
            "_cancel_unsettled_entry_order",
            return_value=(None, "cancel timeout"),
        )
        self._patch(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(False, None, 1),
        )
        fallback = self._patch(exchange, "_place_reconciled_market_order")

        result = exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            reference_price=100,
            context="ENTRY",
        )
        reconciliation = exchange.get_execution_reconciliation(result)

        self.assertEqual(resolve.call_count, 2)
        cancel.assert_called_once()
        fallback.assert_not_called()
        self.assertFalse(reconciliation["order_terminal"])
        self.assertEqual(reconciliation["status"], "PENDING")
        self.assertEqual(reconciliation["executed_quantity"], 0)

    def test_disabling_reconciliation_bypasses_smart_execution(self):
        self._set_config("EXECUTION_RECONCILIATION_ENABLED", False)
        self._patch(
            exchange,
            "get_book_ticker",
            return_value={
                "bid": 99.99,
                "ask": 100,
                "mid": 99.995,
                "spread_bps": 1,
            },
        )
        self._patch(exchange, "normalize_order_price", return_value=100.01)
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        ioc_submit = self._patch(
            exchange,
            "_submit_entry_ioc_limit_order",
            return_value=execution_order("FILLED", 1),
        )
        self._patch(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, {"amount": 1, "entry_price": 100}, 1),
        )
        market_submit = self._patch(
            exchange,
            "_submit_entry_market_order",
            return_value=execution_order(
                "FILLED",
                1,
                order_id="market-1",
                client_order_id="cid-market",
            ),
        )

        result = exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            context="ENTRY",
        )

        market_submit.assert_called_once()
        ioc_submit.assert_not_called()
        self.assertEqual(result["orderId"], "market-1")


class ExchangeRuleTests(OfflineExecutionCase):
    @staticmethod
    def _exchange_info():
        return {
            "symbols": [{
                "symbol": SYMBOL,
                "pricePrecision": 2,
                "quantityPrecision": 3,
                "filters": [
                    {
                        "filterType": "PRICE_FILTER",
                        "tickSize": "1",
                        "minPrice": "1",
                        "maxPrice": "100000",
                    },
                    {
                        "filterType": "MARKET_LOT_SIZE",
                        "stepSize": "1",
                        "minQty": "1",
                        "maxQty": "10",
                    },
                    {
                        "filterType": "LOT_SIZE",
                        "stepSize": "0.1",
                        "minQty": "0.1",
                        "maxQty": "100",
                    },
                ],
            }],
        }

    def test_market_quantity_uses_market_lot_size(self):
        self._patch(
            exchange,
            "get_exchange_info",
            return_value=self._exchange_info(),
        )

        self.assertEqual(exchange.normalize_order_quantity(SYMBOL, 2.6), 2)
        self.assertEqual(exchange.normalize_order_quantity(SYMBOL, 0.5), 0)

    def test_ioc_limit_quantity_uses_lot_size_not_market_lot_size(self):
        self._patch(
            exchange,
            "get_exchange_info",
            return_value=self._exchange_info(),
        )
        self._patch(
            exchange,
            "get_book_ticker",
            return_value={
                "bid": 99,
                "ask": 100,
                "mid": 99.5,
                "spread_bps": 100.5,
            },
        )
        self._patch(exchange, "normalize_order_price", return_value=100)
        submit = self._patch(
            exchange,
            "_submit_entry_ioc_limit_order",
            return_value=execution_order("FILLED", 0.5),
        )
        self._patch(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, {"amount": 0.5, "entry_price": 100}, 1),
        )
        fallback = self._patch(
            exchange,
            "_place_reconciled_market_order",
            return_value=reconciled_market_order(0.5),
        )

        exchange.place_market_order(
            SYMBOL,
            "BUY",
            0.5,
            pre_position_amount=0,
            context="ENTRY",
        )

        submit.assert_called_once()
        self.assertEqual(submit.call_args.args[2], 0.5)
        fallback.assert_not_called()

    def test_coarse_tick_rounding_never_exceeds_max_cross_price(self):
        self._patch(
            exchange,
            "get_book_ticker",
            return_value={
                "bid": 99,
                "ask": 100,
                "mid": 99.5,
                "spread_bps": 100.5,
            },
        )
        self._patch(
            exchange,
            "get_symbol_price_rules",
            return_value={
                "available": True,
                "tick_size": "1",
                "min_price": "1",
                "max_price": "100000",
                "precision": 2,
            },
        )
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        submit = self._patch(
            exchange,
            "_submit_entry_ioc_limit_order",
            return_value=execution_order("FILLED", 1),
        )
        self._patch(
            exchange,
            "_wait_for_position_reconciliation",
            return_value=(True, {"amount": 1, "entry_price": 101}, 1),
        )
        fallback = self._patch(exchange, "_place_reconciled_market_order")

        exchange.place_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            context="ENTRY",
        )

        submitted_limit = submit.call_args.args[3]
        maximum_cross_price = 100 * (1 + 2 / 10000)
        self.assertLessEqual(submitted_limit, maximum_cross_price)
        fallback.assert_not_called()


class PositionVerificationTests(OfflineExecutionCase):
    def test_terminal_zero_fill_does_not_verify_an_absent_position(self):
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        self._patch(
            exchange,
            "_new_execution_client_order_id",
            return_value="cid-market",
        )
        self._patch(
            exchange,
            "_submit_entry_market_order",
            return_value=execution_order(
                "EXPIRED",
                0,
                order_id="market-1",
                client_order_id="cid-market",
            ),
        )
        self._patch(
            exchange,
            "_execution_position_detail",
            return_value=(True, None),
        )

        result = exchange._place_reconciled_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            context="ENTRY",
        )

        self.assertIsNone(result)
        record = self.telemetry.call_args.args[0]
        self.assertFalse(record["position_verified"])
        self.assertEqual(record["executed_quantity"], 0)

    def test_unrelated_position_increase_never_creates_entry_fill_attribution(self):
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        self._patch(
            exchange,
            "_new_execution_client_order_id",
            side_effect=["group-id", "cid-market"],
        )
        self._patch(
            exchange,
            "_submit_entry_market_order",
            return_value=execution_order(
                "EXPIRED",
                0,
                order_id="market-1",
                client_order_id="cid-market",
            ),
        )
        self._patch(
            exchange,
            "_execution_position_detail",
            return_value=(
                True,
                {"amount": 1, "side": "BUY", "entry_price": 100},
            ),
        )

        result = exchange._place_reconciled_market_order(
            SYMBOL,
            "BUY",
            1,
            pre_position_amount=0,
            context="ENTRY",
        )

        self.assertIsNone(result)
        record = self.telemetry.call_args.args[0]
        self.assertEqual(record["executed_quantity"], 0)
        self.assertEqual(record["observed_position_increase_quantity"], 1)
        self.assertFalse(record["position_verified"])


class CloseReconciliationTests(OfflineExecutionCase):
    def test_stale_size_is_replaced_by_the_live_position_quantity(self):
        self._patch(
            exchange,
            "normalize_order_quantity",
            side_effect=lambda _, q, **__: q,
        )
        self._patch(
            exchange,
            "_close_position_snapshot",
            return_value=(
                True,
                {
                    "amount": 2,
                    "side": "BUY",
                    "position_side": "BOTH",
                    "entry_price": 95,
                    "mark_price": 100,
                },
                "",
            ),
        )
        submit = self._patch(
            exchange,
            "_submit_position_close_once",
            return_value=(execution_order("FILLED", 2), "SELL", "BOTH"),
        )
        self._patch(
            exchange,
            "_wait_for_close_position_reconciliation",
            return_value=(True, None, 1, ""),
        )

        result = exchange.close_position_market(SYMBOL, 1)
        reconciliation = exchange.get_execution_reconciliation(result)

        submit.assert_called_once_with(
            SYMBOL,
            2,
            position_side=None,
            client_order_id=ANY,
        )
        self.assertEqual(reconciliation["requested_quantity"], 2)
        self.assertEqual(reconciliation["executed_quantity"], 2)
        self.assertTrue(reconciliation["position_closed"])

    def test_stale_long_close_never_closes_a_new_opposite_short(self):
        opposite_position = {
            "symbol": SYMBOL,
            "amount": -2,
            "side": "SELL",
            "position_side": "SHORT",
            "quantity": 2,
            "entry_price": 105,
            "mark_price": 100,
        }
        self._patch(
            exchange,
            "get_open_position_detail_rows",
            return_value=[opposite_position],
        )
        submit = self._patch(exchange, "_submit_position_close_once")

        result = exchange.close_position_market(
            SYMBOL,
            1,
            position_side="LONG",
        )
        submit.assert_not_called()
        self.assertIsNone(result)

    def test_one_way_close_uses_authoritative_live_opposite_sign(self):
        opposite_position = {
            "symbol": SYMBOL,
            "amount": -2,
            "side": "SELL",
            "position_side": "BOTH",
            "quantity": 2,
            "entry_price": 105,
            "mark_price": 100,
        }
        self._patch(
            exchange,
            "get_open_position_detail_rows",
            return_value=[opposite_position],
        )
        self._patch(exchange, "normalize_order_quantity", return_value=2)
        self._patch(
            exchange,
            "_close_position_snapshot",
            return_value=(True, opposite_position, ""),
        )
        submit = self._patch(
            exchange,
            "_submit_position_close_once",
            return_value=(execution_order("FILLED", 2), "BUY", "BOTH"),
        )
        self._patch(
            exchange,
            "_wait_for_close_position_reconciliation",
            return_value=(True, None, 1, ""),
        )

        result = exchange.close_position_market(SYMBOL, 1)
        reconciliation = exchange.get_execution_reconciliation(result)

        submit.assert_called_once_with(
            SYMBOL,
            -2,
            position_side=None,
            client_order_id=ANY,
        )
        self.assertEqual(reconciliation["pre_position_amount"], -2)
        self.assertEqual(reconciliation["executed_quantity"], 2)
        self.assertTrue(reconciliation["position_closed"])

    def test_confirmed_close_preserves_exact_order_attribution(self):
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        self._patch(
            exchange,
            "_close_position_snapshot",
            return_value=(
                True,
                {
                    "amount": 1,
                    "side": "BUY",
                    "position_side": "BOTH",
                    "entry_price": 95,
                    "mark_price": 100,
                },
                "",
            ),
        )
        self._patch(
            exchange,
            "_new_execution_client_order_id",
            side_effect=["close-group-77", "close-client-77"],
        )
        exact_order = execution_order(
            "FILLED",
            1,
            average_price=99.75,
            order_id="close-77",
            client_order_id="close-client-77",
        )
        self._patch(
            exchange,
            "_submit_position_close_once",
            return_value=(exact_order, "SELL", "BOTH"),
        )
        self._patch(
            exchange,
            "_wait_for_close_position_reconciliation",
            return_value=(True, None, 1, ""),
        )

        result = exchange.close_position_market(
            SYMBOL,
            1,
            reference_price=100,
            context="TP1_EXIT",
        )
        reconciliation = exchange.get_execution_reconciliation(result)

        self.assertEqual(reconciliation["executed_quantity"], 1)
        self.assertEqual(reconciliation["average_fill_price"], 99.75)
        self.assertEqual(reconciliation["order_ids"], "close-77")
        self.assertEqual(
            reconciliation["client_order_ids"],
            "close-client-77",
        )
        self.assertEqual(reconciliation["context"], "TP1_EXIT")
        self.assertTrue(reconciliation["position_closed"])

    def test_external_position_change_is_not_attributed_to_close_order(self):
        self._patch(exchange, "normalize_order_quantity", return_value=1)
        self._patch(
            exchange,
            "_close_position_snapshot",
            return_value=(
                True,
                {
                    "amount": 1,
                    "side": "BUY",
                    "position_side": "BOTH",
                    "entry_price": 95,
                    "mark_price": 100,
                },
                "",
            ),
        )
        exact_partial_order = execution_order(
            "FILLED",
            0.25,
            average_price=99.75,
            order_id="close-partial-1",
            client_order_id="close-partial-client-1",
        )
        self._patch(
            exchange,
            "_submit_position_close_once",
            return_value=(exact_partial_order, "SELL", "BOTH"),
        )
        self._patch(
            exchange,
            "_wait_for_close_position_reconciliation",
            return_value=(
                True,
                {
                    "amount": 0.5,
                    "side": "BUY",
                    "position_side": "BOTH",
                    "entry_price": 95,
                    "mark_price": 100,
                },
                1,
                "",
            ),
        )

        result = exchange.close_position_market(SYMBOL, 1)

        self.assertIsNone(result)
        record = self.telemetry.call_args.args[0]
        self.assertEqual(record["status"], "RESIDUAL_OPEN")
        self.assertEqual(record["residual_quantity"], 0.5)
        self.assertEqual(record["executed_quantity"], 0.25)
        self.assertEqual(record["order_ids"], "close-partial-1")


class ExecutionTelemetryHardeningTests(unittest.TestCase):
    def setUp(self):
        execution_telemetry.flush_execution_telemetry()

    def tearDown(self):
        execution_telemetry.flush_execution_telemetry()

    def test_async_journal_flushes_extended_execution_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "execution.csv"

            with patch.object(
                config,
                "EXECUTION_TELEMETRY_ENABLED",
                True,
            ), patch.object(
                config,
                "EXECUTION_TELEMETRY_PATH",
                str(path),
            ):
                self.assertTrue(
                    execution_telemetry.validate_execution_telemetry_path()
                )
                self.assertTrue(execution_telemetry.append_execution_telemetry({
                    "execution_id": "v7ioc-test",
                    "context": "ENTRY",
                    "execution_mode": "SMART_IOC_MARKET_FALLBACK",
                    "fallback_used": True,
                    "symbol": SYMBOL,
                    "order_side": "SELL",
                    "requested_quantity": 1,
                    "executed_quantity": 1,
                    "fallback_quantity": 0.4,
                    "best_bid": 100,
                    "best_ask": 100.1,
                    "limit_price": 100,
                    "slippage_bps": execution_telemetry.calculate_slippage_bps(
                        "SELL",
                        100,
                        99,
                    ),
                }))
                execution_telemetry.flush_execution_telemetry()
                health = execution_telemetry.execution_telemetry_health()

            with path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["execution_id"], "v7ioc-test")
        self.assertEqual(
            rows[0]["execution_mode"],
            "SMART_IOC_MARKET_FALLBACK",
        )
        self.assertEqual(rows[0]["fallback_used"], "True")
        self.assertEqual(float(rows[0]["fallback_quantity"]), 0.4)
        self.assertEqual(float(rows[0]["slippage_bps"]), 100)
        self.assertFalse(health["worker_running"])


if __name__ == "__main__":
    unittest.main()
