import unittest
from unittest.mock import Mock, patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import exchange


class PrivateRestCallWeightTests(unittest.TestCase):
    def test_default_weight_does_not_touch_the_shared_budget(self):
        with patch.object(
            exchange, "_raise_if_private_rest_backoff"
        ), patch.object(
            exchange, "_rate_limit_public_request"
        ) as rate_limit:
            exchange._private_rest_call("some_call", lambda: "ok")

        rate_limit.assert_not_called()

    def test_positive_weight_paces_against_the_shared_public_budget(self):
        with patch.object(
            exchange, "_raise_if_private_rest_backoff"
        ), patch.object(
            exchange, "_rate_limit_public_request"
        ) as rate_limit:
            result = exchange._private_rest_call(
                "futures_position_information:all",
                lambda: "ok",
                weight=5,
            )

        rate_limit.assert_called_once_with(5)
        self.assertEqual(result, "ok")

    def test_backoff_is_still_set_on_failure_regardless_of_weight(self):
        def boom():
            raise RuntimeError("APIError(code=-1003): too many requests")

        with patch.object(
            exchange, "_raise_if_private_rest_backoff"
        ), patch.object(
            exchange, "_rate_limit_public_request"
        ), patch.object(
            exchange, "_set_private_rest_backoff"
        ) as set_backoff:
            with self.assertRaises(RuntimeError):
                exchange._private_rest_call("boom_call", boom, weight=5)

        set_backoff.assert_called_once()
        self.assertEqual(set_backoff.call_args.args[1], "boom_call")


class PositionInformationRequestWeightTests(unittest.TestCase):
    def test_all_positions_lookup_uses_the_heavier_weight(self):
        with patch.object(
            exchange, "_get_cached_position_info", return_value=None
        ), patch.object(
            exchange, "is_private_rest_backoff_active", return_value=False
        ), patch.object(
            exchange, "_private_rest_call", return_value=[]
        ) as private_call, patch.object(
            exchange, "_store_position_info"
        ), patch.object(
            config, "POSITION_INFO_REQUEST_WEIGHT", 5
        ):
            exchange._get_futures_position_information(symbol=None)

        self.assertEqual(private_call.call_args.kwargs["weight"], 5)

    def test_single_symbol_lookup_uses_the_lighter_weight(self):
        with patch.object(
            exchange, "_get_cached_position_info", return_value=None
        ), patch.object(
            exchange, "is_private_rest_backoff_active", return_value=False
        ), patch.object(
            exchange, "_private_rest_call", return_value=[]
        ) as private_call, patch.object(
            exchange, "_store_position_info"
        ), patch.object(
            config, "POSITION_INFO_SYMBOL_REQUEST_WEIGHT", 1
        ):
            exchange._get_futures_position_information(symbol="BTCUSDT")

        self.assertEqual(private_call.call_args.kwargs["weight"], 1)


class AlgoOrderQueryRequestWeightTests(unittest.TestCase):
    # futures_get_algo_order was confirmed via a real -1003 backoff log
    # (ERROR= text) to be firing completely unpaced - reconciliation checks
    # against open positions' SL/TP/DCA algo orders were invisible to the
    # shared weight budget entirely.
    def test_algo_order_lookup_via_client_method_is_weighted(self):
        fake_method = Mock(return_value={})

        with patch.object(
            exchange.client, "futures_get_algo_order", fake_method, create=True
        ), patch.object(
            exchange, "_private_rest_call", return_value={}
        ) as private_call, patch.object(
            config, "ALGO_ORDER_QUERY_REQUEST_WEIGHT", 1
        ):
            exchange._get_algo_order("BTCUSDT", algo_id="123")

        self.assertEqual(private_call.call_args.kwargs["weight"], 1)

    def test_algo_order_lookup_via_fallback_request_is_weighted(self):
        with patch.object(
            exchange.client, "futures_get_algo_order", None, create=True
        ), patch.object(
            exchange, "_private_rest_call", return_value={}
        ) as private_call, patch.object(
            config, "ALGO_ORDER_QUERY_REQUEST_WEIGHT", 1
        ):
            exchange._get_algo_order("BTCUSDT", algo_id="123")

        self.assertEqual(private_call.call_args.kwargs["weight"], 1)


class FuturesToSpotTransferTests(unittest.TestCase):
    def test_successful_transfer_uses_type_2_futures_to_spot(self):
        with patch.object(
            exchange, "_private_rest_call", return_value={"tranId": 1}
        ) as private_call:
            ok, result = exchange.transfer_futures_balance_to_spot("USDT", 137.5)

        self.assertTrue(ok)
        self.assertEqual(result, {"tranId": 1})
        self.assertEqual(
            private_call.call_args.kwargs,
            {"asset": "USDT", "amount": 137.5, "type": 2},
        )

    def test_failed_transfer_returns_false_and_the_error_instead_of_raising(self):
        with patch.object(
            exchange,
            "_private_rest_call",
            side_effect=RuntimeError("APIError(code=-9000): insufficient balance"),
        ):
            ok, result = exchange.transfer_futures_balance_to_spot("USDT", 137.5)

        self.assertFalse(ok)
        self.assertIn("insufficient balance", result)


if __name__ == "__main__":
    unittest.main()
