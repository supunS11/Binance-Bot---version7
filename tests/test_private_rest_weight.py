import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
