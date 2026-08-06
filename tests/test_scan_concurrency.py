import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import patch

import pandas as pd

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import exchange
    import main


class ThrottleKlineRequestTests(unittest.TestCase):
    def test_throttle_delegates_pacing_to_weight_limiter_only(self):
        with patch.object(
            exchange, "_raise_if_public_rest_backoff"
        ) as backoff, patch.object(
            exchange, "_rate_limit_public_request"
        ) as rate_limit, patch.object(
            config, "KLINE_REQUEST_WEIGHT", 2
        ):
            exchange._throttle_kline_request()

        backoff.assert_called_once_with("klines")
        rate_limit.assert_called_once_with(2)

    def test_back_to_back_calls_do_not_block_on_a_flat_delay(self):
        # With the old lock+sleep design, two calls inside
        # REQUEST_THROTTLE_SECONDS of each other would block. There's no
        # such setting anymore - only the weight limiter can slow this
        # down, and it's mocked out here, so both calls return instantly.
        with patch.object(
            exchange, "_raise_if_public_rest_backoff"
        ), patch.object(
            exchange, "_rate_limit_public_request"
        ) as rate_limit, patch("exchange.time.sleep") as sleep_mock:
            exchange._throttle_kline_request()
            exchange._throttle_kline_request()

        self.assertEqual(rate_limit.call_count, 2)
        sleep_mock.assert_not_called()

    def test_dead_throttle_state_was_removed(self):
        self.assertFalse(hasattr(exchange, "_kline_request_lock"))
        self.assertFalse(hasattr(exchange, "_last_kline_request_at"))
        self.assertFalse(hasattr(config, "REQUEST_THROTTLE_SECONDS"))


def _fake_df(rows=5):
    return pd.DataFrame({
        "open": [1.0] * rows,
        "high": [1.0] * rows,
        "low": [1.0] * rows,
        "close": [1.0] * rows,
        "atr": [1.0] * rows,
        "volume": [1.0] * rows,
        "volume_sma": [1.0] * rows,
    })


class ScanSymbolWorkerTests(unittest.TestCase):
    def test_returns_none_when_frames_unavailable(self):
        with patch.object(
            main, "get_signal_frames", return_value=(None, None, None)
        ):
            result = main.scan_symbol_worker("BTCUSDT", object(), "UP")

        self.assertIsNone(result)

    def test_returns_scan_item_and_breadth_sample_when_ready(self):
        df = _fake_df()

        with patch.object(
            main, "get_signal_frames", return_value=(df, df, df)
        ), patch.object(
            main, "build_breadth_sample", return_value={"symbol": "BTCUSDT"}
        ), patch.object(
            main, "observe_signal_outcomes"
        ), patch.object(
            main, "calculate_btc_context", return_value=(0.5, 1.2)
        ), patch.object(
            main, "analyze_signal_cached", return_value={"best_confidence": 90}
        ), patch.object(
            main, "should_fetch_futures_context", return_value=False
        ):
            result = main.scan_symbol_worker("BTCUSDT", object(), "UP")

        self.assertIsNotNone(result)
        self.assertEqual(result["breadth_sample"], {"symbol": "BTCUSDT"})
        self.assertEqual(result["scan_item"]["symbol"], "BTCUSDT")
        self.assertNotIn("futures_priority", result["scan_item"])

    def test_flags_futures_priority_when_context_should_be_queued(self):
        df = _fake_df()

        with patch.object(
            main, "get_signal_frames", return_value=(df, df, df)
        ), patch.object(
            main, "build_breadth_sample", return_value=None
        ), patch.object(
            main, "observe_signal_outcomes"
        ), patch.object(
            main, "calculate_btc_context", return_value=(0.5, 1.2)
        ), patch.object(
            main, "analyze_signal_cached", return_value={"best_confidence": 90}
        ), patch.object(
            main, "should_fetch_futures_context", return_value=True
        ), patch.object(
            main, "futures_context_priority", return_value=3
        ):
            result = main.scan_symbol_worker("BTCUSDT", object(), "UP")

        self.assertEqual(result["scan_item"]["futures_priority"], 3)

    def test_worker_runs_correctly_across_a_real_thread_pool(self):
        # Genuine concurrency smoke test: several "symbols" processed by
        # a real ThreadPoolExecutor should each get their own correct
        # result back, with no cross-symbol data corruption.
        df = _fake_df()
        symbols = [f"SYM{i}USDT" for i in range(10)]

        def fake_signal_frames(symbol, btc_trend_df):
            return df, df, df

        def fake_analyze(*args, **kwargs):
            return {"best_confidence": 90, "namespace": kwargs.get("cache_namespace")}

        with patch.object(
            main, "get_signal_frames", side_effect=fake_signal_frames
        ), patch.object(
            main, "build_breadth_sample", return_value=None
        ), patch.object(
            main, "observe_signal_outcomes"
        ), patch.object(
            main, "calculate_btc_context", return_value=(0.5, 1.2)
        ), patch.object(
            main, "analyze_signal_cached", side_effect=fake_analyze
        ), patch.object(
            main, "should_fetch_futures_context", return_value=False
        ):
            results = {}
            with ThreadPoolExecutor(max_workers=4) as pool:
                future_to_symbol = {
                    pool.submit(
                        main.scan_symbol_worker, symbol, object(), "UP"
                    ): symbol
                    for symbol in symbols
                }
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    results[symbol] = future.result()

        self.assertEqual(len(results), len(symbols))
        for symbol in symbols:
            self.assertEqual(results[symbol]["scan_item"]["symbol"], symbol)
            self.assertEqual(
                results[symbol]["scan_item"]["analysis"]["namespace"],
                symbol,
            )


class FuturesContextWorkerTests(unittest.TestCase):
    def test_returns_participation_and_analysis_without_mutating_scan_item(self):
        df = _fake_df()
        scan_item = {
            "symbol": "BTCUSDT",
            "trend_df": df,
            "confirm_df": df,
            "entry_df": df,
            "btc_trend": "UP",
            "btc_corr": 0.5,
            "rs": 1.2,
        }

        with patch.object(
            main, "get_futures_participation", return_value={"oi_change_pct": 5}
        ), patch.object(
            main, "analyze_signal", return_value={"best_confidence": 91}
        ) as analyze:
            participation, analysis = main.futures_context_worker(scan_item)

        self.assertEqual(participation, {"oi_change_pct": 5})
        self.assertEqual(analysis, {"best_confidence": 91})
        # must not have mutated the caller's scan_item itself
        self.assertNotIn("participation", scan_item)
        self.assertNotIn("analysis", scan_item)
        analyze.assert_called_once_with(
            df, df, df, "UP", 0.5, 1.2,
            participation={"oi_change_pct": 5},
            log_details=False,
        )

    def test_runs_correctly_across_a_real_thread_pool(self):
        df = _fake_df()
        items = [
            {
                "symbol": f"SYM{i}USDT",
                "trend_df": df,
                "confirm_df": df,
                "entry_df": df,
                "btc_trend": "UP",
                "btc_corr": 0.5,
                "rs": 1.2,
            }
            for i in range(10)
        ]

        def fake_participation(symbol):
            return {"symbol": symbol}

        def fake_analyze(*args, **kwargs):
            return {"participation_symbol": kwargs["participation"]["symbol"]}

        with patch.object(
            main, "get_futures_participation", side_effect=fake_participation
        ), patch.object(main, "analyze_signal", side_effect=fake_analyze):
            with ThreadPoolExecutor(max_workers=4) as pool:
                future_to_item = {
                    pool.submit(main.futures_context_worker, item): item
                    for item in items
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    participation, analysis = future.result()
                    self.assertEqual(participation["symbol"], item["symbol"])
                    self.assertEqual(
                        analysis["participation_symbol"], item["symbol"]
                    )


if __name__ == "__main__":
    unittest.main()
