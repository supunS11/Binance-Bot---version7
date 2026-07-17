import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import config
from multi_tp import (
    RUNNER_ACTIVE,
    RUNNER_PENDING,
    TP1_PENDING,
    build_multi_tp_state,
    calculate_runner_stop,
    tp1_fill_confirmed,
)


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import exchange
    import backtest
    import main
    import trade_state


def apply_updates(state, symbol, updates):
    state["positions"][symbol].update(updates)
    return True


class MultiTpStateTests(unittest.TestCase):
    def test_builds_persisted_tp1_state_from_order_result(self):
        state = build_multi_tp_state({
            "multi_tp_active": True,
            "tp_price": 105,
            "tp1_close_pct": 50,
            "tp1_quantity": 0.5,
            "tp1_base_quantity": 1,
            "tp_order": {"algoId": 101},
            "sl_order": {"algoId": 202},
        })

        self.assertTrue(state["multi_tp_active"])
        self.assertEqual(state["multi_tp_stage"], TP1_PENDING)
        self.assertEqual(state["tp1_order_id"], 101)
        self.assertEqual(state["initial_sl_order_id"], 202)

    def test_fill_requires_the_expected_quantity_reduction(self):
        self.assertFalse(tp1_fill_confirmed(1, 0.5, 0.75))
        self.assertTrue(tp1_fill_confirmed(1, 0.5, 0.5))

    def test_runner_stop_locks_profit_and_stays_below_buy_market(self):
        frame = pd.DataFrame({
            "low": [100.8, 101.0, 101.2, 101.4, 101.5, 101.7, 101.9, 102.0, 102.1, 102.2],
            "high": [101.5, 101.7, 101.9, 102.0, 102.2, 102.4, 102.6, 102.8, 103.0, 103.1],
            "atr": [1.0] * 10,
        })

        with patch.object(config, "TP1_RUNNER_MIN_LOCK_ROI", 5), patch.object(
            config,
            "TP1_RUNNER_BREAKEVEN_BUFFER_PCT",
            0.12,
        ):
            stop, info = calculate_runner_stop(
                "BUY",
                100,
                103,
                frame,
                leverage=10,
            )

        self.assertEqual(info["reason"], "RUNNER_STOP_OK")
        self.assertGreaterEqual(stop, 100.5)
        self.assertLess(stop, 103)

    def test_runner_stop_locks_profit_and_stays_above_sell_market(self):
        frame = pd.DataFrame({
            "low": [98.5, 98.3, 98.1, 98.0, 97.8, 97.6, 97.4, 97.2, 97.0, 96.9],
            "high": [99.2, 99.0, 98.8, 98.6, 98.5, 98.3, 98.1, 98.0, 97.9, 97.8],
            "atr": [1.0] * 10,
        })

        with patch.object(config, "TP1_RUNNER_MIN_LOCK_ROI", 5), patch.object(
            config,
            "TP1_RUNNER_BREAKEVEN_BUFFER_PCT",
            0.12,
        ):
            stop, info = calculate_runner_stop(
                "SELL",
                100,
                97,
                frame,
                leverage=10,
            )

        self.assertEqual(info["reason"], "RUNNER_STOP_OK")
        self.assertLessEqual(stop, 99.5)
        self.assertGreater(stop, 97)

    def test_runner_pending_state_survives_restart_reload(self):
        state = {
            "positions": {
                "BTCUSDT": {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                    "multi_tp_stage": RUNNER_PENDING,
                    "runner_sl_order_id": 22,
                    "runner_tp_order_id": "",
                }
            }
        }

        with TemporaryDirectory() as temp_dir, patch.object(
            config,
            "DCA_STATE_PATH",
            str(Path(temp_dir) / "state.json"),
        ):
            trade_state.save_trade_state(state)
            reloaded = trade_state.load_trade_state()

        position = reloaded["positions"]["BTCUSDT"]
        self.assertEqual(position["multi_tp_stage"], RUNNER_PENDING)
        self.assertEqual(position["runner_sl_order_id"], 22)
        self.assertEqual(position["runner_tp_order_id"], "")


class MultiTpExchangeTests(unittest.TestCase):
    def test_partial_tp_is_quantity_based_and_reduce_only_in_one_way_mode(self):
        with patch.object(
            exchange,
            "normalize_order_quantity",
            side_effect=[10.0, 5.0, 5.0],
        ), patch.object(
            exchange,
            "place_algo_order",
            return_value={"algoId": 123},
        ) as place:
            order, quantity = exchange.place_partial_take_profit(
                "BTCUSDT",
                exchange.SIDE_BUY,
                10,
                50,
                105,
            )

        params = place.call_args.kwargs
        self.assertEqual(order["algoId"], 123)
        self.assertEqual(quantity, 5)
        self.assertEqual(params["side"], exchange.SIDE_SELL)
        self.assertEqual(params["quantity"], 5)
        self.assertEqual(params["reduceOnly"], "true")
        self.assertNotIn("closePosition", params)

    def test_partial_tp_uses_position_side_without_reduce_only_in_hedge_mode(self):
        with patch.object(
            exchange,
            "normalize_order_quantity",
            side_effect=[10.0, 5.0, 5.0],
        ), patch.object(
            exchange,
            "place_algo_order",
            return_value={"algoId": 123},
        ) as place:
            exchange.place_partial_take_profit(
                "BTCUSDT",
                exchange.SIDE_BUY,
                10,
                50,
                105,
                position_side="LONG",
            )

        params = place.call_args.kwargs
        self.assertEqual(params["positionSide"], "LONG")
        self.assertNotIn("reduceOnly", params)

    def test_partial_tp_failure_falls_back_to_full_position_protection(self):
        with patch.object(config, "MULTI_TP_ENABLED", True), patch.object(
            config,
            "STATIC_TP_ENABLED",
            True,
        ), patch.object(config, "STATIC_TP_ROI", 20), patch.object(
            exchange,
            "is_stop_loss_enabled_for_signal",
            return_value=False,
        ), patch.object(
            exchange,
            "get_price_precision",
            return_value=2,
        ), patch.object(
            exchange,
            "get_mark_price",
            return_value=100,
        ), patch.object(
            exchange,
            "place_partial_take_profit",
            side_effect=RuntimeError("partial rejected"),
        ), patch.object(
            exchange,
            "place_close_position_protection",
            return_value={"algoId": 321},
        ) as full_order:
            result = exchange.place_tp_sl(
                "BTCUSDT",
                exchange.SIDE_BUY,
                100,
                1,
                None,
                signal_type="TREND",
                enable_multi_tp=True,
                return_details=True,
            )

        self.assertTrue(result["ok"])
        self.assertFalse(result["multi_tp_active"])
        self.assertEqual(result["tp_order"]["algoId"], 321)
        self.assertIn("partial rejected", result["multi_tp_fallback_reason"])
        full_order.assert_called_once()

    def test_accepted_partial_tp_is_cancelled_when_later_protection_fails(self):
        with patch.object(config, "MULTI_TP_ENABLED", True), patch.object(
            config,
            "STATIC_TP_ENABLED",
            True,
        ), patch.object(config, "STATIC_TP_ROI", 20), patch(
            "exchange.is_stop_loss_enabled_for_signal",
            return_value=True,
        ), patch(
            "exchange.get_signal_stop_loss",
            return_value=98,
        ), patch(
            "exchange.get_price_precision",
            return_value=2,
        ), patch(
            "exchange.get_mark_price",
            return_value=100,
        ), patch(
            "exchange.place_partial_take_profit",
            return_value=({"algoId": 123}, 0.5),
        ), patch(
            "exchange.place_close_position_protection",
            side_effect=RuntimeError("SL endpoint unavailable"),
        ), patch(
            "exchange.cancel_algo_order",
            return_value=True,
        ) as cancel:
            result = exchange.place_tp_sl(
                "BTCUSDT",
                exchange.SIDE_BUY,
                100,
                1,
                None,
                signal_type="TREND",
                enable_multi_tp=True,
                return_details=True,
            )

        self.assertFalse(result["ok"])
        self.assertFalse(result["multi_tp_active"])
        self.assertIsNone(result["tp_order"])
        cancel.assert_called_once_with("BTCUSDT", 123)

    def test_uncancelled_partial_tp_remains_tracked_and_blocks_retry(self):
        with patch.object(config, "MULTI_TP_ENABLED", True), patch.object(
            config,
            "STATIC_TP_ENABLED",
            True,
        ), patch.object(config, "STATIC_TP_ROI", 20), patch(
            "exchange.is_stop_loss_enabled_for_signal",
            return_value=True,
        ), patch(
            "exchange.get_signal_stop_loss",
            return_value=98,
        ), patch(
            "exchange.get_price_precision",
            return_value=2,
        ), patch(
            "exchange.get_mark_price",
            return_value=100,
        ), patch(
            "exchange.place_partial_take_profit",
            return_value=({"algoId": 123}, 0.5),
        ), patch(
            "exchange.place_close_position_protection",
            side_effect=RuntimeError("SL endpoint unavailable"),
        ), patch(
            "exchange.cancel_algo_order",
            return_value=False,
        ):
            result = exchange.place_tp_sl(
                "BTCUSDT",
                exchange.SIDE_BUY,
                100,
                1,
                None,
                signal_type="TREND",
                enable_multi_tp=True,
                return_details=True,
            )

        self.assertFalse(result["ok"])
        self.assertTrue(result["multi_tp_active"])
        self.assertTrue(result["protection_cleanup_failed"])
        self.assertEqual(result["uncancelled_tp_order_id"], 123)
        self.assertEqual(result["tp_order"]["algoId"], 123)


class MultiTpMonitorTests(unittest.TestCase):
    def test_recovery_does_not_retry_when_order_cleanup_is_unconfirmed(self):
        failed = {
            "ok": False,
            "protection_cleanup_failed": True,
            "tp_mode": "TP1_TEST",
        }

        with patch.object(config, "TP_ORDER_RETRY_ATTEMPTS", 3), patch.object(
            config,
            "TP_FAILURE_FALLBACK_ROI_ENABLED",
            True,
        ), patch(
            "main.place_tp_sl",
            return_value=failed,
        ) as place, patch("main.send_tp_failure_message"):
            result = main.place_tp_sl_with_recovery(
                "BTCUSDT",
                exchange.SIDE_BUY,
                100,
                1,
                None,
            )

        self.assertIs(result, failed)
        place.assert_called_once()

    def test_persisted_active_runner_keeps_monitor_enabled_after_config_disable(self):
        state = {
            "positions": {
                "BTCUSDT": {
                    "multi_tp_active": True,
                    "multi_tp_stage": RUNNER_ACTIVE,
                }
            }
        }

        with patch.object(config, "MULTI_TP_ENABLED", False), patch.object(
            config,
            "DCA_WEBSOCKET_ENABLED",
            False,
        ), patch("main.load_trade_state", return_value=state):
            monitor = main.DcaWebsocketMonitor()

        self.assertTrue(monitor.enabled)

    def test_closed_position_is_pruned_after_protection_cleanup(self):
        state = {
            "positions": {
                "BTCUSDT": {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                }
            }
        }

        with patch(
            "main.cancel_open_protection_orders",
            return_value=True,
        ) as cleanup, patch(
            "main.prune_closed_positions",
            return_value=["BTCUSDT"],
        ) as prune:
            removed = main.prune_and_cleanup_closed_positions(state, {})

        cleanup.assert_called_once_with("BTCUSDT")
        prune.assert_called_once_with(state, {})
        self.assertEqual(removed, ["BTCUSDT"])

    def test_closed_position_state_is_retained_when_cleanup_must_retry(self):
        state = {
            "positions": {
                "BTCUSDT": {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                }
            }
        }

        with patch(
            "main.cancel_open_protection_orders",
            return_value=False,
        ) as cleanup, patch(
            "main.prune_closed_positions",
            return_value=[],
        ) as prune, patch("main.log_warning"):
            removed = main.prune_and_cleanup_closed_positions(state, {})

        cleanup.assert_called_once_with("BTCUSDT")
        prune.assert_called_once_with(state, {"BTCUSDT": 0})
        self.assertEqual(removed, [])

    def test_confirmed_partial_fill_moves_state_to_runner_pending(self):
        monitor = main.DcaWebsocketMonitor()
        state = {
            "positions": {
                "BTCUSDT": {
                    "managed_by_bot": True,
                    "multi_tp_active": True,
                    "multi_tp_stage": TP1_PENDING,
                    "side": "BUY",
                    "avg_entry": 100,
                    "tp1_price": 105,
                    "tp1_base_quantity": 1,
                    "tp1_quantity": 0.5,
                    "tp1_order_id": 77,
                }
            }
        }
        monitor.synced_position_details = {
            "BTCUSDT": {"quantity": 0.5}
        }

        with patch("main.load_trade_state", return_value=state), patch(
            "main.get_open_position_details",
            return_value={
                "BTCUSDT": {
                    "quantity": 0.5,
                    "amount": 0.5,
                    "position_side": "BOTH",
                }
            },
        ), patch(
            "main.update_position_runtime_fields",
            side_effect=apply_updates,
        ), patch(
            "main.cancel_algo_order",
            return_value=True,
        ), patch.object(
            monitor,
            "_configure_multi_tp_runner",
            return_value=True,
        ) as configure:
            handled = monitor._handle_multi_tp_runner(
                "BTCUSDT",
                105,
                state,
            )

        self.assertTrue(handled)
        self.assertEqual(
            state["positions"]["BTCUSDT"]["multi_tp_stage"],
            RUNNER_PENDING,
        )
        configure.assert_called_once()

    def test_runner_places_stop_before_tp2_and_then_becomes_active(self):
        monitor = main.DcaWebsocketMonitor()
        position = {
            "managed_by_bot": True,
            "multi_tp_active": True,
            "multi_tp_stage": RUNNER_PENDING,
            "side": "BUY",
            "avg_entry": 100,
            "runner_basis_price": 105,
            "runner_tp_price": 108,
            "runner_tp_mode": "TP2_STRUCTURE_TEST",
            "runner_sl_price": 101,
            "runner_sl_mode": "RUNNER_STRUCTURE",
            "runner_sl_order_id": "",
            "runner_tp_order_id": "",
            "initial_sl_order_id": 11,
            "sl_enabled": True,
        }
        state = {"positions": {"BTCUSDT": position}}

        with patch("main.get_price_precision", return_value=2), patch(
            "main.get_open_stop_loss_info",
            return_value={},
        ), patch(
            "main.get_open_take_profit_info",
            return_value={},
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
        ):
            handled = monitor._configure_multi_tp_runner(
                "BTCUSDT",
                105,
                state,
                position,
                {"quantity": 0.5, "position_side": "BOTH"},
            )

        self.assertTrue(handled)
        self.assertEqual(place.call_args_list[0].args[2], "STOP_MARKET")
        self.assertEqual(place.call_args_list[1].args[2], "TAKE_PROFIT_MARKET")
        self.assertEqual(position["multi_tp_stage"], RUNNER_ACTIVE)
        cancel.assert_called_once_with("BTCUSDT", 11)

    def test_dca_tick_is_blocked_after_tp1(self):
        state = {
            "positions": {
                "BTCUSDT": {
                    "managed_by_bot": True,
                    "multi_tp_stage": RUNNER_ACTIVE,
                    "side": "BUY",
                }
            }
        }

        with patch.object(config, "TP1_RUNNER_DISABLE_DCA", True):
            self.assertFalse(main.dca_tick_ready("BTCUSDT", 90, state=state))


class MultiTpBacktestTests(unittest.TestCase):
    def test_backtest_realizes_tp1_and_exits_remaining_size_at_tp2(self):
        interval_ms = 15 * 60_000
        rows = []

        for index in range(305):
            price = 100
            high = 100.5
            low = 99.5

            if index == 300:
                high = 102.2
                low = 100
                price = 102
            elif index == 301:
                high = 104.2
                low = 101.5
                price = 104

            rows.append({
                "time": index * interval_ms,
                "close_time": (index + 1) * interval_ms - 1,
                "open": 100,
                "high": high,
                "low": low,
                "close": price,
            })

        data = pd.DataFrame(rows)
        frame = backtest.BacktestData(raw=data, indicators=data)
        frames = {
            "trend": frame,
            "confirm": frame,
            "entry": frame,
            "exit": frame,
        }
        profit_info = {
            "armed": False,
            "should_exit": False,
            "peak_roi": 0,
            "floor_roi": 0,
        }

        with patch.object(config, "MULTI_TP_ENABLED", True), patch.object(
            config,
            "BACKTEST_MULTI_TP_ENABLED",
            True,
        ), patch.object(config, "TP1_CLOSE_POSITION_PCT", 50), patch.object(
            config,
            "BACKTEST_USE_DCA",
            False,
        ), patch.object(config, "EARLY_FLOW_EXIT_ENABLED", False), patch.object(
            config,
            "BACKTEST_MAX_HOLD_CANDLES",
            5,
        ), patch(
            "backtest.compute_take_profit",
            return_value=(102, "TP1_TEST"),
        ), patch(
            "backtest.compute_runner_take_profit",
            return_value=(104, "TP2_TEST"),
        ), patch(
            "backtest.compute_stop_loss",
            return_value=(None, "SL_DISABLED"),
        ), patch(
            "backtest.calculate_runner_stop",
            return_value=(101, {"source": "PROFIT_LOCK_FLOOR"}),
        ), patch(
            "backtest.evaluate_route_profit_protection",
            return_value=profit_info,
        ):
            result = backtest.simulate_trade(
                "BTCUSDT",
                "BUY",
                "TREND",
                90,
                300 * interval_ms,
                100,
                300,
                frames,
            )

        self.assertTrue(result["tp1_hit"])
        self.assertEqual(result["exit_reason"], "TP2")
        self.assertEqual(result["tp1_close_pct"], 50)
        self.assertEqual(result["tp2_mode"], "TP2_TEST")
        self.assertGreater(result["net_pnl"], 0)


if __name__ == "__main__":
    unittest.main()
