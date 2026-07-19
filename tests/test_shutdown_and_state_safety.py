import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import config
import execution_telemetry


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main
    import trade_state


class TradeStateSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_directory.cleanup)
        self.state_path = (
            Path(self.temp_directory.name) / "open_trades_v7.json"
        )
        self.path_patch = patch.object(
            config,
            "DCA_STATE_PATH",
            str(self.state_path),
        )
        self.path_patch.start()
        self.addCleanup(self.path_patch.stop)

    def test_missing_state_is_allowed_only_when_no_recovery_exists(self):
        self.assertFalse(trade_state.trade_state_file_exists())
        self.assertEqual(
            trade_state.load_trade_state(),
            {"positions": {}, "pending_executions": {}},
        )

    def test_corrupt_existing_state_fails_closed(self):
        self.state_path.write_text("{not-json", encoding="utf-8")

        with self.assertRaises(trade_state.TradeStateLoadError):
            trade_state.load_trade_state()

    def test_atomic_save_retains_previous_backup_for_recovery(self):
        first = {
            "positions": {"BTCUSDT": {"side": "BUY"}},
            "pending_executions": {},
        }
        second = {
            "positions": {"BTCUSDT": {"side": "BUY", "dca_count": 1}},
            "pending_executions": {},
        }

        self.assertTrue(trade_state.save_trade_state(first))
        self.assertTrue(trade_state.save_trade_state(second))
        backup_path = self.state_path.with_name("open_trades_v7.bak.json")
        self.assertTrue(backup_path.exists())
        self.assertEqual(trade_state.load_trade_state(), second)

        self.state_path.unlink()
        self.assertTrue(trade_state.trade_state_file_exists())
        self.assertEqual(trade_state.load_trade_state(), first)

    def test_open_positions_require_primary_or_backup_state(self):
        with patch.object(
            config,
            "REQUIRE_STATE_FOR_OPEN_POSITIONS",
            True,
            create=True,
        ):
            with self.assertRaises(trade_state.TradeStateLoadError):
                main.load_runtime_trade_state({"BTCUSDT": 1})


class GracefulShutdownTests(unittest.TestCase):
    def tearDown(self):
        main.shutdown_event.clear()

    def test_sigterm_requests_graceful_shutdown(self):
        main.request_shutdown(signal.SIGTERM)
        self.assertTrue(main.shutdown_event.is_set())

    def test_signal_handlers_register_sigint_and_sigterm(self):
        with patch.object(main.signal, "signal") as register:
            self.assertTrue(main.install_shutdown_signal_handlers())

        registered_signals = [call.args[0] for call in register.call_args_list]
        self.assertEqual(registered_signals, [signal.SIGINT, signal.SIGTERM])

    def test_target_margin_force_exit_uses_cleanup_path(self):
        with patch.object(
            config,
            "TARGET_MARGIN_FORCE_EXIT_ENABLED",
            True,
        ), patch.object(
            config,
            "TARGET_MARGIN_EXIT_DELAY_SECONDS",
            0,
        ):
            main.force_target_margin_process_exit(close_success=True)

        self.assertTrue(main.shutdown_event.is_set())

    def test_target_monitor_stop_waits_for_close_workflow(self):
        monitor = main.TargetMarginBalanceMonitor()
        close_started = threading.Event()
        allow_close_to_finish = threading.Event()
        close_finished = threading.Event()

        def close_positions():
            close_started.set()
            allow_close_to_finish.wait(2)
            close_finished.set()
            return True

        with patch.object(
            main,
            "close_all_open_positions_for_target_stop",
            side_effect=close_positions,
        ), patch.object(
            main,
            "force_target_margin_process_exit",
        ), patch.object(
            main,
            "send_telegram_message",
        ):
            monitor.thread = threading.Thread(
                target=main.trigger_target_margin_stop,
                args=(1000,),
                daemon=True,
            )
            monitor.thread.start()
            self.assertTrue(close_started.wait(1))

            stop_finished = threading.Event()

            def stop_monitor():
                monitor.stop()
                stop_finished.set()

            stopper = threading.Thread(target=stop_monitor)
            stopper.start()
            time.sleep(0.05)
            self.assertFalse(stop_finished.is_set())

            allow_close_to_finish.set()
            stopper.join(1)

        self.assertTrue(close_finished.is_set())
        self.assertTrue(stop_finished.is_set())
        self.assertIsNone(monitor.thread)

    def test_partial_shadow_start_failure_is_cleaned_up(self):
        dca_monitor = Mock()
        shadow_monitor = Mock()
        shadow_monitor.start.side_effect = RuntimeError("partial startup")
        flow_monitor = Mock()
        target_monitor = Mock()

        with patch.object(
            main.shutdown_event,
            "is_set",
            return_value=True,
        ), patch.multiple(
            main,
            log_info=Mock(),
            log_warning=Mock(),
            install_shutdown_signal_handlers=Mock(),
            sync_client_time=Mock(),
            is_one_way_position_mode=Mock(return_value=True),
            validate_execution_telemetry_path=Mock(return_value=True),
            get_scan_symbols=Mock(return_value=[]),
            log_active_dca_config=Mock(),
            DcaWebsocketMonitor=Mock(return_value=dca_monitor),
            OrderFlowShadowMonitor=Mock(return_value=shadow_monitor),
            MarketFlowMonitor=Mock(return_value=flow_monitor),
            TargetMarginBalanceMonitor=Mock(return_value=target_monitor),
            flush_execution_telemetry=Mock(),
        ):
            main.run_bot()

        shadow_monitor.stop.assert_called_once_with()
        flow_monitor.start.assert_called_once_with()
        flow_monitor.stop.assert_called_once_with()
        dca_monitor.stop.assert_called_once_with()


class ExecutionTelemetryReadinessTests(unittest.TestCase):
    def test_writable_path_is_validated_and_reported(self):
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
                health = execution_telemetry.execution_telemetry_health()

            self.assertEqual(health["path"], str(path))
            self.assertEqual(health["queue_capacity"], 2000)
            self.assertTrue(path.exists())

    def test_unwritable_destination_is_exposed_as_health_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "not-a-file"
            path.mkdir()
            before = execution_telemetry.execution_telemetry_health()[
                "write_errors"
            ]

            with patch.object(
                config,
                "EXECUTION_TELEMETRY_ENABLED",
                True,
            ), patch.object(
                config,
                "EXECUTION_TELEMETRY_PATH",
                str(path),
            ):
                self.assertFalse(
                    execution_telemetry.validate_execution_telemetry_path()
                )
                health = execution_telemetry.execution_telemetry_health()

            self.assertEqual(health["write_errors"], before + 1)
            self.assertTrue(health["last_write_error"])


if __name__ == "__main__":
    unittest.main()
