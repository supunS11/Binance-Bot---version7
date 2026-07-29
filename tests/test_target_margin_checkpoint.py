import unittest
from unittest.mock import patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


class TargetMarginCheckpointDoesNotStopTheBotTests(unittest.TestCase):
    """The user explicitly asked: reaching TARGET_MARGIN_BALANCE should
    close all positions and let the bot keep trading afterward, not stop
    the process (the old behavior)."""

    def tearDown(self):
        main.shutdown_event.clear()

    def test_checkpoint_closes_positions_without_setting_shutdown_event(self):
        with patch.object(
            main, "close_all_open_positions_for_target_stop", return_value=True
        ) as close_all, patch.object(
            main, "send_telegram_message"
        ), patch.object(
            main, "force_target_margin_process_exit"
        ) as force_exit, patch.object(
            main, "transfer_target_margin_profit_to_spot"
        ) as transfer:
            main.trigger_target_margin_checkpoint(2100)

        close_all.assert_called_once()
        force_exit.assert_not_called()
        transfer.assert_called_once()
        self.assertFalse(main.shutdown_event.is_set())

    def test_checkpoint_skips_transfer_when_close_did_not_succeed(self):
        with patch.object(
            main, "close_all_open_positions_for_target_stop", return_value=False
        ), patch.object(
            main, "send_telegram_message"
        ), patch.object(
            main, "transfer_target_margin_profit_to_spot"
        ) as transfer:
            main.trigger_target_margin_checkpoint(2100)

        transfer.assert_not_called()

    def test_checkpoint_is_a_noop_once_shutdown_is_already_requested(self):
        main.shutdown_event.set()

        with patch.object(
            main, "close_all_open_positions_for_target_stop"
        ) as close_all:
            main.trigger_target_margin_checkpoint(2100)

        close_all.assert_not_called()


class TargetMarginProfitTransferTests(unittest.TestCase):
    def test_disabled_by_config_skips_everything(self):
        with patch.object(
            config, "TARGET_MARGIN_TRANSFER_ENABLED", False
        ), patch.object(main, "get_margin_balance") as get_balance, patch.object(
            main, "transfer_futures_balance_to_spot"
        ) as transfer:
            main.transfer_target_margin_profit_to_spot()

        get_balance.assert_not_called()
        transfer.assert_not_called()

    def test_transfers_the_amount_above_target(self):
        with patch.object(
            config, "TARGET_MARGIN_TRANSFER_ENABLED", True
        ), patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ), patch.object(
            config, "TARGET_MARGIN_TRANSFER_ASSET", "USDT"
        ), patch.object(
            config, "TARGET_MARGIN_TRANSFER_MIN_AMOUNT", 1
        ), patch.object(
            main, "get_margin_balance", return_value=2137.5
        ), patch.object(
            main, "transfer_futures_balance_to_spot", return_value=(True, {})
        ) as transfer, patch.object(
            main, "send_telegram_message"
        ) as telegram:
            main.transfer_target_margin_profit_to_spot()

        transfer.assert_called_once_with("USDT", 137.5)
        telegram.assert_called_once()

    def test_amount_below_minimum_is_not_transferred(self):
        with patch.object(
            config, "TARGET_MARGIN_TRANSFER_ENABLED", True
        ), patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ), patch.object(
            config, "TARGET_MARGIN_TRANSFER_MIN_AMOUNT", 5
        ), patch.object(
            main, "get_margin_balance", return_value=2002
        ), patch.object(
            main, "transfer_futures_balance_to_spot"
        ) as transfer:
            main.transfer_target_margin_profit_to_spot()

        transfer.assert_not_called()

    def test_balance_lookup_failure_does_not_raise(self):
        with patch.object(
            config, "TARGET_MARGIN_TRANSFER_ENABLED", True
        ), patch.object(
            main, "get_margin_balance", side_effect=RuntimeError("offline")
        ), patch.object(
            main, "transfer_futures_balance_to_spot"
        ) as transfer:
            main.transfer_target_margin_profit_to_spot()

        transfer.assert_not_called()

    def test_transfer_failure_is_logged_without_a_false_success_message(self):
        with patch.object(
            config, "TARGET_MARGIN_TRANSFER_ENABLED", True
        ), patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ), patch.object(
            config, "TARGET_MARGIN_TRANSFER_MIN_AMOUNT", 1
        ), patch.object(
            main, "get_margin_balance", return_value=2100
        ), patch.object(
            main,
            "transfer_futures_balance_to_spot",
            return_value=(False, "APIError(code=-9000): insufficient balance"),
        ), patch.object(
            main, "send_telegram_message"
        ) as telegram:
            main.transfer_target_margin_profit_to_spot()

        telegram.assert_not_called()


class TargetMarginMonitorRepeatingCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.monitor = main.TargetMarginBalanceMonitor()

    def test_first_crossing_triggers_and_disarms(self):
        with patch.object(
            main, "get_private_rest_backoff_remaining", return_value=0
        ), patch.object(
            main, "get_margin_balance", return_value=2100
        ), patch.object(
            main, "trigger_target_margin_checkpoint"
        ) as trigger, patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ):
            self.monitor._check_once(interval=1)

        trigger.assert_called_once_with(2100)
        self.assertFalse(self.monitor.armed)

    def test_staying_at_or_above_target_does_not_retrigger(self):
        self.monitor.armed = False

        with patch.object(
            main, "get_private_rest_backoff_remaining", return_value=0
        ), patch.object(
            main, "get_margin_balance", return_value=2100
        ), patch.object(
            main, "trigger_target_margin_checkpoint"
        ) as trigger, patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ):
            self.monitor._check_once(interval=1)

        trigger.assert_not_called()

    def test_dropping_below_target_rearms_without_triggering(self):
        self.monitor.armed = False

        with patch.object(
            main, "get_private_rest_backoff_remaining", return_value=0
        ), patch.object(
            main, "get_margin_balance", return_value=1500
        ), patch.object(
            main, "trigger_target_margin_checkpoint"
        ) as trigger, patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ):
            self.monitor._check_once(interval=1)

        trigger.assert_not_called()
        self.assertTrue(self.monitor.armed)

    def test_full_cycle_triggers_again_after_dropping_and_rising(self):
        sequence = [2100, 2100, 1500, 2100]

        with patch.object(
            main, "get_private_rest_backoff_remaining", return_value=0
        ), patch.object(
            main, "get_margin_balance", side_effect=sequence
        ), patch.object(
            main, "trigger_target_margin_checkpoint"
        ) as trigger, patch.object(
            config, "TARGET_MARGIN_BALANCE", 2000
        ):
            for _ in sequence:
                self.monitor._check_once(interval=1)

        self.assertEqual(trigger.call_count, 2)
        self.assertEqual(
            [call.args[0] for call in trigger.call_args_list],
            [2100, 2100],
        )

    def test_backoff_path_skips_balance_check_and_reports_it_already_waited(self):
        with patch.object(
            main, "get_private_rest_backoff_remaining", return_value=30
        ), patch.object(
            main, "get_margin_balance"
        ) as get_balance, patch.object(
            self.monitor.stop_event, "wait"
        ) as wait:
            already_waited = self.monitor._check_once(interval=5)

        self.assertTrue(already_waited)
        get_balance.assert_not_called()
        wait.assert_called_once_with(30)


if __name__ == "__main__":
    unittest.main()
