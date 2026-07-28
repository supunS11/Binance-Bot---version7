import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main

SYMBOL = "BTCUSDT"


def _pending(age_seconds, **overrides):
    created_at = datetime.now() - timedelta(seconds=age_seconds)
    pending = {"created_at": created_at.isoformat(timespec="seconds")}
    pending.update(overrides)
    return pending


class LogStuckPendingExecutionsTests(unittest.TestCase):
    def test_does_not_log_below_threshold(self):
        state = {"pending_executions": {SYMBOL: _pending(10)}}

        with patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_SECONDS", 120
        ), patch.object(main, "log_error") as log_error:
            main._log_stuck_pending_executions(state)

        log_error.assert_not_called()

    def test_logs_lookup_uncertain_reason_and_error_detail_past_threshold(self):
        pending = _pending(
            150,
            last_reconciliation={
                "lookup_uncertain": True,
                "error": "-1021 Timestamp for this request is outside recvWindow",
            },
        )
        state = {"pending_executions": {SYMBOL: pending}}

        with patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_SECONDS", 120
        ), patch.object(main, "log_error") as log_error:
            main._log_stuck_pending_executions(state)

        log_error.assert_called_once()
        message = log_error.call_args[0][0]
        self.assertIn(SYMBOL, message)
        self.assertIn("REASON=LOOKUP_UNCERTAIN", message)
        self.assertIn("-1021", message)

    def test_falls_back_to_absence_reset_reason_when_not_lookup_uncertain(self):
        pending = _pending(
            150,
            last_absence_reset_reason="SYMBOL_POSITION_SNAPSHOT_UNAVAILABLE",
        )
        state = {"pending_executions": {SYMBOL: pending}}

        with patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_SECONDS", 120
        ), patch.object(main, "log_error") as log_error:
            main._log_stuck_pending_executions(state)

        message = log_error.call_args[0][0]
        self.assertIn("REASON=SYMBOL_POSITION_SNAPSHOT_UNAVAILABLE", message)
        self.assertIn("LAST_ERROR=NONE", message)

    def test_does_not_repeat_alert_before_repeat_interval_elapses(self):
        pending = _pending(150)
        state = {"pending_executions": {SYMBOL: pending}}

        with patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_SECONDS", 120
        ), patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_REPEAT_SECONDS", 60
        ), patch.object(main, "log_error") as log_error:
            main._log_stuck_pending_executions(state)
            self.assertEqual(log_error.call_count, 1)

            # Simulate only 10 more seconds elapsing - still within the
            # 60s repeat window, so it must not log again.
            pending["created_at"] = (
                datetime.now() - timedelta(seconds=160)
            ).isoformat(timespec="seconds")
            main._log_stuck_pending_executions(state)

        self.assertEqual(log_error.call_count, 1)

    def test_repeats_alert_once_repeat_interval_elapses(self):
        pending = _pending(150)
        state = {"pending_executions": {SYMBOL: pending}}

        with patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_SECONDS", 120
        ), patch.object(
            config, "PENDING_EXECUTION_STUCK_ALERT_REPEAT_SECONDS", 60
        ), patch.object(main, "log_error") as log_error:
            main._log_stuck_pending_executions(state)
            self.assertEqual(log_error.call_count, 1)

            pending["created_at"] = (
                datetime.now() - timedelta(seconds=220)
            ).isoformat(timespec="seconds")
            main._log_stuck_pending_executions(state)

        self.assertEqual(log_error.call_count, 2)


if __name__ == "__main__":
    unittest.main()
