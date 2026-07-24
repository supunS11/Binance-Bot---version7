import unittest
from unittest.mock import patch

import config
from strategy import time_exit_requires_weakness

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def position_state(**overrides):
    values = {
        "managed_by_bot": True,
        "confirmation_type": "RANGE_REVERSION",
        "side": "BUY",
        "avg_entry": 100.0,
        "opened_at": "2020-01-01T00:00:00+00:00",
        "position_exit_owner": "",
        "position_exit_status": "",
        "multi_tp_active": False,
    }
    values.update(overrides)
    return values


class TimeExitRouteTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "TIME_EXIT_ENABLED": True,
            "TIME_EXIT_TREND_ENABLED": True,
            "TIME_EXIT_REVERSAL_ENABLED": False,
            "TIME_EXIT_RANGE_REVERSION_ENABLED": True,
            "TIME_EXIT_MINUTES": 480,
            "TIME_EXIT_MAX_ROI": 0,
            "TIME_EXIT_RANGE_REVERSION_MINUTES": 180,
            "TIME_EXIT_RANGE_REVERSION_MAX_ROI": 0,
            "TIME_EXIT_POST_DCA_GRACE_MINUTES": 0,
        }
        self.config_patches = [
            patch.object(config, name, value) for name, value in settings.items()
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def _context(self, elapsed_minutes, mark_price, **overrides):
        with patch.object(
            main,
            "seconds_since",
            return_value=elapsed_minutes * 60,
        ), patch.object(
            main,
            "coordinated_position_management_enabled",
            return_value=True,
        ), patch.object(
            main,
            "runner_owns_position",
            return_value=False,
        ):
            return main.DcaWebsocketMonitor._time_exit_context(
                None,
                position_state(**overrides),
                mark_price,
            )

    def test_range_reversion_uses_own_shorter_window_not_trend_480(self):
        too_early = self._context(elapsed_minutes=150, mark_price=99.0)
        self.assertIsNone(too_early)

        eligible = self._context(elapsed_minutes=200, mark_price=99.0)
        self.assertIsNotNone(eligible)
        self.assertEqual(eligible["route"], "RANGE_REVERSION")

    def test_same_elapsed_time_does_not_yet_trigger_trend_position(self):
        result = self._context(
            elapsed_minutes=200,
            mark_price=99.0,
            confirmation_type="TREND",
        )
        self.assertIsNone(result)

    def test_profitable_range_position_is_not_forced_out_by_timer(self):
        result = self._context(elapsed_minutes=200, mark_price=105.0)
        self.assertIsNone(result)

    def test_committed_context_labels_range_reversion_route(self):
        with patch.object(main, "seconds_since", return_value=200 * 60), \
                patch.object(main, "runner_owns_position", return_value=False):
            context = main.DcaWebsocketMonitor._committed_time_exit_context(
                None,
                position_state(),
                99.0,
            )
        self.assertEqual(context["route"], "RANGE_REVERSION")


class TimeExitRequiresWeaknessTests(unittest.TestCase):
    def test_trend_falls_back_to_global_default(self):
        with patch.object(config, "TIME_EXIT_REQUIRE_WEAKNESS", True):
            self.assertTrue(time_exit_requires_weakness("TREND"))

        with patch.object(config, "TIME_EXIT_REQUIRE_WEAKNESS", False):
            self.assertFalse(time_exit_requires_weakness("TREND"))

    def test_range_reversion_uses_its_own_override(self):
        with patch.object(config, "TIME_EXIT_REQUIRE_WEAKNESS", True), \
                patch.object(
                    config,
                    "TIME_EXIT_RANGE_REVERSION_REQUIRE_WEAKNESS",
                    False,
                ):
            self.assertFalse(time_exit_requires_weakness("RANGE_REVERSION"))
            self.assertTrue(time_exit_requires_weakness("TREND"))

    def test_missing_route_defaults_to_trend_behavior(self):
        with patch.object(config, "TIME_EXIT_REQUIRE_WEAKNESS", True):
            self.assertTrue(time_exit_requires_weakness(None))
            self.assertTrue(time_exit_requires_weakness(""))


class TimeExitWeaknessGateTests(unittest.TestCase):
    """A flat/losing range-reversion trade bets on mean reversion, so the
    trend-continuation weakness check almost never confirms for it - it was
    silently stuck waiting for a signal that couldn't fire. RANGE_REVERSION
    gets its own require-weakness switch instead of inheriting TREND's."""

    def _run(self, require_weakness):
        state = {"positions": {"BTCUSDT": position_state()}}
        pending_updates = []

        def fake_update(_state, _symbol, updates):
            pending_updates.append(updates)
            return True

        settings = {
            "TIME_EXIT_ENABLED": True,
            "TIME_EXIT_RANGE_REVERSION_ENABLED": True,
            "TIME_EXIT_RANGE_REVERSION_MINUTES": 180,
            "TIME_EXIT_RANGE_REVERSION_MAX_ROI": 0,
            "TIME_EXIT_RANGE_REVERSION_REQUIRE_WEAKNESS": require_weakness,
            "TIME_EXIT_POST_DCA_GRACE_MINUTES": 0,
            "TIME_EXIT_CHECK_SECONDS": 0,
            "TIME_EXIT_REQUIRE_DATA": False,
        }
        config_patches = [
            patch.object(config, name, value) for name, value in settings.items()
        ]

        for config_patch in config_patches:
            config_patch.start()

        try:
            with patch.object(
                main, "seconds_since", return_value=200 * 60
            ), patch.object(
                main, "coordinated_position_management_enabled", return_value=True
            ), patch.object(
                main, "runner_owns_position", return_value=False
            ), patch.object(
                main, "get_signal_frames", return_value=(None, None, None)
            ), patch.object(
                main,
                "evaluate_time_exit_weakness",
                return_value={
                    "should_exit": False,
                    "reason": "TIME_EXIT_WEAKNESS_INCOMPLETE",
                    "evidence": [],
                    "weakness_score": 0,
                },
            ), patch.object(
                main, "load_trade_state", return_value=state
            ), patch.object(
                main, "committed_position_exit_owner", return_value=""
            ), patch.object(
                main, "update_position_runtime_fields", side_effect=fake_update
            ), patch.object(
                main, "get_open_position_details", return_value=None
            ):
                main.DcaWebsocketMonitor()._handle_time_exit(
                    "BTCUSDT",
                    99.0,
                    state,
                )
        finally:
            for config_patch in reversed(config_patches):
                config_patch.stop()

        return pending_updates

    def test_range_reversion_exits_without_waiting_on_trend_weakness(self):
        updates = self._run(require_weakness=False)

        self.assertTrue(
            any(u.get("time_exit_status") == "PENDING" for u in updates)
        )

    def test_trend_style_weakness_requirement_still_blocks_when_opted_in(self):
        updates = self._run(require_weakness=True)

        self.assertEqual(updates, [])


if __name__ == "__main__":
    unittest.main()
