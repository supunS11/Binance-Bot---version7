import unittest
from unittest.mock import patch

import config


with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def position_state(stage, signal_type="TREND"):
    return {
        "managed_by_bot": True,
        "confirmation_type": signal_type,
        "multi_tp_active": True,
        "multi_tp_stage": stage,
    }


class Tp1ExtraSlotTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "MAX_TOTAL_POSITIONS": 6,
            "MAX_BUY_POSITIONS": 3,
            "MAX_SELL_POSITIONS": 3,
            "TP1_EXTRA_SLOTS_ENABLED": True,
            "TP1_EXTRA_TOTAL_POSITIONS": 2,
            "TP1_EXTRA_BUY_POSITIONS": 1,
            "TP1_EXTRA_SELL_POSITIONS": 1,
        }
        self.config_patches = [
            patch.object(config, name, value)
            for name, value in settings.items()
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def base_positions(self, first_stage):
        open_positions = {
            "BUY1": 1,
            "BUY2": 1,
            "BUY3": 1,
            "SELL1": -1,
            "SELL2": -1,
            "SELL3": -1,
        }
        states = {
            symbol: position_state(
                first_stage if symbol == "BUY1" else "TP1_PENDING"
            )
            for symbol in open_positions
        }
        return {"positions": states}, open_positions

    def test_pending_tp1_does_not_unlock_capacity(self):
        state, open_positions = self.base_positions("TP1_PENDING")
        pool_counts = main.get_position_pool_counts(state, open_positions)
        runners = main.get_tp1_runner_pool_counts(state, open_positions)

        allowed, reason = main.check_entry_position_limits(
            "BUY",
            "TREND",
            pool_counts,
            runners,
        )

        self.assertFalse(allowed)
        self.assertIn("MAX POSITIONS REACHED", reason)
        self.assertEqual(runners["TREND"]["total"], 0)

    def test_disabled_feature_does_not_unlock_capacity(self):
        state, open_positions = self.base_positions("RUNNER_ACTIVE")

        with patch.object(config, "TP1_EXTRA_SLOTS_ENABLED", False):
            pool_counts = main.get_position_pool_counts(
                state,
                open_positions,
            )
            runners = main.get_tp1_runner_pool_counts(
                state,
                open_positions,
            )
            allowed, reason = main.check_entry_position_limits(
                "BUY",
                "TREND",
                pool_counts,
                runners,
            )

        self.assertFalse(allowed)
        self.assertIn("MAX POSITIONS REACHED", reason)

    def test_confirmed_buy_runner_unlocks_one_buy_slot(self):
        state, open_positions = self.base_positions("RUNNER_ACTIVE")
        pool_counts = main.get_position_pool_counts(state, open_positions)
        runners = main.get_tp1_runner_pool_counts(state, open_positions)

        allowed, reason = main.check_entry_position_limits(
            "BUY",
            "TREND",
            pool_counts,
            runners,
        )

        self.assertTrue(allowed, reason)
        self.assertEqual(runners["TREND"]["total"], 1)
        self.assertEqual(runners["TREND"]["buy"], 1)

    def test_one_runner_cannot_unlock_a_second_replacement(self):
        state, open_positions = self.base_positions("RUNNER_PENDING")
        open_positions["BUY4"] = 1
        state["positions"]["BUY4"] = position_state("TP1_PENDING")
        pool_counts = main.get_position_pool_counts(state, open_positions)
        runners = main.get_tp1_runner_pool_counts(state, open_positions)

        allowed, reason = main.check_entry_position_limits(
            "BUY",
            "TREND",
            pool_counts,
            runners,
        )

        self.assertFalse(allowed)
        self.assertIn("MAX POSITIONS REACHED", reason)

    def test_buy_runner_does_not_expand_sell_side_limit(self):
        state, open_positions = self.base_positions("RUNNER_ACTIVE")
        pool_counts = main.get_position_pool_counts(state, open_positions)
        runners = main.get_tp1_runner_pool_counts(state, open_positions)

        allowed, reason = main.check_entry_position_limits(
            "SELL",
            "TREND",
            pool_counts,
            runners,
        )

        self.assertFalse(allowed)
        self.assertIn("MAX SELL POSITIONS REACHED", reason)


if __name__ == "__main__":
    unittest.main()
