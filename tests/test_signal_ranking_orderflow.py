import unittest
from unittest.mock import Mock, patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


def base_candidate(**market_context_overrides):
    market_context = {
        "flow": {},
        "breadth": {},
        "transition": {},
        "calibration": {},
        "order_flow": {},
        "route": "TREND",
    }
    market_context.update(market_context_overrides)
    return {
        "symbol": "BTCUSDT",
        "signal": "BUY",
        "analysis": {
            "buy": {
                "uncapped_score_index": 50,
                "confidence": 50,
                "quality_score": 0,
                "participation_score": 0,
                "smc_score": 0,
                "regime_score": 0,
            },
            "sell": {},
        },
        "market_context": market_context,
    }


class SignalRankOrderFlowTests(unittest.TestCase):
    def setUp(self):
        settings = {
            "SIGNAL_RANKING_ORDERFLOW_ENABLED": False,
            "SIGNAL_RANKING_ORDERFLOW_WEIGHT": 1.0,
        }
        self.config_patches = [
            patch.object(config, name, value) for name, value in settings.items()
        ]

        for config_patch in self.config_patches:
            config_patch.start()

    def tearDown(self):
        for config_patch in reversed(self.config_patches):
            config_patch.stop()

    def test_disabled_by_default_ignores_shadow_score(self):
        with_score = base_candidate(order_flow={"buy_shadow_score": 4.0})
        without_score = base_candidate(order_flow={})

        rank_with = main.calculate_signal_rank(with_score)
        rank_without = main.calculate_signal_rank(without_score)

        self.assertEqual(rank_with, rank_without)

    def test_enabled_applies_weighted_shadow_score(self):
        candidate_zero = base_candidate(order_flow={"buy_shadow_score": 0})
        candidate_boosted = base_candidate(order_flow={"buy_shadow_score": 3.0})

        with patch.object(config, "SIGNAL_RANKING_ORDERFLOW_ENABLED", True), \
                patch.object(config, "SIGNAL_RANKING_ORDERFLOW_WEIGHT", 2.0):
            baseline = main.calculate_signal_rank(candidate_zero)
            boosted = main.calculate_signal_rank(candidate_boosted)

        self.assertAlmostEqual(boosted - baseline, 6.0, places=2)

    def test_enrich_skips_shadow_snapshot_when_disabled(self):
        shadow_monitor = Mock()
        candidate = base_candidate()
        candidate.pop("market_context")

        with patch.object(main, "calculate_regime_transition", return_value={}):
            main.enrich_candidate_market_context(
                candidate,
                None,
                {"available": False},
                shadow_monitor,
            )

        shadow_monitor.snapshot.assert_not_called()
        self.assertFalse(candidate["market_context"]["order_flow"]["available"])

    def test_enrich_calls_shadow_snapshot_without_telemetry_when_enabled(self):
        shadow_monitor = Mock()
        shadow_monitor.snapshot.return_value = {
            "available": True,
            "buy_shadow_score": 1.5,
            "sell_shadow_score": -1.5,
        }
        candidate = base_candidate()
        candidate.pop("market_context")

        with patch.object(config, "SIGNAL_RANKING_ORDERFLOW_ENABLED", True), \
                patch.object(main, "calculate_regime_transition", return_value={}):
            main.enrich_candidate_market_context(
                candidate,
                None,
                {"available": False},
                shadow_monitor,
            )

        shadow_monitor.snapshot.assert_called_once_with(
            "BTCUSDT",
            emit_telemetry=False,
        )
        self.assertEqual(
            candidate["market_context"]["order_flow"]["buy_shadow_score"],
            1.5,
        )

    def test_enrich_handles_missing_shadow_monitor_gracefully(self):
        candidate = base_candidate()
        candidate.pop("market_context")

        with patch.object(config, "SIGNAL_RANKING_ORDERFLOW_ENABLED", True), \
                patch.object(main, "calculate_regime_transition", return_value={}):
            main.enrich_candidate_market_context(
                candidate,
                None,
                {"available": False},
                None,
            )

        self.assertFalse(candidate["market_context"]["order_flow"]["available"])


if __name__ == "__main__":
    unittest.main()
