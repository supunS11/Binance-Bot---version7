import unittest
from unittest.mock import patch

import config

with patch("binance.client.Client.ping", return_value={}), patch(
    "binance.client.Client.get_server_time",
    return_value={"serverTime": 0},
):
    import main


class NewsScanPrefetchWiringTests(unittest.TestCase):
    def test_process_ranked_entry_candidates_prefetches_news_before_executing(self):
        candidates = [
            {"symbol": "BTCUSDT", "rank_score": 10, "signal": "BUY"},
            {"symbol": "ETHUSDT", "rank_score": 5, "signal": "SELL"},
        ]
        call_order = []

        def fake_prepare_news(symbols):
            call_order.append(("news", list(symbols)))

        def fake_prefetch_llm(ranked):
            call_order.append(("llm", [c["symbol"] for c in ranked]))

        def fake_execute(candidate, trade_state, position_details, open_positions, *rest):
            call_order.append(("execute", candidate["symbol"]))
            return position_details, open_positions, True

        with patch.object(
            main, "prepare_news_scan_context", side_effect=fake_prepare_news
        ), patch.object(
            main, "prefetch_llm_candidate_reviews", side_effect=fake_prefetch_llm
        ), patch.object(
            main, "execute_entry_candidate", side_effect=fake_execute
        ), patch.object(
            config, "SIGNAL_RANKING_MAX_CANDIDATES", 0
        ):
            main.process_ranked_entry_candidates(
                candidates, {}, {}, {}, object(), object()
            )

        self.assertEqual(
            call_order,
            [
                ("news", ["BTCUSDT", "ETHUSDT"]),
                ("llm", ["BTCUSDT", "ETHUSDT"]),
                ("execute", "BTCUSDT"),
                ("execute", "ETHUSDT"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
