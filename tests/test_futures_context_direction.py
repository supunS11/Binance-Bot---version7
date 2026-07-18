import unittest

import strategy


class FuturesContextDirectionTests(unittest.TestCase):
    def test_price_up_and_oi_up_supports_buy_not_sell(self):
        buy = {"available": True, "oi_change_pct": 2.0}
        sell = {"available": True, "oi_change_pct": 2.0}

        buy_score = strategy._futures_participation_score("BUY", buy, 1.0)
        sell_score = strategy._futures_participation_score("SELL", sell, 1.0)

        self.assertGreater(buy_score, sell_score)
        self.assertEqual(buy["oi_price_state"], "LONG_BUILD")

    def test_price_down_and_oi_up_supports_sell_not_buy(self):
        buy = {"available": True, "oi_change_pct": 2.0}
        sell = {"available": True, "oi_change_pct": 2.0}

        buy_score = strategy._futures_participation_score("BUY", buy, -1.0)
        sell_score = strategy._futures_participation_score("SELL", sell, -1.0)

        self.assertGreater(sell_score, buy_score)
        self.assertEqual(sell["oi_price_state"], "SHORT_BUILD")

    def test_uncapped_index_preserves_order_above_100(self):
        self.assertEqual(strategy.score_to_confidence(50, 42), 100)
        self.assertGreater(strategy.score_to_uncapped_index(50, 42), 100)


if __name__ == "__main__":
    unittest.main()
