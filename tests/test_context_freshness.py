import unittest
from unittest.mock import Mock, patch

import config
import llm_service
import news_service


class ExternalContextFreshnessTests(unittest.TestCase):
    def test_stale_news_cannot_block_a_trade(self):
        analysis = {
            "buy": {"confidence": 90},
            "sell": {"confidence": 10},
        }
        stale = {
            "enabled": True,
            "available": True,
            "cache_status": "stale",
            "score": -1,
            "high_impact": True,
        }

        with patch.object(
            news_service,
            "get_news_context",
            return_value=stale,
        ):
            allowed, adjusted, context = news_service.apply_news_filter(
                "BTCUSDT",
                "BUY",
                analysis,
            )

        self.assertTrue(allowed)
        self.assertIs(adjusted, analysis)
        self.assertEqual(context["action"], "ALLOW")

    def test_stale_llm_block_is_advisory_only(self):
        analysis = {
            "buy": {"confidence": 90},
            "sell": {"confidence": 10},
        }
        review = {
            "action": "BLOCK",
            "confidence_adjustment": -10,
            "risk_label": "high",
            "reason": "old risk assessment",
        }

        with patch.object(config, "LLM_FILTER_ENABLED", True):
            allowed, adjusted, context = llm_service.apply_llm_filter(
                "BTCUSDT",
                "BUY",
                analysis,
                prefetched_review=review,
                prefetched_source="stale_cache",
            )

        self.assertTrue(allowed)
        self.assertIs(adjusted, analysis)
        self.assertEqual(context["action"], "ALLOW")
        self.assertEqual(context["confidence_adjustment"], 0)

    def test_cryptocompare_key_is_sent_only_in_header(self):
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"Data": []}

        with patch.object(config, "NEWS_API_KEY", "secret-test-key"), patch.object(
            news_service.requests,
            "get",
            return_value=response,
        ) as request:
            news_service._cryptocompare_provider_items({}, 1)

        self.assertNotIn("api_key", request.call_args.kwargs["params"])
        self.assertIn(
            "secret-test-key",
            request.call_args.kwargs["headers"]["authorization"],
        )


if __name__ == "__main__":
    unittest.main()
