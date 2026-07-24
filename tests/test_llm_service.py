import copy
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import llm_service


def build_candidate(symbol="BTCUSDT", side="BUY"):
    opposite = "SELL" if side == "BUY" else "BUY"
    analysis = {
        "signal": side,
        "threshold": 74,
        "min_edge": 6,
        "best_confidence": 82,
        side.lower(): {
            "side": side,
            "confirmation_type": "TREND",
            "confidence": 82,
            "score": 12,
            "hard_ok": True,
            "trend_following_ok": True,
            "trend_ok": True,
            "confirm_ok": True,
            "entry_ok": True,
            "quality_score": 2,
            "regime_score": 1,
            "participation_score": 1,
            "regime_context": {"regime": "TREND"},
        },
        opposite.lower(): {
            "side": opposite,
            "confirmation_type": "NONE",
            "confidence": 35,
            "score": 3,
        },
    }
    return {
        "symbol": symbol,
        "signal": side,
        "analysis": analysis,
        "participation": {"available": True, "oi_change_pct": 1.2},
        "btc_trend": "BULLISH",
        "btc_corr": 0.72,
        "rs": 1.5,
        "news_context": {},
    }


class FakeRateLimitResponse:
    status_code = 429
    text = ""
    headers = {
        "retry-after": "2s",
        "x-ratelimit-reset-requests": "1m2s",
        "x-request-id": "request-test",
    }

    @staticmethod
    def json():
        return {
            "error": {
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
                "message": "Request rate limit reached",
            }
        }


class LlmServiceTests(unittest.TestCase):
    def test_duration_parser_handles_provider_reset_formats(self):
        self.assertAlmostEqual(llm_service._duration_seconds("250ms"), 0.25)
        self.assertEqual(llm_service._duration_seconds("1m2s"), 62)
        self.assertEqual(llm_service._duration_seconds("2h3m4s"), 7384)

    def test_shared_limiter_reserves_without_waiting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = Path(temp_dir) / "llm_limit.json"

            with patch.multiple(
                config,
                LLM_SHARED_RATE_LIMIT_PATH=str(state_path),
                LLM_MIN_REQUEST_INTERVAL_SECONDS=30,
                LLM_SHARED_LOCK_TIMEOUT_SECONDS=0.2,
            ):
                allowed, reason, _ = llm_service._reserve_shared_request()
                self.assertTrue(allowed)
                self.assertEqual(reason, "")

                allowed, reason, metadata = llm_service._reserve_shared_request()
                self.assertFalse(allowed)
                self.assertEqual(reason, "LLM_SHARED_REQUEST_INTERVAL")
                self.assertGreater(metadata["retry_after_seconds"], 0)

                llm_service._register_shared_backoff(
                    time.time() + 60,
                    "test"
                )
                allowed, reason, _ = llm_service._reserve_shared_request()
                self.assertFalse(allowed)
                self.assertEqual(reason, "LLM_SHARED_PROVIDER_BACKOFF")

    def test_429_uses_headers_and_is_not_retried_immediately(self):
        with patch.multiple(
            config,
            LLM_API_KEY="test-key",
            LLM_MODEL="test-model",
            LLM_MAX_RETRIES=3,
            LLM_RATE_LIMIT_BACKOFF_SECONDS=10,
            LLM_RATE_LIMIT_SAFETY_SECONDS=1,
        ), patch.object(
            llm_service,
            "_reserve_shared_request",
            return_value=(True, "", {})
        ), patch.object(
            llm_service.requests,
            "post",
            return_value=FakeRateLimitResponse()
        ) as post, patch.object(llm_service, "_register_shared_backoff"):
            review, reason, metadata = llm_service._request_openai_prompt(
                "system",
                "user"
            )

        self.assertIsNone(review)
        self.assertTrue(reason.startswith("LLM_RATE_LIMITED:"))
        self.assertEqual(metadata["request_reset_seconds"], 62)
        self.assertGreater(metadata["backoff_until"], time.time() + 60)
        self.assertEqual(post.call_count, 1)

    def test_quota_error_is_separate_from_temporary_rate_limit(self):
        metadata = {
            "error_type": "insufficient_quota",
            "error_code": "insufficient_quota",
            "error_message": "Check your plan and billing details",
        }

        with patch.object(config, "LLM_QUOTA_BACKOFF_SECONDS", 21600):
            self.assertTrue(llm_service._is_quota_error(metadata))
            self.assertEqual(llm_service._rate_limit_delay(metadata), 21600)

    def test_malformed_header_delay_is_capped_not_left_unbounded(self):
        metadata = {
            "retry_after_seconds": 0,
            "request_reset_seconds": 4_536_919,
            "token_reset_seconds": 0,
        }

        with patch.multiple(
            config,
            LLM_RATE_LIMIT_BACKOFF_SECONDS=900,
            LLM_RATE_LIMIT_SAFETY_SECONDS=1,
            LLM_MAX_BACKOFF_SECONDS=43200,
        ):
            delay = llm_service._rate_limit_delay(metadata)

        self.assertEqual(delay, 43200)

    def test_quota_delay_is_also_capped_by_max_backoff(self):
        metadata = {
            "error_type": "insufficient_quota",
            "error_code": "insufficient_quota",
            "error_message": "Check your plan and billing details",
        }

        with patch.multiple(
            config,
            LLM_QUOTA_BACKOFF_SECONDS=999_999,
            LLM_MAX_BACKOFF_SECONDS=43200,
        ):
            self.assertEqual(llm_service._rate_limit_delay(metadata), 43200)

    def test_local_backoff_until_caps_a_provided_metadata_value(self):
        now = time.time()

        with patch.object(config, "LLM_MAX_BACKOFF_SECONDS", 43200):
            capped = llm_service._local_backoff_until(
                now,
                "LLM_RATE_LIMITED",
                {"backoff_until": now + 4_536_919}
            )

        self.assertAlmostEqual(capped, now + 43200, delta=1)

    def test_local_backoff_until_stays_uncapped_for_normal_delays(self):
        now = time.time()

        with patch.multiple(
            config,
            LLM_RATE_LIMIT_BACKOFF_SECONDS=900,
            LLM_MAX_BACKOFF_SECONDS=43200,
        ):
            normal = llm_service._local_backoff_until(
                now,
                "LLM_RATE_LIMITED",
                {}
            )

        self.assertAlmostEqual(normal, now + 900, delta=1)

    def test_exhausted_success_headers_backoff_is_capped(self):
        with patch.object(config, "LLM_MAX_BACKOFF_SECONDS", 43200), patch.object(
            config, "LLM_RATE_LIMIT_SAFETY_SECONDS", 1
        ), patch.object(llm_service, "_register_shared_backoff") as register:
            llm_service._register_exhausted_success_headers({
                "remaining_requests": 0,
                "remaining_tokens": 5,
                "request_reset_seconds": 4_536_919,
                "token_reset_seconds": 0,
            })

        self.assertEqual(register.call_count, 1)
        backoff_until = register.call_args.args[0]
        self.assertLessEqual(backoff_until, time.time() + 43200 + 1)

    def test_batch_review_is_reused_by_entry_filter_and_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = Path(temp_dir) / "llm_cache.json"
            candidate = build_candidate()
            second_candidate = build_candidate("ETHUSDT", "SELL")

            def fake_batch(payloads):
                return (
                    {
                        "reviews": [
                            {
                                "symbol": "BTCUSDT",
                                "side": "BUY",
                                "action": "ALLOW",
                                "confidence_adjustment": 0,
                                "risk_label": "low",
                                "reason": "Conditions agree",
                            },
                            {
                                "symbol": "ETHUSDT",
                                "side": "SELL",
                                "action": "PENALTY",
                                "confidence_adjustment": 2,
                                "risk_label": "medium",
                                "reason": "Flow is mildly conflicting",
                            }
                        ]
                    },
                    payloads,
                    "",
                    {},
                )

            with patch.multiple(
                config,
                LLM_FILTER_ENABLED=True,
                LLM_BATCH_ENABLED=True,
                LLM_BATCH_MAX_CANDIDATES=20,
                LLM_BATCH_MAX_PROMPT_CHARS=30000,
                LLM_CACHE_PATH=str(cache_path),
                LLM_CACHE_SECONDS=3600,
                LLM_SYMBOL_SIDE_CACHE_SECONDS=3600,
                LLM_MAX_REQUESTS_PER_SCAN=1,
            ), patch.object(
                llm_service,
                "_request_openai_batch",
                side_effect=fake_batch
            ) as batch_request:
                llm_service.begin_llm_scan_budget()
                prepared = llm_service.prefetch_llm_candidate_reviews(
                    [candidate, second_candidate]
                )
                self.assertEqual(prepared, 2)
                self.assertEqual(batch_request.call_count, 1)
                self.assertEqual(
                    second_candidate["llm_prefetched_review"]["action"],
                    "PENALTY"
                )

                with patch.object(
                    llm_service,
                    "_get_review",
                    side_effect=AssertionError("unexpected individual request")
                ):
                    allowed, _, context = llm_service.apply_llm_filter(
                        candidate["symbol"],
                        candidate["signal"],
                        candidate["analysis"],
                        participation=candidate["participation"],
                        btc_trend=candidate["btc_trend"],
                        btc_corr=candidate["btc_corr"],
                        rs=candidate["rs"],
                        news_context={},
                        prefetched_review=candidate["llm_prefetched_review"],
                        prefetched_source=candidate["llm_prefetched_source"],
                    )

                self.assertTrue(allowed)
                self.assertEqual(context["source"], "batch")

                cached_candidate = copy.deepcopy(build_candidate())
                llm_service.begin_llm_scan_budget()
                prepared = llm_service.prefetch_llm_candidate_reviews(
                    [cached_candidate]
                )
                self.assertEqual(prepared, 1)
                self.assertEqual(batch_request.call_count, 1)
                self.assertIn(
                    cached_candidate["llm_prefetched_source"],
                    {"cache", "symbol_condition_cache"}
                )


if __name__ == "__main__":
    unittest.main()
