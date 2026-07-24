import json
import os
import random
import re
import time
from contextlib import contextmanager
from copy import deepcopy
from email.utils import parsedate_to_datetime
from pathlib import Path

try:
    import requests
except ImportError:
    requests = None

import config
from logger import log_error, log_info, log_warning


_scan_request_count = 0


VALID_ACTIONS = {"ALLOW", "BOOST", "PENALTY", "BLOCK"}
VALID_RISK_LABELS = {"low", "medium", "high"}


def _shared_limit_path():
    path = Path(config.LLM_SHARED_RATE_LIMIT_PATH).expanduser()

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


@contextmanager
def _exclusive_file_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    deadline = time.monotonic() + max(
        float(config.LLM_SHARED_LOCK_TIMEOUT_SECONDS),
        0
    )
    locked = False

    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)

            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()

            while not locked:
                try:
                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                    locked = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("LLM_SHARED_LIMITER_LOCK_TIMEOUT")
                    time.sleep(0.05)
        else:
            import fcntl

            while not locked:
                try:
                    fcntl.flock(
                        handle.fileno(),
                        fcntl.LOCK_EX | fcntl.LOCK_NB
                    )
                    locked = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError("LLM_SHARED_LIMITER_LOCK_TIMEOUT")
                    time.sleep(0.05)

        yield
    finally:
        if locked:
            try:
                if os.name == "nt":
                    import msvcrt

                    handle.seek(0)
                    msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass

        handle.close()


def _load_shared_limit_state(path):
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _save_shared_limit_state(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True, default=str)


def _reserve_shared_request():
    path = _shared_limit_path()
    lock_path = Path(f"{path}.lock")
    now = time.time()

    try:
        with _exclusive_file_lock(lock_path):
            state = _load_shared_limit_state(path)
            backoff_until = float(state.get("backoff_until", 0) or 0)

            if backoff_until > now:
                return False, "LLM_SHARED_PROVIDER_BACKOFF", {
                    "backoff_until": backoff_until,
                    "retry_after_seconds": round(backoff_until - now, 3),
                }

            next_request_at = float(state.get("next_request_at", 0) or 0)

            if next_request_at > now:
                return False, "LLM_SHARED_REQUEST_INTERVAL", {
                    "retry_after_seconds": round(next_request_at - now, 3),
                }

            state["next_request_at"] = (
                now + max(config.LLM_MIN_REQUEST_INTERVAL_SECONDS, 0)
            )
            state["updated_at"] = now
            _save_shared_limit_state(path, state)
            return True, "", {}
    except TimeoutError:
        return False, "LLM_SHARED_LIMITER_BUSY", {}
    except Exception as e:
        log_warning(f"LLM shared limiter unavailable: {_limit_text(e, 160)}")
        return False, "LLM_SHARED_LIMITER_ERROR", {}


def _register_shared_backoff(backoff_until, reason):
    path = _shared_limit_path()
    lock_path = Path(f"{path}.lock")

    try:
        with _exclusive_file_lock(lock_path):
            state = _load_shared_limit_state(path)
            existing = float(state.get("backoff_until", 0) or 0)
            state["backoff_until"] = max(existing, float(backoff_until or 0))
            state["backoff_reason"] = _limit_text(reason, 160)
            state["updated_at"] = time.time()
            _save_shared_limit_state(path, state)
    except Exception as e:
        log_warning(f"LLM shared backoff save error: {_limit_text(e, 160)}")


def _cache_path():
    path = Path(config.LLM_CACHE_PATH)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _load_cache():
    path = _cache_path()

    if not path.exists():
        return {
            "items": {},
            "latest_by_symbol_side": {},
            "provider_backoff_until": 0
        }

    try:
        with path.open("r", encoding="utf-8") as file:
            cache = json.load(file)

        if "items" not in cache:
            cache["items"] = {}

        if "provider_backoff_until" not in cache:
            cache["provider_backoff_until"] = 0

        if "latest_by_symbol_side" not in cache:
            cache["latest_by_symbol_side"] = {}

        return cache

    except Exception as e:
        log_error(f"llm cache load error: {e}")
        return {
            "items": {},
            "latest_by_symbol_side": {},
            "provider_backoff_until": 0
        }


def _save_cache(cache):
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(cache, file, indent=2, sort_keys=True, default=str)

    except Exception as e:
        log_error(f"llm cache save error: {e}")


def _empty_context(symbol, reason, enabled=None):
    return {
        "enabled": config.LLM_FILTER_ENABLED if enabled is None else enabled,
        "available": False,
        "symbol": symbol,
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "action": "DISABLED" if enabled is False else "ALLOW",
        "raw_action": "",
        "confidence_adjustment": 0,
        "risk_label": "",
        "reason": reason,
    }


def _symbol_side_key(payload):
    selected = payload.get("selected_side", {})
    quality = selected.get("quality", {})
    scores = selected.get("scores", {})
    news = payload.get("news_context", {})
    bucket_size = max(float(config.LLM_CONFIDENCE_BUCKET_SIZE), 1)

    try:
        confidence = float(selected.get("confidence", 0) or 0)
        confidence_bucket = int(confidence // bucket_size) * bucket_size
    except Exception:
        confidence_bucket = 0

    try:
        futures_score = float(scores.get("futures", 0) or 0)
    except Exception:
        futures_score = 0

    if futures_score > 0:
        futures_bias = "supportive"
    elif futures_score < 0:
        futures_bias = "conflicting"
    else:
        futures_bias = "neutral"

    return json.dumps(
        {
            "symbol": payload.get("symbol"),
            "side": payload.get("proposed_signal"),
            "confirmation_type": selected.get("confirmation_type"),
            "confidence_bucket": confidence_bucket,
            "regime": quality.get("regime"),
            "futures_bias": futures_bias,
            "news_action": news.get("action") or "",
            "model": config.LLM_MODEL,
        },
        sort_keys=True
    )


def _cached_review(cache, key, now, max_age):
    cached = cache.get("items", {}).get(key)

    if not cached:
        return None, ""

    fetched_at = float(cached.get("fetched_at", 0) or 0)
    age = max(now - fetched_at, 0)

    if max_age is not None and age > max_age:
        return None, ""

    source = "cache" if age <= config.LLM_CACHE_SECONDS else "stale_cache"
    return cached.get("review") or {}, source


def _latest_symbol_side_review(
    cache,
    payload,
    now,
    max_age=None,
    source="symbol_side_stale_cache"
):
    key = _symbol_side_key(payload)
    cached = cache.get("latest_by_symbol_side", {}).get(key)

    if not cached:
        return None, ""

    fetched_at = float(cached.get("fetched_at", 0) or 0)
    age = max(now - fetched_at, 0)

    if max_age is None:
        max_age = config.LLM_STALE_CACHE_SECONDS

    if age > max_age:
        return None, ""

    return cached.get("review") or {}, source


def begin_llm_scan_budget():
    global _scan_request_count

    _scan_request_count = 0


def _claim_scan_request():
    global _scan_request_count

    if (
        config.LLM_MAX_REQUESTS_PER_SCAN > 0 and
        _scan_request_count >= config.LLM_MAX_REQUESTS_PER_SCAN
    ):
        return False

    _scan_request_count += 1
    return True


def _round_value(value, digits=4):
    try:
        return round(float(value), digits)
    except Exception:
        return value if value not in (None, "") else ""


def _limit_text(value, max_len=180):
    text = str(value or "").strip()

    if len(text) <= max_len:
        return text

    return text[: max_len - 3] + "..."


def _safe_bool(value):
    if value in ("", None):
        return False

    return bool(value)


def _requires_provider_backoff(reason):
    text = str(reason or "").upper()
    return text.startswith(
        (
            "LLM_RATE_LIMITED",
            "LLM_QUOTA_EXCEEDED",
            "LLM_SHARED_PROVIDER_BACKOFF",
        )
    )


def _backoff_ceiling_seconds():
    return max(float(getattr(config, "LLM_MAX_BACKOFF_SECONDS", 43200)), 1)


def _capped_backoff_until(now, candidate_until):
    ceiling = now + _backoff_ceiling_seconds()
    return min(float(candidate_until or 0), ceiling)


def _local_backoff_until(now, reason, metadata):
    metadata = metadata or {}
    provided = float(metadata.get("backoff_until", 0) or 0)

    if provided > now:
        return _capped_backoff_until(now, provided)

    if str(reason or "").upper().startswith("LLM_QUOTA_EXCEEDED"):
        return _capped_backoff_until(
            now,
            now + max(config.LLM_QUOTA_BACKOFF_SECONDS, 1)
        )

    return _capped_backoff_until(
        now,
        now + max(config.LLM_RATE_LIMIT_BACKOFF_SECONDS, 1)
    )


def _compact_level(side_data):
    level = side_data.get("level") if side_data else None

    if not isinstance(level, dict):
        return {}

    return {
        "ok": _safe_bool(side_data.get("level_ok")),
        "price": _round_value(level.get("level")),
        "adverse_roi": _round_value(level.get("adverse_roi"), 2),
        "score": _round_value(level.get("score"), 2),
        "source": level.get("source") or "",
        "reason": level.get("reason") or "",
    }


def _compact_quality(side_data):
    entry_quality = side_data.get("entry_quality") or {}
    confirm_quality = side_data.get("confirm_quality") or {}
    regime_context = side_data.get("regime_context") or {}

    return {
        "quality_score": _round_value(side_data.get("quality_score"), 2),
        "regime": regime_context.get("regime") or "",
        "regime_score": _round_value(side_data.get("regime_score"), 2),
        "entry_quality_ok": _safe_bool(entry_quality.get("quality_ok")),
        "entry_chase_atr": _round_value(entry_quality.get("chase_atr"), 3),
        "entry_volume_mult": _round_value(entry_quality.get("volume_mult"), 2),
        "entry_rejection_wick": _round_value(
            entry_quality.get("rejection_wick_ratio"),
            3
        ),
        "confirm_quality_ok": _safe_bool(confirm_quality.get("quality_ok")),
        "confirm_volume_mult": _round_value(
            confirm_quality.get("volume_mult"),
            2
        ),
    }


def _compact_side(side_data):
    side_data = side_data or {}
    timing_rescue = side_data.get("trend_timing_rescue") or {}
    continuation_pullback = side_data.get("continuation_pullback") or {}
    reversal_futures = (
        (side_data.get("reversal_context") or {}).get(
            "futures_confirmation",
            {},
        )
    )

    return {
        "side": side_data.get("side") or "",
        "confirmation_type": side_data.get("confirmation_type") or "NONE",
        "confidence": _round_value(side_data.get("confidence"), 2),
        "score": _round_value(side_data.get("score"), 2),
        "hard_ok": _safe_bool(side_data.get("hard_ok")),
        "trend_following_ok": _safe_bool(side_data.get("trend_following_ok")),
        "reversal_ok": _safe_bool(side_data.get("reversal_ok")),
        "trend_ok": _safe_bool(side_data.get("trend_ok")),
        "confirm_ok": _safe_bool(side_data.get("confirm_ok")),
        "entry_ok": _safe_bool(side_data.get("entry_ok")),
        "trend_timing_rescue": {
            "active": _safe_bool(timing_rescue.get("active")),
            "missed_module": timing_rescue.get("missed_module") or "",
            "futures_score": _round_value(
                timing_rescue.get("futures_score"),
                2
            ),
        },
        "continuation_pullback": {
            "active": _safe_bool(continuation_pullback.get("active")),
            "ema20_distance_atr": _round_value(
                continuation_pullback.get("ema20_distance_atr"),
                2
            ),
            "futures_score": _round_value(
                continuation_pullback.get("futures_score"),
                2
            ),
        },
        "reversal_futures": {
            "required": _safe_bool(reversal_futures.get("required")),
            "active": _safe_bool(reversal_futures.get("active")),
            "available": _safe_bool(reversal_futures.get("available")),
            "score": _round_value(reversal_futures.get("score"), 2),
            "minimum": _round_value(reversal_futures.get("minimum"), 2),
        },
        "level": _compact_level(side_data),
        "quality": _compact_quality(side_data),
        "scores": {
            "trend": _round_value(side_data.get("trend_score"), 2),
            "confirm": _round_value(side_data.get("confirm_score"), 2),
            "entry": _round_value(side_data.get("entry_score"), 2),
            "btc": _round_value(side_data.get("btc_score"), 2),
            "smc": _round_value(side_data.get("smc_score"), 2),
            "futures": _round_value(side_data.get("participation_score"), 2),
            "regime": _round_value(side_data.get("regime_score"), 2),
        },
    }


def _compact_news(news_context):
    if not news_context:
        return {}

    return {
        "available": _safe_bool(news_context.get("available")),
        "label": news_context.get("label") or "",
        "score": _round_value(news_context.get("score"), 3),
        "action": news_context.get("action") or "",
        "reason": news_context.get("reason") or "",
        "headline": _limit_text(news_context.get("headline"), 220),
        "source": news_context.get("source") or "",
        "high_impact": _safe_bool(news_context.get("high_impact")),
    }


def _compact_participation(participation):
    if not participation:
        return {}

    return {
        "available": _safe_bool(participation.get("available")),
        "oi_change_pct": _round_value(participation.get("oi_change_pct"), 3),
        "taker_buy_sell_ratio": _round_value(
            participation.get("taker_buy_sell_ratio"),
            3
        ),
        "global_long_short_ratio": _round_value(
            participation.get("global_long_short_ratio"),
            3
        ),
        "top_long_short_ratio": _round_value(
            participation.get("top_long_short_ratio"),
            3
        ),
        "funding_rate": _round_value(participation.get("funding_rate"), 6),
    }


def _build_payload(
    symbol,
    side,
    analysis,
    participation,
    btc_trend,
    btc_corr,
    rs,
    news_context
):
    side_key = (side or "").lower()

    return {
        "symbol": symbol,
        "proposed_signal": side,
        "selected_side": _compact_side(analysis.get(side_key, {})),
        "opposite_side": _compact_side(
            analysis.get("sell" if side_key == "buy" else "buy", {})
        ),
        "threshold": _round_value(analysis.get("threshold"), 2),
        "min_edge": _round_value(analysis.get("min_edge"), 2),
        "best_confidence": _round_value(analysis.get("best_confidence"), 2),
        "btc": {
            "trend": btc_trend,
            "correlation": _round_value(btc_corr, 3),
            "relative_strength_pct": _round_value(rs, 2),
        },
        "futures_context": _compact_participation(participation),
        "news_context": _compact_news(news_context),
    }


def _cache_key(payload):
    selected = payload.get("selected_side", {})
    quality = selected.get("quality", {})
    level = selected.get("level", {})
    news = payload.get("news_context", {})
    futures = payload.get("futures_context", {})
    key_payload = {
        "symbol": payload.get("symbol"),
        "side": payload.get("proposed_signal"),
        "confidence": selected.get("confidence"),
        "opposite_confidence": payload.get("opposite_side", {}).get("confidence"),
        "quality_score": quality.get("quality_score"),
        "regime": quality.get("regime"),
        "level_source": level.get("source"),
        "news_score": news.get("score"),
        "news_action": news.get("action"),
        "oi": futures.get("oi_change_pct"),
        "taker": futures.get("taker_buy_sell_ratio"),
        "model": config.LLM_MODEL,
    }
    return json.dumps(key_payload, sort_keys=True)


def _system_prompt():
    return (
        "You are a risk review layer for an automated Binance futures bot. "
        "The deterministic strategy already selected a signal. You cannot "
        "create a new trade or flip the side. Review only whether the proposed "
        "signal has clear conflict, late-entry risk, weak confirmation, news "
        "risk, or futures-flow risk. Return only valid JSON with keys: "
        "action, confidence_adjustment, risk_label, reason. action must be "
        "ALLOW, BOOST, PENALTY, or BLOCK. risk_label must be low, medium, or "
        "high. Keep reason under 160 characters."
    )


def _user_prompt(payload):
    compact = deepcopy(payload)
    text = json.dumps(compact, sort_keys=True, default=str)

    if len(text) > config.LLM_MAX_PROMPT_CHARS:
        compact.get("news_context", {}).pop("headline", None)
        compact.get("news_context", {}).pop("reason", None)

        for side_key in ("selected_side", "opposite_side"):
            compact.get(side_key, {}).get("level", {}).pop("reason", None)

        text = json.dumps(compact, sort_keys=True, default=str)

    if len(text) > config.LLM_MAX_PROMPT_CHARS:
        text = json.dumps(
            {
                "symbol": payload.get("symbol"),
                "proposed_signal": payload.get("proposed_signal"),
                "selected_side": payload.get("selected_side"),
                "opposite_side": payload.get("opposite_side"),
                "btc": payload.get("btc"),
                "futures_context": payload.get("futures_context"),
                "news_context": payload.get("news_context"),
            },
            sort_keys=True,
            default=str
        )

    return (
        "Review this proposed futures signal. Be conservative with BLOCK; use "
        "BLOCK only for obvious contradictions or high-risk context. JSON "
        f"payload:\n{text}"
    )


def _parse_json_content(content):
    text = str(content or "").strip()

    if text.startswith("```"):
        text = text.strip("`").strip()

        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start >= 0 and end > start:
        text = text[start:end + 1]

    return json.loads(text)


def _duration_seconds(value):
    if value in (None, ""):
        return 0

    if isinstance(value, (int, float)):
        return max(float(value), 0)

    text = str(value).strip().lower()

    try:
        return max(float(text), 0)
    except ValueError:
        pass

    units = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}
    matches = re.findall(r"(\d+(?:\.\d+)?)(ms|s|m|h)", text)

    if matches:
        return sum(float(amount) * units[unit] for amount, unit in matches)

    try:
        parsed = parsedate_to_datetime(str(value))
        return max(parsed.timestamp() - time.time(), 0)
    except Exception:
        return 0


def _header_float(headers, name):
    try:
        value = headers.get(name)
        return float(value) if value not in (None, "") else None
    except Exception:
        return None


def _response_metadata(response):
    headers = {
        str(key).lower(): value
        for key, value in dict(getattr(response, "headers", {}) or {}).items()
    }
    error_type = ""
    error_code = ""
    error_message = ""

    try:
        data = response.json()
        error = data.get("error", {}) if isinstance(data, dict) else {}

        if isinstance(error, dict):
            error_type = str(error.get("type") or "")
            error_code = str(error.get("code") or "")
            error_message = str(error.get("message") or "")
    except Exception:
        data = None

    if not error_message:
        error_message = str(getattr(response, "text", "") or "")

    return {
        "status_code": int(getattr(response, "status_code", 0) or 0),
        "request_id": str(headers.get("x-request-id") or ""),
        "error_type": error_type,
        "error_code": error_code,
        "error_message": _limit_text(error_message, 240),
        "retry_after_seconds": _duration_seconds(headers.get("retry-after")),
        "request_reset_seconds": _duration_seconds(
            headers.get("x-ratelimit-reset-requests")
        ),
        "token_reset_seconds": _duration_seconds(
            headers.get("x-ratelimit-reset-tokens")
        ),
        "remaining_requests": _header_float(
            headers,
            "x-ratelimit-remaining-requests"
        ),
        "remaining_tokens": _header_float(
            headers,
            "x-ratelimit-remaining-tokens"
        ),
    }


def _is_quota_error(metadata):
    text = " ".join(
        str(metadata.get(key) or "")
        for key in ("error_type", "error_code", "error_message")
    ).lower()
    return any(
        marker in text
        for marker in (
            "insufficient_quota",
            "billing",
            "credit balance",
            "credits exhausted",
            "check your plan",
        )
    )


def _rate_limit_delay(metadata):
    ceiling = _backoff_ceiling_seconds()

    if _is_quota_error(metadata):
        return min(max(float(config.LLM_QUOTA_BACKOFF_SECONDS), 1), ceiling)

    header_delay = max(
        float(metadata.get("retry_after_seconds", 0) or 0),
        float(metadata.get("request_reset_seconds", 0) or 0),
        float(metadata.get("token_reset_seconds", 0) or 0),
    )
    configured = max(float(config.LLM_RATE_LIMIT_BACKOFF_SECONDS), 1)
    delay = max(header_delay, configured) + max(
        float(config.LLM_RATE_LIMIT_SAFETY_SECONDS),
        0
    )
    return min(delay, ceiling)


def _register_exhausted_success_headers(metadata):
    delays = []

    if metadata.get("remaining_requests") == 0:
        delays.append(float(metadata.get("request_reset_seconds", 0) or 0))

    if metadata.get("remaining_tokens") == 0:
        delays.append(float(metadata.get("token_reset_seconds", 0) or 0))

    delay = max(delays or [0])

    if delay > 0:
        delay += max(float(config.LLM_RATE_LIMIT_SAFETY_SECONDS), 0)
        delay = min(delay, _backoff_ceiling_seconds())
        _register_shared_backoff(
            time.time() + delay,
            "LLM_RESPONSE_BUDGET_EXHAUSTED"
        )


def _retry_delay(attempt):
    base = max(float(config.LLM_RETRY_BASE_SECONDS), 0.1)
    maximum = max(float(config.LLM_RETRY_MAX_SECONDS), base)
    exponential = min(base * (2 ** attempt), maximum)
    return max(
        exponential * random.uniform(0.75, 1.25),
        float(config.LLM_MIN_REQUEST_INTERVAL_SECONDS)
    )


def _request_openai_prompt(system_prompt, user_prompt):
    if requests is None:
        return None, "LLM_REQUESTS_PACKAGE_MISSING", {}

    if not config.LLM_API_KEY:
        return None, "LLM_API_KEY_MISSING", {}

    if not config.LLM_MODEL:
        return None, "LLM_MODEL_MISSING", {}

    headers = {
        "Authorization": f"Bearer {config.LLM_API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": config.LLM_MODEL,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }

    last_error = "LLM_REQUEST_FAILED"
    attempts = max(config.LLM_MAX_RETRIES, 1)

    for attempt in range(attempts):
        allowed, limiter_reason, limiter_metadata = _reserve_shared_request()

        if not allowed:
            return None, limiter_reason, limiter_metadata

        try:
            response = requests.post(
                config.LLM_BASE_URL,
                headers=headers,
                json=body,
                timeout=config.LLM_TIMEOUT_SECONDS,
            )
            metadata = _response_metadata(response)
            status_code = metadata["status_code"]

            if status_code == 429:
                delay = _rate_limit_delay(metadata)
                metadata["backoff_until"] = time.time() + delay
                reason = (
                    "LLM_QUOTA_EXCEEDED"
                    if _is_quota_error(metadata)
                    else "LLM_RATE_LIMITED"
                )
                _register_shared_backoff(metadata["backoff_until"], reason)
                return None, (
                    f"{reason}:{metadata.get('error_message') or 'HTTP_429'}"
                ), metadata

            if status_code >= 400:
                last_error = (
                    f"LLM_HTTP_{status_code}:"
                    f"{metadata.get('error_message') or 'REQUEST_FAILED'}"
                )

                if status_code < 500 or attempt + 1 >= attempts:
                    return None, last_error, metadata

                time.sleep(_retry_delay(attempt))
                continue

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            _register_exhausted_success_headers(metadata)
            return _parse_json_content(content), "", metadata

        except Exception as e:
            last_error = f"LLM_REQUEST_ERROR:{_limit_text(e, 240)}"

            if attempt + 1 < attempts:
                time.sleep(_retry_delay(attempt))

    return None, last_error, {}


def _request_openai_compatible(payload):
    return _request_openai_prompt(_system_prompt(), _user_prompt(payload))


def _normalise_review(review):
    if not isinstance(review, dict):
        review = {}

    action = str(review.get("action") or "ALLOW").strip().upper()

    if action not in VALID_ACTIONS:
        action = "ALLOW"

    risk_label = str(review.get("risk_label") or "medium").strip().lower()

    if risk_label not in VALID_RISK_LABELS:
        risk_label = "medium"

    try:
        raw_adjustment = float(review.get("confidence_adjustment") or 0)
    except Exception:
        raw_adjustment = 0

    if action == "BOOST":
        delta = min(abs(raw_adjustment) or config.LLM_CONFIDENCE_BOOST,
                    config.LLM_CONFIDENCE_BOOST)
    elif action == "PENALTY":
        delta = -min(abs(raw_adjustment) or config.LLM_CONFIDENCE_PENALTY,
                     config.LLM_CONFIDENCE_PENALTY)
    else:
        delta = 0

    return {
        "action": action,
        "confidence_adjustment": round(delta, 2),
        "risk_label": risk_label,
        "reason": _limit_text(review.get("reason") or "LLM_REVIEW_READY", 180),
    }


def _store_review(cache, payload, review, fetched_at):
    cache.setdefault("items", {})[_cache_key(payload)] = {
        "fetched_at": fetched_at,
        "review": review,
    }
    cache.setdefault("latest_by_symbol_side", {})[
        _symbol_side_key(payload)
    ] = {
        "fetched_at": fetched_at,
        "review": review,
    }


def _get_review(payload):
    cache = _load_cache()
    now = time.time()
    key = _cache_key(payload)
    review, source = _cached_review(cache, key, now, config.LLM_CACHE_SECONDS)

    if review is not None:
        return review, source, ""

    review, source = _latest_symbol_side_review(
        cache,
        payload,
        now,
        max_age=config.LLM_SYMBOL_SIDE_CACHE_SECONDS,
        source="symbol_condition_cache"
    )

    if review is not None:
        return review, source, ""

    provider = config.LLM_PROVIDER
    backoff_until = float(cache.get("provider_backoff_until", 0) or 0)

    if backoff_until > now:
        review, source = _cached_review(
            cache,
            key,
            now,
            config.LLM_STALE_CACHE_SECONDS
        )

        if review is not None:
            return review, source, ""

        review, source = _latest_symbol_side_review(cache, payload, now)

        if review is not None:
            return review, source, ""

        return None, provider, "LLM_PROVIDER_RATE_LIMIT_BACKOFF"

    if provider in ("openai", "openai-compatible", "compatible"):
        if not _claim_scan_request():
            review, source = _cached_review(
                cache,
                key,
                now,
                config.LLM_STALE_CACHE_SECONDS
            )

            if review is not None:
                return review, source, ""

            review, source = _latest_symbol_side_review(cache, payload, now)

            if review is not None:
                return review, source, ""

            return None, provider, "LLM_SCAN_REQUEST_LIMIT_REACHED"

        review, reason, metadata = _request_openai_compatible(payload)
    else:
        review = None
        reason = f"LLM_PROVIDER_UNSUPPORTED:{provider}"
        metadata = {}

    if review is not None:
        cache["provider_backoff_until"] = 0
        _store_review(cache, payload, review, now)
        _save_cache(cache)
    elif _requires_provider_backoff(reason):
        backoff_until = _local_backoff_until(now, reason, metadata)
        cache["provider_backoff_until"] = max(
            float(cache.get("provider_backoff_until", 0) or 0),
            backoff_until
        )
        _save_cache(cache)
        log_warning(
            f"LLM provider rate limited | "
            f"backoff={round(max(backoff_until - now, 0), 1)}s"
        )

    return review, provider, reason


def _batch_system_prompt():
    return (
        "You are a risk review layer for ranked Binance futures signals. "
        "The deterministic strategy selected each side. Do not create a new "
        "trade or flip a side. Review conflicts, late-entry risk, weak "
        "confirmation, and futures-flow risk. Return only a JSON object with "
        "a reviews array. Every review must contain symbol, side, action, "
        "confidence_adjustment, risk_label, and reason. action must be ALLOW, "
        "BOOST, PENALTY, or BLOCK. risk_label must be low, medium, or high. "
        "Keep each reason under 160 characters and be conservative with BLOCK."
    )


def _fit_batch_payloads(payloads):
    selected = []
    max_candidates = max(int(config.LLM_BATCH_MAX_CANDIDATES), 1)
    max_chars = max(int(config.LLM_BATCH_MAX_PROMPT_CHARS), 1000)

    for payload in payloads[:max_candidates]:
        proposed = selected + [payload]
        text = json.dumps(
            {"candidates": proposed},
            sort_keys=True,
            default=str
        )

        if len(text) > max_chars:
            break

        selected = proposed

    return selected


def _request_openai_batch(payloads):
    selected = _fit_batch_payloads(payloads)

    if not selected:
        return None, [], "LLM_BATCH_PROMPT_LIMIT_REACHED", {}

    prompt = (
        "Review all proposed signals and return one review for every candidate. "
        "News is enforced separately by deterministic code. JSON payload:\n"
        + json.dumps(
            {"candidates": selected},
            sort_keys=True,
            default=str
        )
    )
    response, reason, metadata = _request_openai_prompt(
        _batch_system_prompt(),
        prompt
    )
    return response, selected, reason, metadata


def _attach_prefetched_review(candidate, review, source):
    candidate["llm_prefetched_review"] = review
    candidate["llm_prefetched_source"] = source


def prefetch_llm_candidate_reviews(candidates):
    if not config.LLM_FILTER_ENABLED or not config.LLM_BATCH_ENABLED:
        return 0

    cache = _load_cache()
    now = time.time()
    misses = []
    prepared = 0

    for candidate in candidates[:max(config.LLM_BATCH_MAX_CANDIDATES, 1)]:
        side = candidate.get("signal")

        if not side:
            continue

        payload = _build_payload(
            candidate.get("symbol"),
            side,
            candidate.get("analysis") or {},
            candidate.get("participation"),
            candidate.get("btc_trend"),
            candidate.get("btc_corr"),
            candidate.get("rs"),
            candidate.get("news_context") or {}
        )
        review, source = _cached_review(
            cache,
            _cache_key(payload),
            now,
            config.LLM_CACHE_SECONDS
        )

        if review is None:
            review, source = _latest_symbol_side_review(
                cache,
                payload,
                now,
                max_age=config.LLM_SYMBOL_SIDE_CACHE_SECONDS,
                source="symbol_condition_cache"
            )

        if review is not None:
            _attach_prefetched_review(candidate, review, source)
            prepared += 1
        else:
            misses.append((candidate, payload))

    if not misses:
        if prepared:
            log_info(f"LLM batch cache ready | candidates={prepared}")
        return prepared

    backoff_until = float(cache.get("provider_backoff_until", 0) or 0)

    if backoff_until > now:
        for candidate, payload in misses:
            review, source = _cached_review(
                cache,
                _cache_key(payload),
                now,
                config.LLM_STALE_CACHE_SECONDS
            )

            if review is None:
                review, source = _latest_symbol_side_review(
                    cache,
                    payload,
                    now
                )

            if review is not None:
                _attach_prefetched_review(candidate, review, source)
                prepared += 1

        log_info(
            f"LLM batch skipped | RATE_LIMIT_BACKOFF | cached={prepared}"
        )
        return prepared

    if config.LLM_PROVIDER not in ("openai", "openai-compatible", "compatible"):
        log_warning(
            f"LLM batch unavailable | provider={config.LLM_PROVIDER}"
        )
        return prepared

    if not _claim_scan_request():
        log_info(f"LLM batch skipped | SCAN_REQUEST_LIMIT | cached={prepared}")
        return prepared

    response, included, reason, metadata = _request_openai_batch(
        [payload for _, payload in misses]
    )

    if response is None:
        if _requires_provider_backoff(reason):
            backoff_until = _local_backoff_until(now, reason, metadata)
            cache["provider_backoff_until"] = max(
                float(cache.get("provider_backoff_until", 0) or 0),
                backoff_until
            )
            _save_cache(cache)
            log_warning(
                f"LLM batch rate limited | "
                f"backoff={round(max(backoff_until - now, 0), 1)}s"
            )
        elif str(reason or "").startswith("LLM_SHARED_"):
            log_info(f"LLM batch skipped | {reason}")
        else:
            log_warning(f"LLM batch unavailable | {reason}")
        return prepared

    raw_reviews = response.get("reviews", []) if isinstance(response, dict) else []
    review_map = {}

    for raw_review in raw_reviews:
        if not isinstance(raw_review, dict):
            continue

        review_key = (
            str(raw_review.get("symbol") or "").upper(),
            str(raw_review.get("side") or "").upper()
        )
        review_map[review_key] = {
            "action": raw_review.get("action"),
            "confidence_adjustment": raw_review.get("confidence_adjustment"),
            "risk_label": raw_review.get("risk_label"),
            "reason": raw_review.get("reason"),
        }

    included_keys = {
        (
            str(payload.get("symbol") or "").upper(),
            str(payload.get("proposed_signal") or "").upper()
        )
        for payload in included
    }

    for candidate, payload in misses:
        key = (
            str(payload.get("symbol") or "").upper(),
            str(payload.get("proposed_signal") or "").upper()
        )

        if key not in included_keys:
            continue

        review = review_map.get(key)

        if review is None:
            continue

        _store_review(cache, payload, review, now)
        _attach_prefetched_review(candidate, review, "batch")
        prepared += 1

    cache["provider_backoff_until"] = 0
    _save_cache(cache)
    log_info(
        f"LLM batch ready | requested={len(included)} | reviewed={prepared}"
    )
    return prepared


def _adjust_side(side_data, delta):
    if not side_data:
        return

    confidence = float(side_data.get("confidence", 0) or 0)
    side_data["confidence"] = round(max(min(confidence + delta, 100), 0), 2)
    side_data["llm_adjustment"] = delta


def apply_llm_filter(
    symbol,
    side,
    analysis,
    participation=None,
    btc_trend=None,
    btc_corr=None,
    rs=None,
    news_context=None,
    prefetched_review=None,
    prefetched_source=""
):
    if not config.LLM_FILTER_ENABLED:
        return True, analysis, _empty_context(symbol, "LLM_FILTER_DISABLED", False)

    if not side:
        return True, analysis, _empty_context(symbol, "LLM_NO_SIGNAL")

    payload = _build_payload(
        symbol,
        side,
        analysis,
        participation,
        btc_trend,
        btc_corr,
        rs,
        news_context
    )

    if isinstance(prefetched_review, dict):
        review = prefetched_review
        source = prefetched_source or "batch"
        reason = ""
    else:
        review, source, reason = _get_review(payload)

    if review is None:
        context = _empty_context(symbol, reason or "LLM_UNAVAILABLE")
        context["source"] = source

        if context.get("reason") == "LLM_PROVIDER_RATE_LIMIT_BACKOFF":
            log_info(f"{symbol} LLM skipped | RATE_LIMIT_BACKOFF")
        elif context.get("reason") == "LLM_SCAN_REQUEST_LIMIT_REACHED":
            log_info(f"{symbol} LLM skipped | SCAN_REQUEST_LIMIT")
        elif str(context.get("reason") or "").startswith("LLM_SHARED_"):
            log_info(f"{symbol} LLM skipped | {context.get('reason')}")
        else:
            log_warning(f"{symbol} LLM unavailable | {context.get('reason')}")

        if config.LLM_FAIL_OPEN:
            return True, analysis, context

        return False, analysis, context

    normalised = _normalise_review(review)
    adjusted = deepcopy(analysis)
    side_key = side.lower()
    delta = normalised["confidence_adjustment"]
    context = {
        "enabled": True,
        "available": True,
        "symbol": symbol,
        "provider": config.LLM_PROVIDER,
        "model": config.LLM_MODEL,
        "source": source,
        "action": normalised["action"],
        "raw_action": str(review.get("action") or ""),
        "confidence_adjustment": delta,
        "risk_label": normalised["risk_label"],
        "reason": normalised["reason"],
    }

    if "stale" in str(source or "").lower():
        context["action"] = "ALLOW"
        context["reason"] = "LLM_STALE_CACHE_ADVISORY_ONLY"
        context["confidence_adjustment"] = 0
        log_warning(
            f"{symbol} LLM stale cache used as advisory only | "
            f"SOURCE={source}"
        )
        return True, analysis, context

    log_info(
        f"{symbol} LLM | ACTION={context['action']} | "
        f"RISK={context['risk_label']} | ADJ={delta} | "
        f"SOURCE={context['source']} | REASON={context['reason']}"
    )

    if context["action"] == "BLOCK":
        context["reason"] = f"LLM_BLOCK:{context['reason']}"

        if config.LLM_BLOCK_HIGH_RISK:
            return False, adjusted, context

        return True, adjusted, context

    if delta:
        _adjust_side(adjusted.get(side_key, {}), delta)
        current = adjusted.get(side_key, {}).get("confidence", 0)
        adjusted["best_confidence"] = current

        if current < config.LONG_TERM_SIGNAL_THRESHOLD:
            context["action"] = "BLOCK"
            context["reason"] = "LLM_ADJUSTED_CONFIDENCE_BELOW_THRESHOLD"
            return False, adjusted, context

    return True, adjusted, context
