from binance.client import Client
from binance.enums import *

from collections import deque
from decimal import (
    Decimal,
    InvalidOperation,
    ROUND_DOWN,
    ROUND_HALF_UP,
    ROUND_UP,
)
import pandas as pd
import re
import threading
import time
import uuid
import numpy as np

import config
from indicators import apply_indicators
from logger import log_info, log_warning, log_error
from execution_telemetry import (
    aggregate_order_execution,
    append_execution_telemetry,
    calculate_slippage_bps,
)


client = Client(config.API_KEY, config.SECRET_KEY, ping=False)
_exchange_info_cache = None
_last_kline_request_at = 0.0
_public_rest_backoff_until = 0.0
_public_rest_log_times = {}
_public_request_weights = deque()
_public_request_lock = threading.Lock()
_public_rest_lock = threading.Lock()
_kline_request_lock = threading.Lock()
_kline_cache = {}
_kline_cache_lock = threading.RLock()
_futures_context_cache = {}
_private_rest_backoff_until = 0.0
_private_rest_log_times = {}
_private_rest_lock = threading.Lock()
_private_account_cache = {"time": 0.0, "data": None}
_private_balance_cache = {"time": 0.0, "data": None}
_private_position_cache = {}
_private_account_cache_lock = threading.Lock()
_private_balance_cache_lock = threading.Lock()
_private_position_cache_lock = threading.Lock()
_BAN_UNTIL_RE = re.compile(r"banned until\s+(\d+)", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"(code=-1003|too many requests)", re.IGNORECASE)

def sync_client_time():
    try:
        server_time = client.get_server_time()
        client.timestamp_offset = (
            server_time["serverTime"] - int(time.time() * 1000)
        )
        return True

    except Exception as exc:
        client.timestamp_offset = 0
        log_warning(
            "Binance startup time sync unavailable; "
            "using zero timestamp offset | "
            f"ERROR={exc}"
        )
        return False


def is_one_way_position_mode():
    """Return True only when Binance confirms one-way (non-hedge) mode."""
    try:
        response = _private_rest_call(
            "futures_get_position_mode",
            client.futures_get_position_mode,
        )
        dual_side = (response or {}).get("dualSidePosition")

        if isinstance(dual_side, str):
            normalised = dual_side.strip().lower()

            if normalised not in {"true", "false"}:
                log_error(
                    "Position mode verification returned an invalid "
                    f"dualSidePosition value: {dual_side!r}"
                )
                return False

            dual_side = normalised == "true"

        if not isinstance(dual_side, bool):
            log_error(
                "Position mode verification did not return an explicit "
                "dualSidePosition boolean"
            )
            return False

        return dual_side is False

    except Exception as exc:
        log_error(f"Position mode verification failed: {exc}")
        return False


def _throttle_kline_request():
    global _last_kline_request_at

    _raise_if_public_rest_backoff("klines")

    with _kline_request_lock:
        delay = getattr(config, "REQUEST_THROTTLE_SECONDS", 0)

        if delay > 0:
            elapsed = time.time() - _last_kline_request_at

            if elapsed < delay:
                time.sleep(delay - elapsed)

        _rate_limit_public_request(getattr(config, "KLINE_REQUEST_WEIGHT", 2))
        _last_kline_request_at = time.time()


def _rate_limit_public_request(weight=1):
    max_weight = float(
        getattr(config, "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800)
    )

    if max_weight <= 0:
        return

    window_seconds = max(
        float(getattr(config, "BINANCE_PUBLIC_RATE_WINDOW_SECONDS", 60)),
        1.0
    )
    weight = max(float(weight or 1), 0.1)

    while True:
        now = time.time()

        with _public_request_lock:
            while (
                _public_request_weights and
                now - _public_request_weights[0][0] >= window_seconds
            ):
                _public_request_weights.popleft()

            used_weight = sum(item[1] for item in _public_request_weights)

            if used_weight + weight <= max_weight:
                _public_request_weights.append((now, weight))
                return

            oldest_at = _public_request_weights[0][0]
            wait_seconds = window_seconds - (now - oldest_at) + 0.05

        time.sleep(min(max(wait_seconds, 0.05), 5.0))


def _try_reserve_shadow_public_weight(weight=1):
    """Non-blocking shadow reservation that preserves core REST headroom."""
    max_weight = max(
        float(
            getattr(config, "BINANCE_PUBLIC_WEIGHT_LIMIT_PER_MINUTE", 1800)
        ),
        0,
    )

    if max_weight <= 0:
        return True

    reserve = max(
        float(
            getattr(config, "ORDER_FLOW_SHADOW_CORE_WEIGHT_RESERVE", 500)
        ),
        0,
    )
    shadow_ceiling = max(max_weight - min(reserve, max_weight), 0)
    window_seconds = max(
        float(getattr(config, "BINANCE_PUBLIC_RATE_WINDOW_SECONDS", 60)),
        1,
    )
    weight = max(float(weight or 1), 0.1)
    now = time.time()

    with _public_request_lock:
        while (
            _public_request_weights and
            now - _public_request_weights[0][0] >= window_seconds
        ):
            _public_request_weights.popleft()

        used_weight = sum(item[1] for item in _public_request_weights)

        if used_weight + weight > shadow_ceiling:
            return False

        _public_request_weights.append((now, weight))
        return True

def _futures_context_throttle():
    _raise_if_public_rest_backoff("futures_context")
    _rate_limit_public_request(
        getattr(config, "FUTURES_CONTEXT_REQUEST_WEIGHT", 1)
    )


def _public_rest_log_allowed(key):
    now = time.time()
    cooldown = max(
        float(getattr(config, "PUBLIC_REST_LOG_COOLDOWN_SECONDS", 30)),
        1.0
    )

    with _public_rest_lock:
        last_logged_at = _public_rest_log_times.get(key, 0.0)

        if now - last_logged_at < cooldown:
            return False

        _public_rest_log_times[key] = now
        return True


def _extract_public_rest_backoff_seconds(error):
    message = str(error)
    buffer_seconds = max(
        float(getattr(config, "PUBLIC_REST_BACKOFF_BUFFER_SECONDS", 60)),
        0.0
    )
    match = _BAN_UNTIL_RE.search(message)

    if match:
        try:
            banned_until_ms = int(match.group(1))
            banned_until_seconds = banned_until_ms / 1000
            return max(
                banned_until_seconds - time.time() + buffer_seconds,
                buffer_seconds
            )
        except (TypeError, ValueError):
            pass

    if _RATE_LIMIT_RE.search(message):
        return max(
            float(getattr(config, "PUBLIC_REST_DEFAULT_BACKOFF_SECONDS", 300)),
            1.0
        )

    return 0.0


def _set_public_rest_backoff(error, context):
    global _public_rest_backoff_until

    pause_seconds = _extract_public_rest_backoff_seconds(error)

    if pause_seconds <= 0:
        return

    until = time.time() + pause_seconds

    with _public_rest_lock:
        _public_rest_backoff_until = max(_public_rest_backoff_until, until)

    if _public_rest_log_allowed("public_rest_backoff_set"):
        log_warning(
            "Binance public REST backoff active | "
            f"CALL={context} | PAUSE_SECONDS={round(pause_seconds, 1)} | "
            f"ERROR={error}"
        )


def get_public_rest_backoff_remaining():
    with _public_rest_lock:
        return max(_public_rest_backoff_until - time.time(), 0.0)


def is_public_rest_backoff_active():
    return get_public_rest_backoff_remaining() > 0


def _raise_if_public_rest_backoff(context):
    remaining = get_public_rest_backoff_remaining()

    if remaining <= 0:
        return

    if _public_rest_log_allowed(f"public_rest_backoff_skip:{context}"):
        log_warning(
            "Binance public REST call skipped during backoff | "
            f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
        )

    raise RuntimeError(
        "Binance public REST backoff active | "
        f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
    )


def _is_public_rest_backoff_error(error):
    return "Binance public REST backoff active" in str(error)


def _public_rest_call(context, func, *args, weight=1, **kwargs):
    _raise_if_public_rest_backoff(context)
    _rate_limit_public_request(weight)

    try:
        return func(*args, **kwargs)

    except Exception as e:
        _set_public_rest_backoff(e, context)
        raise


def _private_rest_log_allowed(key):
    now = time.time()
    cooldown = max(
        float(getattr(config, "PRIVATE_REST_LOG_COOLDOWN_SECONDS", 30)),
        1.0
    )

    with _private_rest_lock:
        last_logged_at = _private_rest_log_times.get(key, 0.0)

        if now - last_logged_at < cooldown:
            return False

        _private_rest_log_times[key] = now
        return True


def _extract_private_rest_backoff_seconds(error):
    message = str(error)
    buffer_seconds = max(
        float(getattr(config, "PRIVATE_REST_BACKOFF_BUFFER_SECONDS", 60)),
        0.0
    )
    match = _BAN_UNTIL_RE.search(message)

    if match:
        try:
            banned_until_ms = int(match.group(1))
            banned_until_seconds = banned_until_ms / 1000
            return max(
                banned_until_seconds - time.time() + buffer_seconds,
                buffer_seconds
            )
        except (TypeError, ValueError):
            pass

    if _RATE_LIMIT_RE.search(message):
        return max(
            float(getattr(config, "PRIVATE_REST_DEFAULT_BACKOFF_SECONDS", 300)),
            1.0
        )

    return 0.0


def _set_private_rest_backoff(error, context):
    global _private_rest_backoff_until

    pause_seconds = _extract_private_rest_backoff_seconds(error)

    if pause_seconds <= 0:
        return

    until = time.time() + pause_seconds

    with _private_rest_lock:
        _private_rest_backoff_until = max(_private_rest_backoff_until, until)

    if _private_rest_log_allowed("private_rest_backoff_set"):
        log_warning(
            "Binance private REST backoff active | "
            f"CALL={context} | PAUSE_SECONDS={round(pause_seconds, 1)} | "
            f"ERROR={error}"
        )


def get_private_rest_backoff_remaining():
    with _private_rest_lock:
        return max(_private_rest_backoff_until - time.time(), 0.0)


def is_private_rest_backoff_active():
    return get_private_rest_backoff_remaining() > 0


def _raise_if_private_rest_backoff(context):
    remaining = get_private_rest_backoff_remaining()

    if remaining <= 0:
        return

    if _private_rest_log_allowed(f"private_rest_backoff_skip:{context}"):
        log_warning(
            "Binance private REST call skipped during backoff | "
            f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
        )

    raise RuntimeError(
        "Binance private REST backoff active | "
        f"CALL={context} | WAIT_SECONDS={round(remaining, 1)}"
    )


def _private_rest_call(context, func, *args, **kwargs):
    _raise_if_private_rest_backoff(context)

    try:
        return func(*args, **kwargs)

    except Exception as e:
        _set_private_rest_backoff(e, context)
        raise


def _copy_response(data):
    if isinstance(data, list):
        return [
            dict(item) if isinstance(item, dict) else item
            for item in data
        ]

    if isinstance(data, dict):
        return dict(data)

    return data


def _cached_private_data(cache, lock, cache_seconds):
    if cache_seconds <= 0:
        return None, 0.0

    with lock:
        data = cache.get("data")
        age = time.time() - float(cache.get("time", 0.0))

    if data is None or age > cache_seconds:
        return None, age

    return _copy_response(data), age


def _stale_private_data(cache, lock):
    stale_seconds = max(
        float(getattr(config, "PRIVATE_REST_STALE_CACHE_SECONDS", 60)),
        0.0
    )

    with lock:
        data = cache.get("data")
        age = time.time() - float(cache.get("time", 0.0))

    if data is None:
        return None

    if stale_seconds > 0 and age > stale_seconds:
        return None

    return _copy_response(data)


def _store_private_cache(cache, lock, data):
    with lock:
        cache["time"] = time.time()
        cache["data"] = _copy_response(data)


def _get_futures_account(force=False):
    cache_seconds = max(
        float(getattr(config, "PRIVATE_ACCOUNT_CACHE_SECONDS", 5)),
        0.0
    )

    if not force:
        cached, _ = _cached_private_data(
            _private_account_cache,
            _private_account_cache_lock,
            cache_seconds
        )

        if cached is not None:
            return cached

    if is_private_rest_backoff_active():
        stale = _stale_private_data(
            _private_account_cache,
            _private_account_cache_lock
        )

        if stale is not None:
            return stale

    account = _private_rest_call(
        "futures_account",
        client.futures_account
    )
    _store_private_cache(
        _private_account_cache,
        _private_account_cache_lock,
        account
    )
    return _copy_response(account)


def _get_futures_account_balance(force=False):
    cache_seconds = max(
        float(getattr(config, "PRIVATE_ACCOUNT_CACHE_SECONDS", 5)),
        0.0
    )

    if not force:
        cached, _ = _cached_private_data(
            _private_balance_cache,
            _private_balance_cache_lock,
            cache_seconds
        )

        if cached is not None:
            return cached

    if is_private_rest_backoff_active():
        stale = _stale_private_data(
            _private_balance_cache,
            _private_balance_cache_lock
        )

        if stale is not None:
            return stale

    balances = _private_rest_call(
        "futures_account_balance",
        client.futures_account_balance
    )
    _store_private_cache(
        _private_balance_cache,
        _private_balance_cache_lock,
        balances
    )
    return _copy_response(balances)


def _position_cache_key(symbol=None):
    return symbol if symbol else "__all__"


def _get_cached_position_info(symbol, cache_seconds):
    key = _position_cache_key(symbol)

    with _private_position_cache_lock:
        cached = _private_position_cache.get(key)

        if cached:
            age = time.time() - cached["time"]

            if cache_seconds > 0 and age <= cache_seconds:
                return _copy_response(cached["data"])

        if symbol:
            all_cached = _private_position_cache.get("__all__")

            if all_cached:
                age = time.time() - all_cached["time"]

                if cache_seconds > 0 and age <= cache_seconds:
                    positions = [
                        item
                        for item in all_cached["data"]
                        if item.get("symbol") == symbol
                    ]
                    return _copy_response(positions)

    return None


def _get_stale_position_info(symbol):
    stale_seconds = max(
        float(getattr(config, "PRIVATE_REST_STALE_CACHE_SECONDS", 60)),
        0.0
    )
    keys = [_position_cache_key(symbol)]

    if symbol:
        keys.append("__all__")

    with _private_position_cache_lock:
        for key in keys:
            cached = _private_position_cache.get(key)

            if not cached:
                continue

            age = time.time() - cached["time"]

            if stale_seconds > 0 and age > stale_seconds:
                continue

            data = cached["data"]

            if symbol and key == "__all__":
                data = [
                    item
                    for item in data
                    if item.get("symbol") == symbol
                ]

            return _copy_response(data)

    return None


def _store_position_info(symbol, positions):
    key = _position_cache_key(symbol)

    with _private_position_cache_lock:
        _private_position_cache[key] = {
            "time": time.time(),
            "data": _copy_response(positions)
        }


def _clear_position_cache(symbol=None):
    with _private_position_cache_lock:
        if symbol:
            _private_position_cache.pop(symbol, None)

        _private_position_cache.pop("__all__", None)


def _get_futures_position_information(symbol=None, force=False):
    cache_seconds = max(
        float(getattr(config, "PRIVATE_POSITION_CACHE_SECONDS", 3)),
        0.0
    )

    if not force:
        cached = _get_cached_position_info(symbol, cache_seconds)

        if cached is not None:
            return cached

    if not force and is_private_rest_backoff_active():
        stale = _get_stale_position_info(symbol)

        if stale is not None:
            return stale

    params = {}

    if symbol:
        params["symbol"] = symbol

    positions = _private_rest_call(
        f"futures_position_information:{symbol or 'all'}",
        client.futures_position_information,
        **params
    )
    _store_position_info(symbol, positions)
    return _copy_response(positions)


def _kline_cache_expiry(df, now=None):
    now = time.time() if now is None else float(now)
    fallback_seconds = max(
        float(getattr(config, "KLINE_CACHE_SECONDS", 0)),
        0
    )
    fallback_expiry = now + fallback_seconds

    if not getattr(config, "KLINE_CACHE_CANDLE_AWARE_ENABLED", True):
        return fallback_expiry

    try:
        close_time_ms = float(df["close_time"].iloc[-1])
        close_time_seconds = close_time_ms / 1000
        grace_seconds = max(
            float(getattr(config, "KLINE_CACHE_CLOSE_GRACE_SECONDS", 2)),
            0
        )
        candle_expiry = close_time_seconds + grace_seconds

        if candle_expiry > now:
            return candle_expiry
    except (KeyError, IndexError, TypeError, ValueError):
        pass

    return fallback_expiry


def _get_cached_kline_df(key):
    cache_seconds = float(getattr(config, "KLINE_CACHE_SECONDS", 0))

    if cache_seconds <= 0:
        return None

    now = time.time()

    with _kline_cache_lock:
        cached = _kline_cache.get(key)

        if not cached:
            return None

        expires_at = float(
            cached.get("expires_at", cached["time"] + cache_seconds)
        )

        if now >= expires_at:
            _kline_cache.pop(key, None)
            return None

        return cached["data"].copy(deep=True)


def _store_cached_kline_df(key, df):
    cache_seconds = float(getattr(config, "KLINE_CACHE_SECONDS", 0))

    if cache_seconds <= 0 or df is None:
        return

    max_items = max(int(getattr(config, "KLINE_CACHE_MAX_ITEMS", 2400)), 1)
    now = time.time()

    with _kline_cache_lock:
        _kline_cache[key] = {
            "time": now,
            "expires_at": _kline_cache_expiry(df, now=now),
            "data": df.copy(deep=True)
        }

        while len(_kline_cache) > max_items:
            oldest_key = min(
                _kline_cache,
                key=lambda item: _kline_cache[item]["time"]
            )
            _kline_cache.pop(oldest_key, None)


def get_exchange_info():
    global _exchange_info_cache

    if _exchange_info_cache is None:
        _exchange_info_cache = _public_rest_call(
            "futures_exchange_info",
            client.futures_exchange_info,
            weight=1
        )

    return _exchange_info_cache


def get_supported_symbols():

    try:
        symbols = set()

        for item in get_exchange_info().get("symbols", []):
            if item.get("status") != "TRADING":
                continue

            if item.get("contractType") != "PERPETUAL":
                continue

            symbols.add(item["symbol"])

        return symbols

    except Exception as e:
        log_error(f"supported symbols error: {e}")
        return set()


def _to_float(value, default=None):
    try:
        if value in (None, ""):
            return default

        return float(value)

    except (TypeError, ValueError):
        return default


def _latest_item(items):
    if not items:
        return None

    return items[-1]


def _change_pct(items, field):
    if not items or len(items) < 2:
        return None

    first = _to_float(items[0].get(field))
    last = _to_float(items[-1].get(field))

    if not first:
        return None

    return round(((last - first) / first) * 100, 2)


def _ema_value(items, field, alpha=0.35):
    values = [
        _to_float(item.get(field))
        for item in (items or [])
        if _to_float(item.get(field)) is not None
    ]

    if not values:
        return None

    alpha = min(max(float(alpha), 0.01), 1.0)
    result = values[0]

    for value in values[1:]:
        result = (alpha * value) + ((1 - alpha) * result)

    return round(result, 6)


def _get_taker_longshort_ratio(params):
    method = getattr(client, "futures_taker_longshort_ratio", None)

    if method:
        return method(**params)

    return client._request_futures_data_api(
        "get",
        "takerlongshortRatio",
        data=params
    )


def get_futures_participation(symbol):
    if not config.FUTURES_CONTEXT_ENABLED:
        return {"available": False, "reason": "DISABLED"}

    key = (
        symbol,
        config.FUTURES_CONTEXT_PERIOD,
        config.FUTURES_CONTEXT_LIMIT
    )
    cached = _futures_context_cache.get(key)

    if cached and time.time() - cached["time"] <= config.FUTURES_CONTEXT_CACHE_SECONDS:
        return cached["data"]

    period = config.FUTURES_CONTEXT_PERIOD
    limit = config.FUTURES_CONTEXT_LIMIT
    params = {"symbol": symbol, "period": period, "limit": limit}
    data = {
        "available": True,
        "symbol": symbol,
        "period": period,
        "limit": limit,
        "oi_change_pct": None,
        "taker_buy_sell_ratio": None,
        "taker_buy_sell_ratio_latest": None,
        "taker_ratio_samples": 0,
        "global_long_short_ratio": None,
        "top_long_short_ratio": None,
        "funding_rate": None,
        "errors": [],
    }

    try:
        _futures_context_throttle()
        oi_hist = client.futures_open_interest_hist(**params)
        data["oi_change_pct"] = _change_pct(oi_hist, "sumOpenInterest")
    except Exception as e:
        _set_public_rest_backoff(e, f"open_interest_hist:{symbol}")
        data["errors"].append(f"OI:{e}")

    try:
        _futures_context_throttle()
        taker_history = _get_taker_longshort_ratio(params) or []
        taker = _latest_item(taker_history)
        data["taker_buy_sell_ratio_latest"] = _to_float(
            taker.get("buySellRatio") if taker else None
        )
        data["taker_buy_sell_ratio"] = _ema_value(
            taker_history,
            "buySellRatio",
            getattr(config, "FUTURES_CONTEXT_TAKER_EMA_ALPHA", 0.35),
        )
        data["taker_ratio_samples"] = len(taker_history)
    except Exception as e:
        _set_public_rest_backoff(e, f"taker_longshort_ratio:{symbol}")
        data["errors"].append(f"TAKER:{e}")

    try:
        _futures_context_throttle()
        global_ratio = _latest_item(client.futures_global_longshort_ratio(**params))
        data["global_long_short_ratio"] = _to_float(
            global_ratio.get("longShortRatio") if global_ratio else None
        )
    except Exception as e:
        _set_public_rest_backoff(e, f"global_longshort_ratio:{symbol}")
        data["errors"].append(f"GLOBAL_LS:{e}")

    try:
        _futures_context_throttle()
        top_ratio = _latest_item(client.futures_top_longshort_position_ratio(**params))
        data["top_long_short_ratio"] = _to_float(
            top_ratio.get("longShortRatio") if top_ratio else None
        )
    except Exception as e:
        _set_public_rest_backoff(e, f"top_longshort_ratio:{symbol}")
        data["errors"].append(f"TOP_LS:{e}")

    try:
        _futures_context_throttle()
        premium = client.futures_mark_price(symbol=symbol)
        data["funding_rate"] = _to_float(premium.get("lastFundingRate"))
    except Exception as e:
        _set_public_rest_backoff(e, f"mark_price:{symbol}")
        data["errors"].append(f"FUNDING:{e}")

    usable_values = [
        data["oi_change_pct"],
        data["taker_buy_sell_ratio"],
        data["global_long_short_ratio"],
        data["top_long_short_ratio"],
        data["funding_rate"],
    ]

    data["available"] = any(value is not None for value in usable_values)

    if data["errors"]:
        log_warning(f"{symbol} futures context partial: {' | '.join(data['errors'])}")

    _futures_context_cache[key] = {
        "time": time.time(),
        "data": data
    }

    return data


# =========================
# MARGIN TYPE
# =========================
def set_margin_type(symbol, allow_open_order_block=False):

    try:
        _private_rest_call(
            f"futures_change_margin_type:{symbol}",
            client.futures_change_margin_type,
            symbol=symbol,
            marginType=config.MARGIN_TYPE
        )

        log_info(f"{symbol} Margin: {config.MARGIN_TYPE}")
        return True

    except Exception as e:
        message = str(e)

        if (
            allow_open_order_block
            and (
                "code=-4047" in message
                or "Margin type cannot be changed if there exists open orders" in message
                or "code=-4067" in message
                or "Position side cannot be changed if there exists open orders" in message
            )
        ):
            log_warning(
                f"{symbol} margin setup unchanged | open orders exist; "
                "continuing with current margin type"
            )
            return True

        if "No need to change margin type" not in message:
            log_warning(message)
            return False

        log_info(f"{symbol} Margin already {config.MARGIN_TYPE}")
        return True


# =========================
# LEVERAGE
# =========================
def setup_leverage(symbol):

    try:

        response = _private_rest_call(
            f"futures_change_leverage:{symbol}",
            client.futures_change_leverage,
            symbol=symbol,
            leverage=config.LEVERAGE
        )

        actual = int(response['leverage'])

        if actual != config.LEVERAGE:
            log_warning(f"{symbol} leverage mismatch")
            return False

        log_info(f"{symbol} leverage set: {actual}x")
        return True

    except Exception as e:
        log_error(f"{symbol} leverage error: {e}")
        return False


# =========================
# BALANCE
# =========================
def get_balance():

    balances = _get_futures_account_balance()

    for b in balances:
        if b['asset'] == 'USDT':
            return float(b['balance'])

    return 0


def get_margin_balance():
    return float(_get_futures_account()['totalMarginBalance'])


def get_unrealized_pnl():
    return float(_get_futures_account()['totalUnrealizedProfit'])


def get_mark_price(symbol):

    try:
        mark = _public_rest_call(
            f"futures_mark_price:{symbol}",
            client.futures_mark_price,
            symbol=symbol,
            weight=1
        )
        return float(mark['markPrice'])

    except Exception as e:
        if _is_public_rest_backoff_error(e):
            return None

        _set_public_rest_backoff(e, f"futures_mark_price:{symbol}")
        log_error(f"{symbol} mark price error: {e}")
        return None


def get_book_ticker(symbol):
    try:
        response = _public_rest_call(
            f"futures_orderbook_ticker:{symbol}",
            client.futures_orderbook_ticker,
            symbol=symbol,
            weight=getattr(config, "SMART_EXECUTION_BOOK_TICKER_WEIGHT", 2),
        )
        bid = _to_float((response or {}).get("bidPrice"), 0) or 0
        ask = _to_float((response or {}).get("askPrice"), 0) or 0
        bid_quantity = _to_float((response or {}).get("bidQty"), 0) or 0
        ask_quantity = _to_float((response or {}).get("askQty"), 0) or 0

        if bid <= 0 or ask <= 0 or ask < bid:
            return None

        return {
            "symbol": symbol,
            "bid": bid,
            "ask": ask,
            "bid_quantity": max(bid_quantity, 0),
            "ask_quantity": max(ask_quantity, 0),
            "mid": (bid + ask) / 2,
            "spread_bps": ((ask - bid) / ((bid + ask) / 2)) * 10000,
        }

    except Exception as exc:
        log_warning(f"{symbol} book ticker unavailable: {exc}")
        return None


# =========================
# KLINES
# =========================
def get_futures_depth_snapshot(symbol, limit=None):
    limit = max(
        int(
            limit or getattr(
                config,
                "ORDER_FLOW_SHADOW_DEPTH_SNAPSHOT_LIMIT",
                1000,
            )
        ),
        5,
    )
    weight = 20 if limit >= 1000 else 10 if limit >= 500 else 5

    if get_public_rest_backoff_remaining() > 0:
        return None

    if not _try_reserve_shadow_public_weight(weight):
        if _public_rest_log_allowed(f"shadow_depth_budget:{symbol}"):
            log_warning(
                f"{symbol} shadow depth snapshot deferred | "
                "core public REST reserve protected"
            )
        return None

    try:
        return client.futures_order_book(
            symbol=symbol,
            limit=limit,
        )

    except Exception as exc:
        _set_public_rest_backoff(exc, f"shadow_futures_order_book:{symbol}")

        if _public_rest_log_allowed(f"shadow_depth_error:{symbol}"):
            log_warning(f"{symbol} depth snapshot unavailable: {exc}")

        return None


# =========================
# KLINES
# =========================

def get_klines(symbol, interval, limit=None):

    try:
        limit = limit if limit is not None else config.KLINE_LIMIT
        cache_key = (symbol, interval, int(limit))
        cached = _get_cached_kline_df(cache_key)

        if cached is not None:
            return cached

        _throttle_kline_request()

        klines = client.futures_klines(
            symbol=symbol,
            interval=interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'qav', 'trades', 'tbbav', 'tbqav', 'ignore'
        ])

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        _store_cached_kline_df(cache_key, df)
        return df.copy(deep=True)

    except Exception as e:
        if _is_public_rest_backoff_error(e):
            return None

        _set_public_rest_backoff(e, f"futures_klines:{symbol}:{interval}")
        log_error(f"{symbol} klines error: {e}")
        return None


# =========================
# POSITION CHECKS
# =========================
def has_open_position(symbol, force=False):

    try:
        positions = _get_futures_position_information(
            symbol=symbol,
            force=force
        )

        for p in positions:
            if float(p['positionAmt']) != 0:
                return True

        return False

    except Exception as e:
        log_error(str(e))
        return False


def is_position_closed(symbol, force=False):

    try:
        positions = _get_futures_position_information(
            symbol=symbol,
            force=force
        )

        for p in positions:
            if abs(float(p['positionAmt'])) > 0:
                return False

        return True

    except Exception as e:
        log_error(f"{symbol} position check error: {e}")
        return False


def get_open_positions(force=False):

    try:
        positions = _get_futures_position_information(force=force)
        open_positions = {}

        for p in positions:
            amount = float(p["positionAmt"])

            if amount != 0:
                open_positions[p["symbol"]] = amount

        return open_positions

    except Exception as e:
        log_error(f"open positions error: {e}")
        return None


def _build_open_position_detail(position):
    amount = float(position["positionAmt"])

    if amount == 0:
        return None

    position_symbol = position["symbol"]
    position_side = (position.get("positionSide") or "BOTH").upper()

    return {
        "symbol": position_symbol,
        "amount": amount,
        "side": "BUY" if amount > 0 else "SELL",
        "position_side": position_side,
        "quantity": abs(amount),
        "entry_price": abs(_to_float(position.get("entryPrice"), 0) or 0),
        "mark_price": abs(_to_float(position.get("markPrice"), 0) or 0),
        "liquidation_price": abs(
            _to_float(position.get("liquidationPrice"), 0) or 0
        ),
        "unrealized_pnl": _to_float(position.get("unRealizedProfit"), 0) or 0,
    }


def get_open_position_detail_rows(symbol=None, force=False):

    try:
        positions = _get_futures_position_information(
            symbol=symbol,
            force=force
        )

        open_positions = []

        for position in positions:
            detail = _build_open_position_detail(position)

            if detail:
                open_positions.append(detail)

        return open_positions

    except Exception as e:
        label = symbol if symbol else "all"
        log_error(f"{label} open position row detail error: {e}")
        return None


def get_open_position_details(symbol=None, force=False):

    rows = get_open_position_detail_rows(symbol, force=force)

    if rows is None:
        return None

    return {
        detail["symbol"]: detail
        for detail in rows
    }


def get_open_position_counts(open_positions=None):

    try:

        if open_positions is None:
            open_positions = get_open_positions()

        if open_positions is None:
            return {"total": 0, "buy": 0, "sell": 0}

        total = buy = sell = 0

        for amt in open_positions.values():

            if amt == 0:
                continue

            total += 1

            if amt > 0:
                buy += 1
            else:
                sell += 1

        return {
            "total": total,
            "buy": buy,
            "sell": sell
        }

    except Exception as e:
        log_error(f"position count error: {e}")
        return {"total": 0, "buy": 0, "sell": 0}


def _get_open_algo_orders(symbol):
    method = getattr(client, "futures_get_open_algo_orders", None)

    if method:
        return _private_rest_call(
            f"futures_get_open_algo_orders:{symbol}",
            method,
            symbol=symbol
        )

    return _private_rest_call(
        f"futures_get_open_algo_orders:{symbol}",
        client._request_futures_api,
        "get",
        "openAlgoOrders",
        True,
        data={"symbol": symbol}
    )


def _cancel_algo_order(symbol, algo_id):
    method = getattr(client, "futures_cancel_algo_order", None)

    if method:
        return _private_rest_call(
            f"futures_cancel_algo_order:{symbol}",
            method,
            symbol=symbol,
            algoId=algo_id
        )

    return _private_rest_call(
        f"futures_cancel_algo_order:{symbol}",
        client._request_futures_api,
        "delete",
        "algoOrder",
        True,
        data={
            "symbol": symbol,
            "algoId": algo_id,
        }
    )


def _get_algo_order(symbol, algo_id=None, client_algo_id=None):
    params = {"symbol": symbol}

    if algo_id:
        params["algoId"] = algo_id
    elif client_algo_id:
        params["clientAlgoId"] = client_algo_id
    else:
        raise ValueError("algo_id or client_algo_id is required")

    method = getattr(client, "futures_get_algo_order", None)

    if method:
        return _private_rest_call(
            f"futures_get_algo_order:{symbol}",
            method,
            **params,
        )

    return _private_rest_call(
        f"futures_get_algo_order:{symbol}",
        client._request_futures_api,
        "get",
        "algoOrder",
        True,
        data=params,
    )


def _normalise_algo_order_response(response):
    if not isinstance(response, dict):
        return {}

    data = response.get("data")

    if isinstance(data, dict):
        return data

    return response


def get_algo_order_info(symbol, algo_id=None, client_algo_id=None):
    """Return an exact conditional-order lookup with explicit query state."""
    identity = algo_id or client_algo_id or ""

    try:
        order = _normalise_algo_order_response(
            _get_algo_order(
                symbol,
                algo_id=algo_id,
                client_algo_id=client_algo_id,
            )
        )
        order["_query_ok"] = True
        order["_found"] = bool(
            order.get("algoId") or order.get("clientAlgoId")
        )
        return order

    except Exception as exc:
        message = str(exc).lower()
        not_found = (
            "unknown order" in message or
            "order does not exist" in message or
            "not found" in message
        )

        if not not_found:
            log_warning(
                f"{symbol} exact algo lookup failed | ID={identity} | ERROR={exc}"
            )

        return {
            "algoId": algo_id or "",
            "clientAlgoId": client_algo_id or "",
            "_query_ok": bool(not_found),
            "_found": False,
            "_error": str(exc),
        }


def get_algo_order_execution(symbol, algo_id):
    """Resolve an algo order and, when triggered, its exact child order."""
    algo_order = get_algo_order_info(symbol, algo_id=algo_id)
    algo_query_ok = bool(algo_order.get("_query_ok"))
    result = {
        "query_ok": algo_query_ok,
        "algo_query_ok": algo_query_ok,
        "child_query_ok": True,
        "ambiguous": False,
        "found": bool(algo_order.get("_found")),
        "algo_order": algo_order,
        "algo_status": str(algo_order.get("algoStatus") or "").upper(),
        "actual_order": {},
        "actual_status": "",
        # Algo-level actualQty is not reliable fill evidence. Only the exact
        # triggered child order may contribute an attributed execution.
        "executed_quantity": 0.0,
        "triggered": bool(algo_order.get("actualOrderId")),
        "filled": False,
        "open": False,
        "terminal": False,
    }

    if not result["found"]:
        return result

    actual_order_id = algo_order.get("actualOrderId")

    if actual_order_id:
        result["child_query_ok"] = False

        try:
            actual_order = _private_rest_call(
                f"futures_get_order:{symbol}",
                client.futures_get_order,
                symbol=symbol,
                orderId=actual_order_id,
            )
            actual_order = actual_order if isinstance(actual_order, dict) else {}
            result["actual_order"] = actual_order
            result["actual_status"] = str(
                actual_order.get("status") or ""
            ).upper()
            result["child_query_ok"] = bool(result["actual_status"])
            result["executed_quantity"] = _execution_quantity(
                actual_order.get("executedQty")
            )
        except Exception as exc:
            result["actual_order_error"] = str(exc)
            result["ambiguous"] = True

        if not result["child_query_ok"]:
            result["ambiguous"] = True

    algo_status = result["algo_status"]
    algo_open = algo_status in {
        "NEW",
        "PENDING",
        "WORKING",
        "ACTIVE",
        "TRIGGERING",
        "TRIGGERED",
    }
    child_open = result["actual_status"] in {
        "NEW",
        "PARTIALLY_FILLED",
        "PENDING_CANCEL",
    }
    child_terminal = result["actual_status"] in {
        "FILLED",
        "CANCELED",
        "CANCELLED",
        "EXPIRED",
        "EXPIRED_IN_MATCH",
        "REJECTED",
    }
    algo_terminal = algo_status in {
        "FINISHED",
        "CANCELED",
        "CANCELLED",
        "EXPIRED",
        "REJECTED",
        "FAILED",
    }
    result["query_ok"] = bool(
        result["algo_query_ok"] and result["child_query_ok"]
    )
    result["open"] = bool(
        not result["ambiguous"] and (algo_open or child_open)
    )
    result["terminal"] = bool(
        not result["ambiguous"] and
        (child_terminal if actual_order_id else algo_terminal)
    )
    # `actualQty` may equal the requested quantity even on an untriggered,
    # canceled algo order. Only the exact triggered child order can prove fill.
    result["filled"] = bool(
        result["triggered"] and
        result["child_query_ok"] and
        result["actual_status"] == "FILLED"
    )
    return result

def cancel_algo_order(symbol, algo_id):
    if not algo_id:
        return True

    try:
        _cancel_algo_order(symbol, algo_id)
        log_info(f"{symbol} algo order cancelled | ALGO_ID={algo_id}")
        return True

    except Exception as e:
        message = str(e).lower()

        if (
            "unknown order" in message or
            "order does not exist" in message or
            "not found" in message
        ):
            return True

        log_warning(
            f"{symbol} algo order cancel failed | "
            f"ALGO_ID={algo_id} | ERROR={e}"
        )
        return False


def _normalise_algo_orders(response):
    if isinstance(response, dict):
        return response.get("orders") or response.get("data") or []

    return response or []


def _order_trigger_price(order):
    for key in ("triggerPrice", "stopPrice", "activatePrice"):
        value = order.get(key)

        if value not in (None, ""):
            return value

    return ""


def find_matching_open_algo_order(
    symbol,
    order_type,
    close_side,
    position_side=None,
    trigger_price=None,
    close_position=None,
    quantity=None,
    reduce_only=None,
    working_type="MARK_PRICE",
):
    """Find only a protection order whose intent matches every supplied field."""
    try:
        orders = _normalise_algo_orders(_get_open_algo_orders(symbol))
    except Exception as exc:
        log_warning(f"{symbol} matching algo lookup error: {exc}")
        return {
            "query_ok": False,
            "error": str(exc),
        }

    expected_type = str(order_type or "").upper()
    expected_side = str(close_side or "").upper()
    expected_position_side = str(position_side or "BOTH").upper()
    expected_close_position = (
        None
        if close_position is None
        else bool(close_position)
    )
    target_price = _to_float(trigger_price, None)
    target_quantity = _to_float(quantity, None)
    price_rules = get_symbol_price_rules(symbol)
    tick_size = float(_to_float(price_rules.get("tick_size"), 0) or 0)
    price_tolerance = max(
        tick_size * 0.49 if tick_size > 0 else 0,
        abs(float(target_price or 0)) * 1e-12,
        1e-12,
    )

    for order in orders:
        current_type = str(
            order.get("orderType") or order.get("type") or ""
        ).upper()
        current_side = str(order.get("side") or "").upper()
        current_position_side = str(
            order.get("positionSide") or "BOTH"
        ).upper()
        current_reduce_only = str(
            order.get("reduceOnly", False)
        ).lower() == "true"
        current_working_type = str(
            order.get("workingType") or ""
        ).upper()

        if current_type != expected_type or current_side != expected_side:
            continue

        if current_position_side != expected_position_side:
            continue

        if expected_close_position is not None:
            current_close_position = str(
                order.get("closePosition", False)
            ).lower() == "true"

            if current_close_position != expected_close_position:
                continue

        if reduce_only is not None and current_reduce_only != bool(reduce_only):
            continue

        if working_type and current_working_type != str(working_type).upper():
            continue

        if target_price is not None:
            current_price = _to_float(_order_trigger_price(order), None)

            if (
                current_price is None or
                abs(float(current_price) - float(target_price)) > price_tolerance
            ):
                continue

        if target_quantity is not None:
            current_quantity = _to_float(order.get("quantity"), None)
            quantity_tolerance = max(abs(float(target_quantity)) * 1e-6, 1e-12)

            if (
                current_quantity is None or
                abs(float(current_quantity) - float(target_quantity)) >
                quantity_tolerance
            ):
                continue

        return {
            "query_ok": True,
            "price": _order_trigger_price(order),
            "type": current_type,
            "side": current_side,
            "position_side": current_position_side,
            "close_position": str(
                order.get("closePosition", False)
            ).lower() == "true",
            "reduce_only": current_reduce_only,
            "working_type": current_working_type,
            "quantity": order.get("quantity"),
            "order_id": order.get("algoId", ""),
            "client_order_id": order.get("clientAlgoId", ""),
            "order": order,
        }

    return {"query_ok": True}

def get_open_take_profit_info(symbol):
    try:
        tp_types = {"TAKE_PROFIT", "TAKE_PROFIT_MARKET"}

        orders = _private_rest_call(
            f"futures_get_open_orders:{symbol}",
            client.futures_get_open_orders,
            symbol=symbol
        )

        for order in orders:
            order_type = order.get("type")

            if order_type not in tp_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "tp_price": _order_trigger_price(order),
                "type": order_type,
                "source": "order",
                "order_id": order.get("orderId", ""),
            }

        for order in _normalise_algo_orders(_get_open_algo_orders(symbol)):
            order_type = order.get("orderType") or order.get("type")

            if order_type not in tp_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "tp_price": _order_trigger_price(order),
                "type": order_type,
                "source": "algo",
                "order_id": order.get("algoId", ""),
            }

        return {}

    except Exception as e:
        log_warning(f"{symbol} open TP lookup error: {e}")
        return {}


def get_open_stop_loss_info(symbol):
    try:
        sl_types = {"STOP", "STOP_MARKET"}
        orders = _private_rest_call(
            f"futures_get_open_orders:{symbol}",
            client.futures_get_open_orders,
            symbol=symbol
        )

        for order in orders:
            order_type = order.get("type")

            if order_type not in sl_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "sl_price": _order_trigger_price(order),
                "type": order_type,
                "source": "order",
                "order_id": order.get("orderId", ""),
            }

        for order in _normalise_algo_orders(_get_open_algo_orders(symbol)):
            order_type = order.get("orderType") or order.get("type")

            if order_type not in sl_types:
                continue

            return {
                "price": _order_trigger_price(order),
                "sl_price": _order_trigger_price(order),
                "type": order_type,
                "source": "algo",
                "order_id": order.get("algoId", ""),
            }

        return {}

    except Exception as e:
        log_warning(f"{symbol} open SL lookup error: {e}")
        return {}


def cancel_open_protection_orders(symbol):

    try:
        orders = _private_rest_call(
            f"futures_get_open_orders:{symbol}",
            client.futures_get_open_orders,
            symbol=symbol
        )
        protection_types = {
            "TAKE_PROFIT",
            "TAKE_PROFIT_MARKET",
            "STOP",
            "STOP_MARKET",
            "TRAILING_STOP_MARKET",
        }
        cancelled = 0

        for order in orders:
            order_type = order.get("type")
            close_position = str(order.get("closePosition", "")).lower() == "true"
            reduce_only = str(order.get("reduceOnly", "")).lower() == "true"

            if order_type not in protection_types and not (close_position or reduce_only):
                continue

            _private_rest_call(
                f"futures_cancel_order:{symbol}",
                client.futures_cancel_order,
                symbol=symbol,
                orderId=order["orderId"]
            )
            cancelled += 1

        if cancelled:
            log_info(f"{symbol} cancelled {cancelled} protection order(s)")

        algo_orders = _normalise_algo_orders(_get_open_algo_orders(symbol))
        algo_cancelled = 0

        for order in algo_orders:
            order_type = order.get("orderType") or order.get("type")

            if order_type not in protection_types:
                continue

            algo_id = order.get("algoId")

            if not algo_id:
                log_warning(f"{symbol} algo protection missing algoId; cancel unsafe")
                return False

            _cancel_algo_order(symbol, algo_id)
            algo_cancelled += 1

        if algo_cancelled:
            log_info(f"{symbol} cancelled {algo_cancelled} algo protection order(s)")

        return True

    except Exception as e:
        log_error(f"{symbol} protection cancel error: {e}")
        return False


# =========================
# PRECISION
# =========================
def get_symbol_precision(symbol):

    info = get_exchange_info()

    for s in info['symbols']:
        if s['symbol'] == symbol:
            return s['quantityPrecision']

    return 3


def get_price_precision(symbol):

    info = get_exchange_info()

    for s in info['symbols']:
        if s['symbol'] == symbol:
            return int(s['pricePrecision'])

    return 4


def get_symbol_price_rules(symbol):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            price_filter = filters.get("PRICE_FILTER") or {}
            return {
                "available": True,
                "tick_size": price_filter.get("tickSize", "0"),
                "min_price": price_filter.get("minPrice", "0"),
                "max_price": price_filter.get("maxPrice", "0"),
                "precision": int(item.get("pricePrecision", 4)),
            }

    except Exception as exc:
        log_warning(f"{symbol} price rule lookup warning: {exc}")

    return {
        "available": False,
        "tick_size": "0",
        "min_price": "0",
        "max_price": "0",
        "precision": 8,
    }


def normalize_order_price(symbol, price, rounding="nearest"):
    try:
        rules = get_symbol_price_rules(symbol)

        if not rules.get("available", True):
            return 0.0

        value = Decimal(str(float(price)))
        tick = Decimal(str(rules.get("tick_size") or "0"))
        minimum = Decimal(str(rules.get("min_price") or "0"))
        maximum = Decimal(str(rules.get("max_price") or "0"))

        if value <= 0:
            return 0.0

        if tick <= 0:
            precision = max(int(rules.get("precision", 4)), 0)
            return round(float(value), precision)

        rounding_mode = {
            "down": ROUND_DOWN,
            "up": ROUND_UP,
            "nearest": ROUND_HALF_UP,
        }.get(str(rounding or "nearest").lower(), ROUND_HALF_UP)
        steps = (value / tick).to_integral_value(rounding=rounding_mode)
        normalized = steps * tick

        if minimum > 0:
            normalized = max(normalized, minimum)

        if maximum > 0:
            normalized = min(normalized, maximum)

        return float(normalized)

    except (InvalidOperation, TypeError, ValueError) as exc:
        log_warning(f"{symbol} price normalization warning: {exc}")
        return 0.0


def normalize_trigger_price(symbol, side, order_type, price):
    side = str(side or "").upper()
    order_type = str(order_type or "").upper()
    take_profit = "TAKE_PROFIT" in order_type

    if side == SIDE_BUY:
        rounding = "down" if take_profit else "up"
    else:
        rounding = "up" if take_profit else "down"

    return normalize_order_price(symbol, price, rounding=rounding)


def get_symbol_quantity_rules(symbol, order_type="MARKET"):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            order_type = str(order_type or "MARKET").upper()

            if order_type == "MARKET":
                lot_size = (
                    filters.get("MARKET_LOT_SIZE") or
                    filters.get("LOT_SIZE") or
                    {}
                )
            else:
                lot_size = filters.get("LOT_SIZE") or {}

            return {
                "available": True,
                "step_size": lot_size.get("stepSize", "1"),
                "min_qty": lot_size.get("minQty", "0"),
                "max_qty": lot_size.get("maxQty", "0"),
                "precision": int(item.get("quantityPrecision", 3)),
            }
    except Exception as e:
        log_warning(f"{symbol} quantity rule lookup warning: {e}")

    return {
        "available": False,
        "step_size": "0",
        "min_qty": "0",
        "max_qty": "0",
        "precision": 8,
    }


def normalize_order_quantity(
    symbol,
    quantity,
    round_down=True,
    order_type="MARKET",
):
    try:
        rules = get_symbol_quantity_rules(symbol, order_type=order_type)

        if not rules.get("available", True):
            return 0.0

        value = Decimal(str(abs(float(quantity))))
        step = Decimal(str(rules.get("step_size") or "1"))
        min_qty = Decimal(str(rules.get("min_qty") or "0"))
        max_qty = Decimal(str(rules.get("max_qty") or "0"))

        if step > 0 and round_down:
            value = (value / step).to_integral_value(rounding=ROUND_DOWN) * step

        if value < min_qty or value <= 0:
            return 0.0

        if max_qty > 0:
            value = min(value, max_qty)

        precision = max(int(rules.get("precision", 3)), 0)
        quantum = Decimal("1").scaleb(-precision)
        value = value.quantize(quantum, rounding=ROUND_DOWN)
        return float(value)

    except (InvalidOperation, TypeError, ValueError) as e:
        log_warning(f"{symbol} quantity normalization warning: {e}")
        return 0.0


# =========================
# ENTRY PRICE
# =========================
def get_entry_price(symbol, order=None):

    if order:
        avg_price = float(order.get("avgPrice", 0) or 0)

        if avg_price <= 0:
            executed_qty = float(order.get("executedQty", 0) or 0)
            cum_quote = float(order.get("cumQuote", 0) or 0)

            if executed_qty > 0 and cum_quote > 0:
                avg_price = cum_quote / executed_qty

        if avg_price > 0:
            return avg_price

    last_error = None

    for attempt in range(config.ENTRY_PRICE_RETRIES):
        try:
            positions = _get_futures_position_information(
                symbol=symbol,
                force=True
            )
            entry_price = abs(float(positions[0]["entryPrice"]))

            if entry_price > 0:
                return entry_price

        except Exception as e:
            last_error = e

        if attempt < config.ENTRY_PRICE_RETRIES - 1:
            time.sleep(config.ENTRY_PRICE_RETRY_DELAY_SECONDS)

    if last_error:
        log_warning(f"{symbol} entry price polling error: {last_error}")

    return 0


# =========================
# MARKET ORDER
# =========================
def _execution_position_detail(symbol, position_side=None, expected_side=None):
    rows = get_open_position_detail_rows(symbol, force=True)

    if rows is None:
        return False, None

    position_side = str(position_side or "").upper()
    expected_side = str(expected_side or "").upper()

    if position_side in ("LONG", "SHORT"):
        for detail in rows:
            if detail.get("position_side") == position_side:
                return True, detail

        return True, None

    if expected_side in ("BUY", "SELL"):
        for detail in rows:
            if detail.get("side") == expected_side:
                return True, detail

        return True, None

    if len(rows) == 1:
        return True, rows[0]

    return True, None


def _execution_quantity(value):
    return max(float(_to_float(value, 0) or 0), 0)


def _normalised_residual_quantity(symbol, requested_quantity, executed_quantity):
    residual = max(
        _execution_quantity(requested_quantity) -
        _execution_quantity(executed_quantity),
        0,
    )
    tolerance = max(_execution_quantity(requested_quantity) * 1e-9, 1e-12)

    if residual <= tolerance:
        return 0.0, 0.0

    return residual, normalize_order_quantity(symbol, residual)


def _wait_for_position_reconciliation(
    symbol,
    position_side=None,
    expected_side=None,
    accept_condition=None,
):
    attempts = max(
        int(getattr(config, "EXECUTION_VERIFY_ATTEMPTS", 4)),
        1,
    )
    delay_seconds = max(
        float(getattr(config, "EXECUTION_VERIFY_DELAY_SECONDS", 0.25)),
        0,
    )
    last_detail = None

    for attempt in range(1, attempts + 1):
        if attempt > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

        available, detail = _execution_position_detail(
            symbol,
            position_side=position_side,
            expected_side=expected_side,
        )

        if available:
            last_detail = detail

            if accept_condition is None or accept_condition(detail):
                return True, detail, attempt

    return False, last_detail, attempts


def _inferred_entry_fill_price(
    pre_position_amount,
    pre_average_price,
    post_position_detail,
    executed_quantity,
):
    executed_quantity = _execution_quantity(executed_quantity)

    if executed_quantity <= 0 or not post_position_detail:
        return 0

    post_average = _execution_quantity(post_position_detail.get("entry_price"))
    post_quantity = abs(
        float(_to_float(post_position_detail.get("amount"), 0) or 0)
    )
    pre_quantity = abs(float(_to_float(pre_position_amount, 0) or 0))
    pre_average = _execution_quantity(pre_average_price)

    if pre_quantity <= 0:
        return post_average

    if post_average <= 0 or post_quantity <= pre_quantity or pre_average <= 0:
        return 0

    inferred_quote = (
        (post_average * post_quantity) -
        (pre_average * pre_quantity)
    )
    return max(inferred_quote / executed_quantity, 0)


def _enrich_reconciled_order(order, reconciliation):
    result = dict(order or {})
    result["executedQty"] = str(
        round(_execution_quantity(reconciliation.get("executed_quantity")), 12)
    )
    average_fill_price = _execution_quantity(
        reconciliation.get("average_fill_price")
    )

    if average_fill_price > 0:
        result["avgPrice"] = str(round(average_fill_price, 12))

    result["_execution_reconciliation"] = reconciliation
    return result


def get_execution_reconciliation(order):
    if not isinstance(order, dict):
        return {}

    reconciliation = order.get("_execution_reconciliation")
    return reconciliation if isinstance(reconciliation, dict) else {}


def get_reconciled_executed_quantity(order, fallback=0):
    if isinstance(order, dict):
        reconciliation = get_execution_reconciliation(order)
        value = reconciliation.get("executed_quantity")

        if value is not None:
            return _execution_quantity(value)

        if "executedQty" in order:
            return _execution_quantity(order.get("executedQty"))

    return _execution_quantity(fallback)


def is_reconciled_execution_settled(order):
    if not isinstance(order, dict):
        return False

    reconciliation = order.get("_execution_reconciliation")

    if not isinstance(reconciliation, dict):
        return True

    return bool(reconciliation.get("order_terminal", True))


_TERMINAL_EXECUTION_ORDER_STATUSES = {
    "FILLED",
    "CANCELED",
    "CANCELLED",
    "EXPIRED",
    "EXPIRED_IN_MATCH",
    "REJECTED",
}


def _new_execution_client_order_id(label="m"):
    timestamp = int(time.time() * 1000)
    label = re.sub(r"[^a-zA-Z0-9]", "", str(label or "m"))[:4]
    return f"v7{label}-{timestamp}-{uuid.uuid4().hex[:8]}"


def _execution_order_is_terminal(order):
    status = str((order or {}).get("status") or "").upper()
    return status in _TERMINAL_EXECUTION_ORDER_STATUSES


def _resolve_entry_order(symbol, client_order_id, initial_order=None):
    latest_order = initial_order if isinstance(initial_order, dict) else None
    errors = []
    attempts = max(
        int(getattr(config, "EXECUTION_VERIFY_ATTEMPTS", 4)),
        1,
    )
    delay_seconds = max(
        float(getattr(config, "EXECUTION_VERIFY_DELAY_SECONDS", 0.25)),
        0,
    )

    for attempt in range(1, attempts + 1):
        if _execution_order_is_terminal(latest_order):
            return latest_order, True, attempt - 1, errors

        if attempt > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

        try:
            queried_order = _private_rest_call(
                f"futures_get_order:{symbol}",
                client.futures_get_order,
                symbol=symbol,
                origClientOrderId=client_order_id,
            )

            if isinstance(queried_order, dict):
                latest_order = queried_order

        except Exception as exc:
            errors.append(str(exc))

    return (
        latest_order,
        _execution_order_is_terminal(latest_order),
        attempts,
        errors,
    )


def _cancel_unsettled_entry_order(symbol, client_order_id):
    try:
        response = _private_rest_call(
            f"futures_cancel_order:{symbol}",
            client.futures_cancel_order,
            symbol=symbol,
            origClientOrderId=client_order_id,
        )
        _clear_position_cache(symbol)
        return response if isinstance(response, dict) else None, ""

    except Exception as exc:
        return None, str(exc)


def reconcile_execution_client_orders(
    symbol,
    client_order_ids,
    cancel_unsettled=True,
):
    """Re-query persisted client IDs without ever submitting a new order."""
    if isinstance(client_order_ids, str):
        client_order_ids = [
            item.strip()
            for item in client_order_ids.split(",")
            if item.strip()
        ]
    else:
        client_order_ids = [
            str(item).strip()
            for item in (client_order_ids or [])
            if str(item).strip()
        ]

    orders = []
    errors = []
    all_terminal = bool(client_order_ids)
    verification_attempts = 0

    for client_order_id in client_order_ids:
        order, terminal, attempts, status_errors = _resolve_entry_order(
            symbol,
            client_order_id,
        )
        verification_attempts += attempts
        errors.extend(status_errors)

        if not terminal and cancel_unsettled:
            cancel_order, cancel_error = _cancel_unsettled_entry_order(
                symbol,
                client_order_id,
            )

            if cancel_error:
                errors.append(cancel_error)

            order, terminal, attempts, status_errors = _resolve_entry_order(
                symbol,
                client_order_id,
                initial_order=cancel_order or order,
            )
            verification_attempts += attempts
            errors.extend(status_errors)

        if order:
            orders.append(order)

        if not terminal:
            all_terminal = False

    aggregate = aggregate_order_execution(orders)
    return {
        "order_terminal": all_terminal,
        "orders": orders,
        "executed_quantity": aggregate["executed_quantity"],
        "average_fill_price": aggregate["average_fill_price"],
        "order_ids": aggregate["order_ids"],
        "client_order_ids": ",".join(client_order_ids),
        "verification_attempts": verification_attempts,
        "error": " | ".join(errors),
    }


def _submit_entry_market_order(
    symbol,
    side,
    quantity,
    client_order_id=None,
):
    client_order_id = client_order_id or _new_execution_client_order_id("m")

    try:
        return _private_rest_call(
            f"futures_create_order:{symbol}",
            client.futures_create_order,
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
            newOrderRespType="RESULT",
            newClientOrderId=client_order_id,
        )

    finally:
        _clear_position_cache(symbol)


def _place_reconciled_market_order(
    symbol,
    side,
    quantity,
    pre_position_amount=None,
    pre_average_price=None,
    reference_price=None,
    context="ENTRY",
    execution_mode="MARKET",
    fallback_used=False,
    emit_telemetry=True,
):
    requested_quantity = _execution_quantity(quantity)

    if requested_quantity <= 0:
        return None

    if not getattr(config, "EXECUTION_RECONCILIATION_ENABLED", True):
        try:
            order = _submit_entry_market_order(symbol, side, requested_quantity)
            log_info(f"{symbol} MARKET ORDER: {side}")
            return order
        except Exception as exc:
            log_error(f"{symbol} order error: {exc}")
            return None

    started_at = time.monotonic()
    execution_id = _new_execution_client_order_id("grp")
    expected_side = "BUY" if str(side).upper() == "BUY" else "SELL"
    direction = 1 if expected_side == "BUY" else -1
    pre_detail = None

    if pre_position_amount is None:
        _, pre_detail = _execution_position_detail(
            symbol,
            expected_side=expected_side,
        )
        pre_position_amount = (
            float(pre_detail.get("amount", 0) or 0)
            if pre_detail
            else 0
        )

        if pre_average_price is None and pre_detail:
            pre_average_price = pre_detail.get("entry_price")

    pre_position_amount = float(_to_float(pre_position_amount, 0) or 0)
    residual_retry_attempts = max(
        int(getattr(config, "EXECUTION_RESIDUAL_RETRY_ATTEMPTS", 1)),
        0,
    )
    orders = []
    client_order_ids = []
    errors = []
    submission_attempts = 0
    submitted_quantity = 0.0
    verification_attempts = 0
    position_verified = False
    all_submissions_terminal = True
    post_detail = None
    executed_quantity = 0.0
    observed_position_increase = 0.0
    remaining_to_submit = requested_quantity

    for _ in range(residual_retry_attempts + 1):
        submit_quantity = normalize_order_quantity(symbol, remaining_to_submit)

        if submit_quantity <= 0:
            break

        submission_attempts += 1
        submitted_quantity += submit_quantity
        client_order_id = _new_execution_client_order_id("m")
        client_order_ids.append(client_order_id)
        submitted_order = None

        try:
            submitted_order = _submit_entry_market_order(
                symbol,
                side,
                submit_quantity,
                client_order_id=client_order_id,
            )
            log_info(
                f"{symbol} MARKET ORDER: {side} | "
                f"ATTEMPT={submission_attempts} | QTY={submit_quantity}"
            )
        except Exception as exc:
            errors.append(str(exc))
            log_error(
                f"{symbol} order error | ATTEMPT={submission_attempts}: {exc}"
            )

        resolved_order, order_terminal, status_attempts, status_errors = (
            _resolve_entry_order(
                symbol,
                client_order_id,
                initial_order=submitted_order,
            )
        )
        verification_attempts += status_attempts
        errors.extend(status_errors)

        if not order_terminal:
            cancel_order, cancel_error = _cancel_unsettled_entry_order(
                symbol,
                client_order_id,
            )

            if cancel_error:
                errors.append(cancel_error)

            resolved_order, order_terminal, status_attempts, status_errors = (
                _resolve_entry_order(
                    symbol,
                    client_order_id,
                    initial_order=cancel_order or resolved_order,
                )
            )
            verification_attempts += status_attempts
            errors.extend(status_errors)

        if resolved_order:
            orders.append(resolved_order)

        if not order_terminal:
            all_submissions_terminal = False

        known_execution = aggregate_order_execution(orders)[
            "executed_quantity"
        ]

        def entry_position_updated(detail):
            post_amount = (
                float(detail.get("amount", 0) or 0)
                if detail
                else 0
            )
            delta = max(
                (post_amount - pre_position_amount) * direction,
                0,
            )
            required_delta = min(known_execution, requested_quantity)
            return bool(
                required_delta > 0 and
                delta + 1e-12 >= required_delta
            )

        if known_execution > 0:
            reconciled, detail, verify_count = (
                _wait_for_position_reconciliation(
                    symbol,
                    expected_side=expected_side,
                    accept_condition=entry_position_updated,
                )
            )
            verification_attempts += verify_count
        else:
            # Capture one observation for ambiguity telemetry, but a position
            # change can never manufacture fill attribution for this order.
            available, detail = _execution_position_detail(
                symbol,
                expected_side=expected_side,
            )
            reconciled = False
            verification_attempts += 1

            if not available:
                detail = None

        if detail is not None:
            post_detail = detail
            post_amount = float(detail.get("amount", 0) or 0)
            observed_position_increase = max(
                observed_position_increase,
                (post_amount - pre_position_amount) * direction,
                0,
            )

        position_verified = bool(reconciled)

        aggregate = aggregate_order_execution(orders)
        executed_quantity = min(
            aggregate["executed_quantity"],
            requested_quantity,
        )
        residual_quantity, retry_quantity = _normalised_residual_quantity(
            symbol,
            requested_quantity,
            executed_quantity,
        )

        if retry_quantity <= 0:
            break

        if not order_terminal:
            log_warning(
                f"{symbol} residual entry retry skipped | "
                f"ORDER_STATUS_NOT_TERMINAL | "
                f"CLIENT_ORDER_ID={client_order_id}"
            )
            break

        remaining_to_submit = retry_quantity

    aggregate = aggregate_order_execution(orders)
    residual_quantity, retry_quantity = _normalised_residual_quantity(
        symbol,
        requested_quantity,
        executed_quantity,
    )
    fully_filled = (
        all_submissions_terminal and
        executed_quantity > 0 and
        retry_quantity <= 0
    )
    average_fill_price = aggregate["average_fill_price"]

    if average_fill_price <= 0:
        average_fill_price = _inferred_entry_fill_price(
            pre_position_amount,
            pre_average_price,
            post_detail,
            executed_quantity,
        )

    post_position_amount = (
        float(post_detail.get("amount", 0) or 0)
        if post_detail
        else pre_position_amount + (executed_quantity * direction)
    )
    status = (
        "FILLED"
        if fully_filled
        else "PENDING"
        if not all_submissions_terminal
        else "PARTIAL"
        if executed_quantity > 0
        else "FAILED"
    )
    reconciliation = {
        "execution_id": execution_id,
        "context": str(context or "ENTRY").upper(),
        "execution_mode": execution_mode,
        "fallback_used": bool(fallback_used),
        "requested_quantity": requested_quantity,
        "submitted_quantity": round(submitted_quantity, 12),
        "executed_quantity": executed_quantity,
        "observed_position_increase_quantity": round(
            observed_position_increase,
            12,
        ),
        "residual_quantity": residual_quantity,
        "fallback_quantity": requested_quantity if fallback_used else 0,
        "fully_filled": fully_filled,
        "order_terminal": all_submissions_terminal,
        "position_verified": position_verified,
        "position_closed": False,
        "average_fill_price": average_fill_price,
        "reference_price": _execution_quantity(reference_price),
        "status": status,
        "submission_attempts": submission_attempts,
        "verification_attempts": verification_attempts,
        "pre_position_amount": pre_position_amount,
        "post_position_amount": post_position_amount,
        "order_ids": aggregate["order_ids"],
        "client_order_ids": ",".join(client_order_ids),
        "commission": aggregate["commission"],
        "commission_asset": aggregate["commission_asset"],
        "error": " | ".join(errors),
    }
    latency_ms = round((time.monotonic() - started_at) * 1000, 2)
    fill_ratio_pct = round(
        (executed_quantity / requested_quantity) * 100,
        4,
    )
    if emit_telemetry:
        append_execution_telemetry({
            **reconciliation,
            "symbol": symbol,
            "order_side": str(side).upper(),
            "position_side": "",
            "fill_ratio_pct": fill_ratio_pct,
            "slippage_bps": calculate_slippage_bps(
                side,
                reference_price,
                average_fill_price,
            ),
            "latency_ms": latency_ms,
            "commission": aggregate["commission"],
            "commission_asset": aggregate["commission_asset"],
        })

    if executed_quantity <= 0 and all_submissions_terminal:
        return None

    if not all_submissions_terminal:
        log_error(
            f"{symbol} MARKET ORDER UNSETTLED | SIDE={side} | "
            f"OBSERVED={executed_quantity}/{requested_quantity} | "
            f"RESIDUAL={residual_quantity} | "
            f"CLIENT_ORDER_IDS={','.join(client_order_ids)}"
        )
    elif not fully_filled:
        log_warning(
            f"{symbol} MARKET ORDER PARTIAL | SIDE={side} | "
            f"FILLED={executed_quantity}/{requested_quantity} | "
            f"RESIDUAL={residual_quantity}"
        )

    return _enrich_reconciled_order(
        orders[-1] if orders else None,
        reconciliation,
    )


def _submit_entry_ioc_limit_order(
    symbol,
    side,
    quantity,
    limit_price,
    client_order_id,
):
    try:
        return _private_rest_call(
            f"futures_create_order_smart:{symbol}",
            client.futures_create_order,
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_LIMIT,
            timeInForce=str(
                getattr(config, "SMART_EXECUTION_TIME_IN_FORCE", "IOC")
            ).upper(),
            quantity=quantity,
            price=limit_price,
            newOrderRespType="RESULT",
            newClientOrderId=client_order_id,
        )

    finally:
        _clear_position_cache(symbol)


def _smart_execution_context_enabled(context):
    normalized = str(context or "ENTRY").upper()

    if normalized.startswith("DCA"):
        normalized = "DCA"

    return normalized in getattr(
        config,
        "SMART_EXECUTION_CONTEXTS",
        {"ENTRY", "DCA"},
    )


def _place_smart_entry_order(
    symbol,
    side,
    quantity,
    pre_position_amount=None,
    pre_average_price=None,
    reference_price=None,
    context="ENTRY",
):
    requested_quantity = _execution_quantity(quantity)

    if requested_quantity <= 0:
        return None

    quote = get_book_ticker(symbol)

    if not quote:
        log_warning(
            f"{symbol} smart execution quote unavailable; using market fallback"
        )
        return _place_reconciled_market_order(
            symbol,
            side,
            requested_quantity,
            pre_position_amount=pre_position_amount,
            pre_average_price=pre_average_price,
            reference_price=reference_price,
            context=context,
            execution_mode="MARKET_QUOTE_FALLBACK",
            fallback_used=True,
        )

    started_at = time.monotonic()
    expected_side = "BUY" if str(side).upper() == "BUY" else "SELL"
    direction = 1 if expected_side == "BUY" else -1
    pre_detail = None

    if pre_position_amount is None:
        _, pre_detail = _execution_position_detail(
            symbol,
            expected_side=expected_side,
        )
        pre_position_amount = (
            float(pre_detail.get("amount", 0) or 0)
            if pre_detail
            else 0
        )

        if pre_average_price is None and pre_detail:
            pre_average_price = pre_detail.get("entry_price")

    pre_position_amount = float(_to_float(pre_position_amount, 0) or 0)
    reference_price = _execution_quantity(
        reference_price or quote.get("mid")
    )
    cross_bps = max(
        float(getattr(config, "SMART_EXECUTION_MAX_CROSS_BPS", 2.0)),
        0,
    )
    raw_limit_price = (
        quote["ask"] * (1 + cross_bps / 10000)
        if expected_side == "BUY"
        else quote["bid"] * (1 - cross_bps / 10000)
    )
    limit_price = normalize_order_price(
        symbol,
        raw_limit_price,
        # The normalized tick must remain inside the configured cross cap.
        rounding="down" if expected_side == "BUY" else "up",
    )
    submit_quantity = normalize_order_quantity(
        symbol,
        requested_quantity,
        order_type="LIMIT",
    )

    marketable_limit = (
        limit_price >= quote["ask"]
        if expected_side == "BUY"
        else limit_price <= quote["bid"]
    )
    within_cross_cap = (
        limit_price <= raw_limit_price + 1e-12
        if expected_side == "BUY"
        else limit_price >= raw_limit_price - 1e-12
    )

    if (
        limit_price <= 0 or
        submit_quantity <= 0 or
        not marketable_limit or
        not within_cross_cap
    ):
        log_info(
            f"{symbol} smart IOC skipped | no marketable tick inside "
            f"{cross_bps} bps cap"
        )
        return _place_reconciled_market_order(
            symbol,
            side,
            requested_quantity,
            pre_position_amount=pre_position_amount,
            pre_average_price=pre_average_price,
            reference_price=reference_price,
            context=context,
            execution_mode="MARKET_PRICE_RULE_FALLBACK",
            fallback_used=True,
        )

    client_order_id = _new_execution_client_order_id("ioc")
    errors = []
    submitted_order = None

    try:
        submitted_order = _submit_entry_ioc_limit_order(
            symbol,
            side,
            submit_quantity,
            limit_price,
            client_order_id,
        )
        log_info(
            f"{symbol} SMART IOC ORDER | SIDE={side} | "
            f"QTY={submit_quantity} | LIMIT={limit_price}"
        )
    except Exception as exc:
        errors.append(str(exc))
        log_warning(f"{symbol} smart IOC response error: {exc}")

    resolved_order, terminal, verification_attempts, status_errors = (
        _resolve_entry_order(
            symbol,
            client_order_id,
            initial_order=submitted_order,
        )
    )
    errors.extend(status_errors)

    if not terminal:
        cancel_order, cancel_error = _cancel_unsettled_entry_order(
            symbol,
            client_order_id,
        )

        if cancel_error:
            errors.append(cancel_error)

        resolved_order, terminal, attempts, status_errors = (
            _resolve_entry_order(
                symbol,
                client_order_id,
                initial_order=cancel_order or resolved_order,
            )
        )
        verification_attempts += attempts
        errors.extend(status_errors)

    limit_orders = [resolved_order] if resolved_order else []
    limit_aggregate = aggregate_order_execution(limit_orders)
    limit_executed = min(
        limit_aggregate["executed_quantity"],
        requested_quantity,
    )
    position_verified = False
    post_detail = None
    observed_position_increase = 0.0

    def smart_position_updated(detail):
        post_amount = (
            float(detail.get("amount", 0) or 0)
            if detail
            else 0
        )
        delta = max(
            (post_amount - pre_position_amount) * direction,
            0,
        )

        return bool(
            limit_executed > 0 and
            delta + 1e-12 >= min(limit_executed, requested_quantity)
        )

    if limit_executed > 0:
        reconciled, detail, verify_count = _wait_for_position_reconciliation(
            symbol,
            expected_side=expected_side,
            accept_condition=smart_position_updated,
        )
        verification_attempts += verify_count
    else:
        available, detail = _execution_position_detail(
            symbol,
            expected_side=expected_side,
        )
        reconciled = False
        verification_attempts += 1

        if not available:
            detail = None

    if detail is not None:
        post_detail = detail
        post_amount = float(detail.get("amount", 0) or 0)
        observed_position_increase = max(
            (post_amount - pre_position_amount) * direction,
            0,
        )

    if reconciled:
        position_verified = True

    residual_quantity, fallback_quantity = _normalised_residual_quantity(
        symbol,
        requested_quantity,
        limit_executed,
    )
    fallback_order = None
    fallback_reconciliation = {}
    fallback_used = False

    if (
        fallback_quantity > 0 and
        terminal and
        getattr(config, "SMART_EXECUTION_MARKET_FALLBACK_ENABLED", True)
    ):
        fallback_used = True
        fallback_pre_amount = (
            float(post_detail.get("amount", 0) or 0)
            if post_detail
            else pre_position_amount + (limit_executed * direction)
        )
        fallback_pre_average = (
            post_detail.get("entry_price")
            if post_detail
            else pre_average_price
        )
        fallback_order = _place_reconciled_market_order(
            symbol,
            side,
            fallback_quantity,
            pre_position_amount=fallback_pre_amount,
            pre_average_price=fallback_pre_average,
            reference_price=reference_price,
            context=context,
            execution_mode="MARKET_SMART_FALLBACK",
            fallback_used=True,
            emit_telemetry=False,
        )
        fallback_reconciliation = get_execution_reconciliation(fallback_order)

    fallback_executed = _execution_quantity(
        fallback_reconciliation.get("executed_quantity")
    )
    total_executed = min(
        limit_executed + fallback_executed,
        requested_quantity,
    )

    if total_executed > 0:
        expected_total = total_executed

        def final_smart_position_updated(detail):
            post_amount = (
                float(detail.get("amount", 0) or 0)
                if detail
                else 0
            )
            delta = max(
                (post_amount - pre_position_amount) * direction,
                0,
            )
            return delta + 1e-12 >= expected_total

        final_reconciled, final_detail, final_attempts = (
            _wait_for_position_reconciliation(
                symbol,
                expected_side=expected_side,
                accept_condition=final_smart_position_updated,
            )
        )
        verification_attempts += final_attempts
        position_verified = bool(final_reconciled)

        if final_detail is not None:
            post_detail = final_detail
            final_amount = float(final_detail.get("amount", 0) or 0)
            observed_position_increase = max(
                observed_position_increase,
                (final_amount - pre_position_amount) * direction,
                0,
            )
    else:
        position_verified = False

    residual_quantity, normalized_residual = _normalised_residual_quantity(
        symbol,
        requested_quantity,
        total_executed,
    )
    fallback_terminal = (
        bool(fallback_reconciliation.get("order_terminal", True))
        if fallback_used
        else True
    )
    all_terminal = bool(terminal and fallback_terminal)
    fully_filled = (
        all_terminal and
        total_executed > 0 and
        normalized_residual <= 0
    )
    combined_orders = list(limit_orders)

    if fallback_executed > 0:
        combined_orders.append({
            "executedQty": str(fallback_executed),
            "avgPrice": str(
                fallback_reconciliation.get("average_fill_price") or 0
            ),
            "orderId": fallback_reconciliation.get("order_ids") or "",
            "clientOrderId": (
                fallback_reconciliation.get("client_order_ids") or ""
            ),
            "status": fallback_reconciliation.get("status") or "",
            "commission": fallback_reconciliation.get("commission"),
            "commissionAsset": fallback_reconciliation.get(
                "commission_asset"
            ),
        })

    aggregate = aggregate_order_execution(combined_orders)
    average_fill_price = aggregate["average_fill_price"]

    if average_fill_price <= 0:
        average_fill_price = _inferred_entry_fill_price(
            pre_position_amount,
            pre_average_price,
            post_detail,
            total_executed,
        )

    status = (
        "FILLED"
        if fully_filled
        else "PENDING"
        if not all_terminal
        else "PARTIAL"
        if total_executed > 0
        else "FAILED"
    )
    post_position_amount = (
        float(fallback_reconciliation.get("post_position_amount"))
        if fallback_reconciliation.get("post_position_amount") is not None
        else float(post_detail.get("amount", 0) or 0)
        if post_detail
        else pre_position_amount + (total_executed * direction)
    )
    order_ids = ",".join(
        value
        for value in (
            limit_aggregate.get("order_ids", ""),
            str(fallback_reconciliation.get("order_ids") or ""),
        )
        if value
    )
    client_order_ids = ",".join(
        value
        for value in (
            client_order_id,
            str(fallback_reconciliation.get("client_order_ids") or ""),
        )
        if value
    )
    reconciliation = {
        "execution_id": client_order_id,
        "context": str(context or "ENTRY").upper(),
        "execution_mode": (
            "SMART_IOC_MARKET_FALLBACK"
            if fallback_used
            else "SMART_IOC"
        ),
        "fallback_used": fallback_used,
        "requested_quantity": requested_quantity,
        "submitted_quantity": round(
            submit_quantity + (
                _execution_quantity(
                    fallback_reconciliation.get("submitted_quantity")
                )
                if fallback_used
                else 0
            ),
            12,
        ),
        "executed_quantity": total_executed,
        "observed_position_increase_quantity": round(
            observed_position_increase,
            12,
        ),
        "residual_quantity": residual_quantity,
        "fallback_quantity": fallback_quantity if fallback_used else 0,
        "fully_filled": fully_filled,
        "order_terminal": all_terminal,
        "position_verified": bool(
            position_verified or
            fallback_reconciliation.get("position_verified")
        ),
        "position_closed": False,
        "average_fill_price": average_fill_price,
        "reference_price": reference_price,
        "status": status,
        "submission_attempts": 1 + int(
            fallback_reconciliation.get("submission_attempts") or 0
        ),
        "verification_attempts": verification_attempts + int(
            fallback_reconciliation.get("verification_attempts") or 0
        ),
        "pre_position_amount": pre_position_amount,
        "post_position_amount": post_position_amount,
        "order_ids": order_ids,
        "client_order_ids": client_order_ids,
        "commission": aggregate["commission"],
        "commission_asset": aggregate["commission_asset"],
        "error": " | ".join(
            errors +
            ([fallback_reconciliation.get("error")] if fallback_reconciliation.get("error") else [])
        ),
    }
    append_execution_telemetry({
        **reconciliation,
        "symbol": symbol,
        "order_side": expected_side,
        "position_side": "",
        "fill_ratio_pct": round(
            (total_executed / requested_quantity) * 100,
            4,
        ),
        "best_bid": quote.get("bid"),
        "best_ask": quote.get("ask"),
        "spread_bps": round(float(quote.get("spread_bps") or 0), 4),
        "limit_price": limit_price,
        "slippage_bps": calculate_slippage_bps(
            side,
            reference_price,
            average_fill_price,
        ),
        "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
        "commission": aggregate["commission"],
        "commission_asset": aggregate["commission_asset"],
    })

    if total_executed <= 0 and all_terminal:
        return None

    if not all_terminal:
        log_error(
            f"{symbol} SMART ORDER UNSETTLED | SIDE={side} | "
            f"OBSERVED={total_executed}/{requested_quantity} | "
            f"NO UNSAFE MARKET FALLBACK"
        )

    return _enrich_reconciled_order(
        fallback_order or resolved_order,
        reconciliation,
    )


def place_market_order(
    symbol,
    side,
    quantity,
    pre_position_amount=None,
    pre_average_price=None,
    reference_price=None,
    context="ENTRY",
):
    if (
        getattr(config, "SMART_EXECUTION_ENABLED", False) and
        getattr(config, "EXECUTION_RECONCILIATION_ENABLED", True) and
        _smart_execution_context_enabled(context)
    ):
        return _place_smart_entry_order(
            symbol,
            side,
            quantity,
            pre_position_amount=pre_position_amount,
            pre_average_price=pre_average_price,
            reference_price=reference_price,
            context=context,
        )

    if (
        getattr(config, "SMART_EXECUTION_ENABLED", False) and
        not getattr(config, "EXECUTION_RECONCILIATION_ENABLED", True)
    ):
        log_warning(
            f"{symbol} smart execution bypassed because reconciliation is disabled"
        )

    return _place_reconciled_market_order(
        symbol,
        side,
        quantity,
        pre_position_amount=pre_position_amount,
        pre_average_price=pre_average_price,
        reference_price=reference_price,
        context=context,
    )


def _is_position_side_error(error):
    message = str(error).lower()
    return (
        "position side" in message
        or "hedge" in message
        or "reduceonly" in message
        or "reduce only" in message
    )


def _submit_close_order(
    symbol,
    side,
    quantity,
    position_side=None,
    reduce_only=True,
    client_order_id=None,
):
    params = {
        "symbol": symbol,
        "side": side,
        "type": FUTURE_ORDER_TYPE_MARKET,
        "quantity": quantity,
        "newOrderRespType": "RESULT",
    }

    if client_order_id:
        params["newClientOrderId"] = client_order_id

    if position_side and position_side != "BOTH":
        params["positionSide"] = position_side
    elif reduce_only:
        params["reduceOnly"] = True

    order = _private_rest_call(
        f"futures_create_order_close:{symbol}",
        client.futures_create_order,
        **params
    )
    _clear_position_cache(symbol)
    return order


def _submit_position_close_once(
    symbol,
    amount,
    position_side=None,
    client_order_id=None,
):
    amount = float(amount)
    quantity = normalize_order_quantity(symbol, abs(amount))
    position_side = (position_side or "").upper()

    if quantity <= 0:
        raise ValueError(f"{symbol} close quantity is below exchange minimum")

    if position_side in ("LONG", "SHORT"):
        side = SIDE_SELL if position_side == "LONG" else SIDE_BUY
        order = _submit_close_order(
            symbol,
            side,
            quantity,
            position_side=position_side,
            reduce_only=False,
            client_order_id=client_order_id,
        )
        log_warning(
            f"{symbol} HEDGE CLOSE ORDER: {side} | POSITION_SIDE={position_side}"
        )
        return order, side, position_side

    side = SIDE_SELL if amount > 0 else SIDE_BUY

    try:
        order = _submit_close_order(
            symbol,
            side,
            quantity,
            reduce_only=True,
            client_order_id=client_order_id,
        )
        log_warning(f"{symbol} REDUCE-ONLY CLOSE ORDER: {side}")
        return order, side, "BOTH"

    except Exception as exc:
        if not _is_position_side_error(exc):
            raise

        inferred_position_side = "LONG" if amount > 0 else "SHORT"
        log_warning(
            f"{symbol} reduce-only close failed; retrying hedge close | "
            f"POSITION_SIDE={inferred_position_side} | ERROR={exc}"
        )
        order = _submit_close_order(
            symbol,
            side,
            quantity,
            position_side=inferred_position_side,
            reduce_only=False,
            client_order_id=client_order_id,
        )
        log_warning(
            f"{symbol} HEDGE CLOSE ORDER: {side} | "
            f"POSITION_SIDE={inferred_position_side}"
        )
        return order, side, inferred_position_side


def _close_position_market_legacy(
    symbol,
    amount,
    position_side=None,
    reference_price=None,
    context="EXIT",
):
    """Close a live position and return success only after it is confirmed flat."""
    try:
        amount = float(amount)
        requested_quantity = abs(amount)

        if requested_quantity <= 0:
            return None

        if not getattr(config, "EXECUTION_RECONCILIATION_ENABLED", True):
            order, _, _ = _submit_position_close_once(
                symbol,
                amount,
                position_side=position_side,
            )
            return order

        started_at = time.monotonic()
        expected_side = "BUY" if amount > 0 else "SELL"
        available, pre_detail = _execution_position_detail(
            symbol,
            position_side=position_side,
            expected_side=expected_side,
        )

        if available and pre_detail is None:
            reconciliation = {
                "context": str(context or "EXIT").upper(),
                "execution_mode": "MARKET_CLOSE",
                "fallback_used": False,
                "requested_quantity": requested_quantity,
                "submitted_quantity": 0,
                "executed_quantity": 0,
                "residual_quantity": 0,
                "fallback_quantity": 0,
                "fully_filled": True,
                "order_terminal": True,
                "position_verified": True,
                "position_closed": True,
                "average_fill_price": 0,
                "reference_price": _execution_quantity(reference_price),
                "status": "ALREADY_CLOSED",
                "submission_attempts": 0,
                "verification_attempts": 1,
                "pre_position_amount": 0,
                "post_position_amount": 0,
                "order_ids": "",
                "client_order_ids": "",
                "error": "",
            }
            append_execution_telemetry({
                **reconciliation,
                "symbol": symbol,
                "order_side": SIDE_SELL if amount > 0 else SIDE_BUY,
                "position_side": str(position_side or "BOTH").upper(),
                "fill_ratio_pct": 100,
                "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
            })
            return _enrich_reconciled_order(None, reconciliation)

        live_amount = (
            float(pre_detail.get("amount", amount) or amount)
            if pre_detail
            else amount
        )
        pre_position_amount = live_amount
        requested_quantity = abs(live_amount)

        if reference_price is None and pre_detail:
            reference_price = pre_detail.get("mark_price")

        residual_retry_attempts = max(
            int(getattr(config, "EXECUTION_RESIDUAL_RETRY_ATTEMPTS", 1)),
            0,
        )
        orders = []
        errors = []
        submission_attempts = 0
        submitted_quantity = 0.0
        verification_attempts = 0
        position_verified = False
        position_closed = False
        post_detail = pre_detail
        order_side = SIDE_SELL if live_amount > 0 else SIDE_BUY
        resolved_position_side = str(position_side or "BOTH").upper()
        remaining_amount = live_amount

        for _ in range(residual_retry_attempts + 1):
            if abs(remaining_amount) <= 0:
                break

            submission_attempts += 1
            submit_quantity = normalize_order_quantity(
                symbol,
                abs(remaining_amount),
            )

            if submit_quantity <= 0:
                errors.append("residual close quantity below exchange minimum")
                break

            submitted_quantity += submit_quantity

            try:
                signed_submit_amount = (
                    submit_quantity if remaining_amount > 0 else -submit_quantity
                )
                order, order_side, resolved_position_side = (
                    _submit_position_close_once(
                        symbol,
                        signed_submit_amount,
                        position_side=position_side,
                    )
                )

                if order:
                    orders.append(order)
            except Exception as exc:
                errors.append(str(exc))
                log_error(
                    f"{symbol} close position error | "
                    f"ATTEMPT={submission_attempts}: {exc}"
                )

            amount_before_submit = abs(remaining_amount)

            def close_position_updated(detail):
                if detail is None:
                    return True

                live_quantity = abs(float(detail.get("amount", 0) or 0))
                return live_quantity < amount_before_submit - 1e-12

            available, detail, verify_count = _wait_for_position_reconciliation(
                symbol,
                position_side=(
                    resolved_position_side
                    if resolved_position_side in ("LONG", "SHORT")
                    else None
                ),
                expected_side=expected_side,
                accept_condition=close_position_updated,
            )
            verification_attempts += verify_count

            if not available:
                break

            position_verified = True
            post_detail = detail

            if detail is None:
                position_closed = True
                remaining_amount = 0
                break

            remaining_amount = float(detail.get("amount", 0) or 0)

            if abs(remaining_amount) >= amount_before_submit - 1e-12:
                break

        aggregate = aggregate_order_execution(orders)
        residual_quantity = abs(remaining_amount) if not position_closed else 0
        executed_quantity = max(requested_quantity - residual_quantity, 0)
        executed_quantity = min(
            max(executed_quantity, aggregate["executed_quantity"]),
            requested_quantity,
        )
        average_fill_price = aggregate["average_fill_price"]
        status = (
            "CLOSED"
            if position_closed
            else "RESIDUAL_OPEN"
            if position_verified
            else "UNVERIFIED"
        )
        reconciliation = {
            "context": str(context or "EXIT").upper(),
            "execution_mode": "MARKET_CLOSE",
            "fallback_used": False,
            "requested_quantity": requested_quantity,
            "submitted_quantity": round(submitted_quantity, 12),
            "executed_quantity": executed_quantity,
            "residual_quantity": residual_quantity,
            "fallback_quantity": 0,
            "fully_filled": position_closed,
            "order_terminal": position_closed,
            "position_verified": position_verified,
            "position_closed": position_closed,
            "average_fill_price": average_fill_price,
            "reference_price": _execution_quantity(reference_price),
            "status": status,
            "submission_attempts": submission_attempts,
            "verification_attempts": verification_attempts,
            "pre_position_amount": pre_position_amount,
            "post_position_amount": (
                float(post_detail.get("amount", 0) or 0)
                if post_detail
                else 0
            ),
            "order_ids": aggregate["order_ids"],
            "client_order_ids": aggregate["client_order_ids"],
            "error": " | ".join(errors),
        }
        append_execution_telemetry({
            **reconciliation,
            "symbol": symbol,
            "order_side": order_side,
            "position_side": resolved_position_side,
            "fill_ratio_pct": round(
                (executed_quantity / requested_quantity) * 100,
                4,
            ),
            "slippage_bps": calculate_slippage_bps(
                order_side,
                reference_price,
                average_fill_price,
            ),
            "latency_ms": round((time.monotonic() - started_at) * 1000, 2),
            "commission": aggregate["commission"],
            "commission_asset": aggregate["commission_asset"],
        })

        if not position_closed:
            log_error(
                f"{symbol} close not confirmed | STATUS={status} | "
                f"RESIDUAL={residual_quantity}"
            )
            return None

        return _enrich_reconciled_order(
            orders[-1] if orders else None,
            reconciliation,
        )

    except Exception as exc:
        log_error(f"{symbol} close position error: {exc}")
        return None


def _close_position_snapshot(symbol, position_side=None):
    rows = get_open_position_detail_rows(symbol, force=True)

    if rows is None:
        return False, None, "POSITION_SNAPSHOT_UNAVAILABLE"

    requested_side = str(position_side or "").upper()

    if requested_side in ("LONG", "SHORT"):
        for detail in rows:
            if str(detail.get("position_side") or "").upper() == requested_side:
                return True, detail, ""

        if rows:
            return False, None, "TARGET_HEDGE_LEG_NOT_FOUND"

        return True, None, ""

    if len(rows) > 1:
        return False, None, "MULTIPLE_POSITION_LEGS_REQUIRE_POSITION_SIDE"

    return True, rows[0] if rows else None, ""


def _wait_for_close_position_reconciliation(
    symbol,
    position_side,
    amount_before_submit,
):
    attempts = max(
        int(getattr(config, "EXECUTION_VERIFY_ATTEMPTS", 4)),
        1,
    )
    delay_seconds = max(
        float(getattr(config, "EXECUTION_VERIFY_DELAY_SECONDS", 0.25)),
        0,
    )
    last_detail = None
    last_reason = ""

    for attempt in range(1, attempts + 1):
        if attempt > 1 and delay_seconds > 0:
            time.sleep(delay_seconds)

        available, detail, reason = _close_position_snapshot(
            symbol,
            position_side=position_side,
        )
        last_reason = reason

        if not available:
            continue

        last_detail = detail

        if detail is None:
            return True, None, attempt, ""

        live_quantity = abs(float(detail.get("amount", 0) or 0))

        if live_quantity < amount_before_submit - 1e-12:
            return True, detail, attempt, ""

    return False, last_detail, attempts, last_reason


def close_position_market(
    symbol,
    amount,
    position_side=None,
    reference_price=None,
    context="EXIT",
):
    """Close the fresh live leg with exact-order and position reconciliation."""
    try:
        caller_amount = float(amount)

        if abs(caller_amount) <= 0:
            return None

        if not getattr(config, "EXECUTION_RECONCILIATION_ENABLED", True):
            order, _, _ = _submit_position_close_once(
                symbol,
                caller_amount,
                position_side=position_side,
            )
            return order

        started_at = time.monotonic()
        execution_id = _new_execution_client_order_id("xgrp")
        available, pre_detail, snapshot_reason = _close_position_snapshot(
            symbol,
            position_side=position_side,
        )

        if not available:
            log_error(
                f"{symbol} close aborted | {snapshot_reason}"
            )
            return None

        if pre_detail is None:
            reconciliation = {
                "execution_id": execution_id,
                "context": str(context or "EXIT").upper(),
                "execution_mode": "MARKET_CLOSE",
                "fallback_used": False,
                "requested_quantity": 0,
                "submitted_quantity": 0,
                "executed_quantity": 0,
                "residual_quantity": 0,
                "fallback_quantity": 0,
                "fully_filled": False,
                "order_terminal": True,
                "position_verified": True,
                "position_closed": True,
                "average_fill_price": 0,
                "reference_price": _execution_quantity(reference_price),
                "status": "ALREADY_CLOSED",
                "submission_attempts": 0,
                "verification_attempts": 1,
                "pre_position_amount": 0,
                "post_position_amount": 0,
                "order_ids": "",
                "client_order_ids": "",
                "error": "",
            }
            append_execution_telemetry({
                **reconciliation,
                "symbol": symbol,
                "order_side": (
                    SIDE_SELL if caller_amount > 0 else SIDE_BUY
                ),
                "position_side": str(position_side or "BOTH").upper(),
                "fill_ratio_pct": 0,
                "latency_ms": round(
                    (time.monotonic() - started_at) * 1000,
                    2,
                ),
            })
            return _enrich_reconciled_order(None, reconciliation)

        live_amount = float(pre_detail.get("amount", 0) or 0)
        resolved_position_side = str(
            pre_detail.get("position_side") or position_side or "BOTH"
        ).upper()

        if live_amount == 0:
            return None

        if caller_amount * live_amount < 0:
            log_warning(
                f"{symbol} close caller side was stale; using fresh live side | "
                f"CALLER_AMOUNT={caller_amount} | LIVE_AMOUNT={live_amount}"
            )

        pre_position_amount = live_amount
        requested_quantity = abs(live_amount)
        remaining_amount = live_amount
        order_side = SIDE_SELL if live_amount > 0 else SIDE_BUY

        if reference_price is None:
            reference_price = pre_detail.get("mark_price")

        residual_retry_attempts = max(
            int(getattr(config, "EXECUTION_RESIDUAL_RETRY_ATTEMPTS", 1)),
            0,
        )
        orders = []
        client_order_ids = []
        errors = []
        submission_attempts = 0
        verification_attempts = 0
        submitted_quantity = 0.0
        position_verified = False
        position_closed = False
        all_orders_terminal = True
        post_detail = pre_detail

        for attempt_index in range(residual_retry_attempts + 1):
            amount_before_submit = abs(remaining_amount)
            submit_quantity = normalize_order_quantity(
                symbol,
                amount_before_submit,
                order_type="MARKET",
            )

            if submit_quantity <= 0:
                errors.append("residual close quantity below exchange minimum")
                break

            client_order_id = _new_execution_client_order_id("x")
            client_order_ids.append(client_order_id)
            submission_attempts += 1
            submitted_quantity += submit_quantity
            submitted_order = None

            try:
                signed_submit_amount = (
                    submit_quantity
                    if remaining_amount > 0
                    else -submit_quantity
                )
                submitted_order, order_side, resolved_position_side = (
                    _submit_position_close_once(
                        symbol,
                        signed_submit_amount,
                        position_side=(
                            resolved_position_side
                            if resolved_position_side in ("LONG", "SHORT")
                            else None
                        ),
                        client_order_id=client_order_id,
                    )
                )
            except Exception as exc:
                errors.append(str(exc))
                log_error(
                    f"{symbol} close submit response error | "
                    f"ATTEMPT={submission_attempts}: {exc}"
                )

            resolved_order, order_terminal, status_attempts, status_errors = (
                _resolve_entry_order(
                    symbol,
                    client_order_id,
                    initial_order=submitted_order,
                )
            )
            verification_attempts += status_attempts
            errors.extend(status_errors)

            if resolved_order:
                orders.append(resolved_order)

            if not order_terminal:
                all_orders_terminal = False

            reconciled, detail, verify_count, position_error = (
                _wait_for_close_position_reconciliation(
                    symbol,
                    (
                        resolved_position_side
                        if resolved_position_side in ("LONG", "SHORT")
                        else None
                    ),
                    amount_before_submit,
                )
            )
            verification_attempts += verify_count

            if position_error:
                errors.append(position_error)

            if detail is not None:
                post_detail = detail

            if reconciled:
                position_verified = True

                if detail is None:
                    position_closed = True
                    remaining_amount = 0.0
                    break

                current_amount = float(detail.get("amount", 0) or 0)

                if current_amount * pre_position_amount < 0:
                    errors.append("position side changed during close")
                    remaining_amount = current_amount
                    break

                remaining_amount = current_amount

            if not order_terminal:
                log_error(
                    f"{symbol} close order unresolved | "
                    f"CLIENT_ORDER_ID={client_order_id} | no duplicate retry"
                )
                break

            if not reconciled or abs(remaining_amount) >= amount_before_submit - 1e-12:
                break

            if attempt_index >= residual_retry_attempts:
                break

        aggregate = aggregate_order_execution(orders)
        residual_quantity = 0.0 if position_closed else abs(remaining_amount)
        observed_reduction = max(requested_quantity - residual_quantity, 0)
        executed_quantity = min(
            aggregate["executed_quantity"],
            requested_quantity,
        )
        tolerance = max(requested_quantity * 1e-9, 1e-12)
        execution_fully_filled = bool(
            position_closed and
            all_orders_terminal and
            executed_quantity + tolerance >= requested_quantity
        )
        status = (
            "CLOSED"
            if execution_fully_filled
            else "CLOSED_UNATTRIBUTED"
            if position_closed
            else "RESIDUAL_OPEN"
            if position_verified
            else "UNVERIFIED"
        )
        reconciliation = {
            "execution_id": execution_id,
            "context": str(context or "EXIT").upper(),
            "execution_mode": "MARKET_CLOSE",
            "fallback_used": False,
            "requested_quantity": requested_quantity,
            "submitted_quantity": round(submitted_quantity, 12),
            "executed_quantity": executed_quantity,
            "observed_position_reduction_quantity": observed_reduction,
            "residual_quantity": residual_quantity,
            "fallback_quantity": 0,
            "fully_filled": execution_fully_filled,
            "order_terminal": all_orders_terminal,
            "position_verified": position_verified,
            "position_closed": position_closed,
            "average_fill_price": aggregate["average_fill_price"],
            "reference_price": _execution_quantity(reference_price),
            "status": status,
            "submission_attempts": submission_attempts,
            "verification_attempts": verification_attempts,
            "pre_position_amount": pre_position_amount,
            "post_position_amount": (
                float(post_detail.get("amount", 0) or 0)
                if post_detail and not position_closed
                else 0
            ),
            "order_ids": aggregate["order_ids"],
            "client_order_ids": ",".join(client_order_ids),
            "error": " | ".join(errors),
        }
        append_execution_telemetry({
            **reconciliation,
            "symbol": symbol,
            "order_side": order_side,
            "position_side": resolved_position_side,
            "fill_ratio_pct": round(
                (executed_quantity / requested_quantity) * 100,
                4,
            ),
            "slippage_bps": calculate_slippage_bps(
                order_side,
                reference_price,
                aggregate["average_fill_price"],
            ),
            "latency_ms": round(
                (time.monotonic() - started_at) * 1000,
                2,
            ),
            "commission": aggregate["commission"],
            "commission_asset": aggregate["commission_asset"],
        })

        if not position_closed:
            log_error(
                f"{symbol} close not confirmed | STATUS={status} | "
                f"RESIDUAL={residual_quantity}"
            )
            return None

        return _enrich_reconciled_order(
            orders[-1] if orders else None,
            reconciliation,
        )

    except Exception as exc:
        log_error(f"{symbol} close position error: {exc}")
        return None


# =========================
# STRUCTURE SL (REQUIRED BY MAIN + STRATEGY)
# =========================
def get_structure_stop_loss(df, side):

    try:

        atr = df['atr'].iloc[-1]

        if side == SIDE_BUY:

            swing_low = df['low'].iloc[-10:-1].min()
            return swing_low - (atr * 0.5)

        else:

            swing_high = df['high'].iloc[-10:-1].max()
            return swing_high + (atr * 0.5)

    except Exception as e:
        log_error(f"SL error: {e}")
        return None


def get_roi_take_profit(side, entry_price, roi, precision):
    if side == SIDE_BUY:
        return round(
            entry_price * (1 + (roi / config.LEVERAGE) / 100),
            precision
        )

    return round(
        entry_price * (1 - (roi / config.LEVERAGE) / 100),
        precision
    )


def _normalise_signal_type(signal_type=None):
    signal_type = str(signal_type or "").upper().strip()
    return "REVERSAL" if signal_type == "REVERSAL" else "TREND"


def is_stop_loss_enabled_for_signal(signal_type=None):
    signal_type = str(signal_type or "").upper().strip()

    if signal_type == "REVERSAL":
        return bool(getattr(config, "REVERSAL_SL_ENABLED", config.SL_ENABLED))

    if signal_type == "TREND":
        return bool(getattr(config, "TREND_SL_ENABLED", config.SL_ENABLED))

    return bool(getattr(config, "SL_ENABLED", False))


def get_max_sl_roi_for_signal(signal_type=None):
    trade_type = _normalise_signal_type(signal_type)

    if trade_type == "REVERSAL":
        return float(getattr(config, "REVERSAL_MAX_SL_ROI", config.MAX_SL_ROI))

    return float(getattr(config, "TREND_MAX_SL_ROI", config.MAX_SL_ROI))


def get_roi_stop_loss(side, entry_price, roi, precision):
    move = (float(roi) / max(float(config.LEVERAGE), 1)) / 100

    if side == SIDE_BUY:
        return round(entry_price * (1 - move), precision)

    return round(entry_price * (1 + move), precision)


def get_signal_stop_loss(side, entry_price, confirm_df, signal_type, precision):
    if not is_stop_loss_enabled_for_signal(signal_type):
        return None

    structure_sl = get_structure_stop_loss(confirm_df, side)
    max_roi = get_max_sl_roi_for_signal(signal_type)
    capped_sl = None

    if max_roi > 0:
        capped_sl = get_roi_stop_loss(side, entry_price, max_roi, precision)

    if structure_sl is None:
        return capped_sl

    structure_sl = round(structure_sl, precision)

    if capped_sl is None:
        return structure_sl

    if side == SIDE_BUY:
        if structure_sl >= entry_price:
            return capped_sl

        return max(structure_sl, capped_sl)

    if structure_sl <= entry_price:
        return capped_sl

    return min(structure_sl, capped_sl)


def is_valid_take_profit(side, tp_price, market_price):
    if side == SIDE_BUY:
        return tp_price > market_price

    return tp_price < market_price


def place_algo_order(**params):
    method = getattr(client, "futures_create_algo_order", None)

    if method:
        return _private_rest_call(
            f"futures_create_algo_order:{params.get('symbol', 'unknown')}",
            method,
            **params
        )

    return _private_rest_call(
        f"futures_create_algo_order:{params.get('symbol', 'unknown')}",
        client._request_futures_api,
        "post",
        "algoOrder",
        True,
        data=params
    )


def _accepted_order_id(order):
    order = order or {}
    status = str(
        order.get("algoStatus") or order.get("status") or ""
    ).upper()

    if status in {"REJECTED", "EXPIRED", "CANCELED", "CANCELLED", "FAILED"}:
        return ""

    return order.get("algoId") or order.get("orderId") or ""


def _position_close_params(position_side=None):
    position_side = str(position_side or "").upper()

    if position_side in ("LONG", "SHORT"):
        return {"positionSide": position_side}

    return {}


def place_partial_take_profit_quantity(
    symbol,
    side,
    quantity,
    trigger_price,
    position_side=None,
    quantity_is_normalized=False,
):
    """Place an exact-quantity reduce-only TP used by TP1 recovery."""
    partial_quantity = (
        abs(float(quantity or 0))
        if quantity_is_normalized
        else normalize_order_quantity(
            symbol,
            quantity,
            order_type="CONDITIONAL",
        )
    )

    if partial_quantity <= 0:
        return None, 0.0

    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    trigger_price = normalize_trigger_price(
        symbol,
        side,
        "TAKE_PROFIT_MARKET",
        trigger_price,
    )

    if trigger_price <= 0:
        raise ValueError(f"{symbol} TP1 trigger price is invalid")

    params = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": close_side,
        "type": "TAKE_PROFIT_MARKET",
        "triggerPrice": trigger_price,
        "quantity": partial_quantity,
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    close_params = _position_close_params(position_side)

    if close_params:
        params.update(close_params)
    else:
        params["reduceOnly"] = "true"

    try:
        order = place_algo_order(**params)
    except Exception as e:
        if close_params or not _is_position_side_error(e):
            raise

        inferred_position_side = "LONG" if side == SIDE_BUY else "SHORT"
        params.pop("reduceOnly", None)
        params["positionSide"] = inferred_position_side
        order = place_algo_order(**params)

    return order, partial_quantity


def place_partial_take_profit(
    symbol,
    side,
    total_quantity,
    close_pct,
    trigger_price,
    position_side=None,
):
    close_pct = min(max(float(close_pct), 0), 100)
    requested_total_quantity = abs(float(total_quantity or 0))

    if close_pct <= 0 or close_pct >= 100:
        return None, 0.0

    total_quantity = normalize_order_quantity(
        symbol,
        total_quantity,
        order_type="CONDITIONAL",
    )
    partial_quantity = normalize_order_quantity(
        symbol,
        total_quantity * close_pct / 100,
        order_type="CONDITIONAL",
    )
    remaining_quantity = normalize_order_quantity(
        symbol,
        total_quantity - partial_quantity,
        order_type="CONDITIONAL",
    )

    if (
        total_quantity <= 0 or
        partial_quantity <= 0 or
        partial_quantity >= total_quantity or
        remaining_quantity <= 0
    ):
        log_warning(
            f"{symbol} partial TP unavailable | TOTAL={total_quantity} | "
            f"PARTIAL={partial_quantity} | REMAINING={remaining_quantity}"
        )
        return None, 0.0

    effective_close_pct = (
        (partial_quantity / requested_total_quantity) * 100
        if requested_total_quantity > 0
        else 0
    )
    max_deviation = max(
        float(getattr(config, "TP1_MAX_CLOSE_PCT_DEVIATION", 5)),
        0,
    )

    if abs(effective_close_pct - close_pct) > max_deviation:
        log_warning(
            f"{symbol} partial TP lot rounding rejected | "
            f"REQUESTED_PCT={round(close_pct, 4)} | "
            f"EFFECTIVE_PCT={round(effective_close_pct, 4)} | "
            f"MAX_DEVIATION={max_deviation}"
        )
        return None, 0.0

    return place_partial_take_profit_quantity(
        symbol,
        side,
        partial_quantity,
        trigger_price,
        position_side=position_side,
        quantity_is_normalized=True,
    )

def place_close_position_protection(
    symbol,
    side,
    order_type,
    trigger_price,
    position_side=None,
):
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
    trigger_price = normalize_trigger_price(
        symbol,
        side,
        order_type,
        trigger_price,
    )

    if trigger_price <= 0:
        raise ValueError(f"{symbol} {order_type} trigger price is invalid")

    params = {
        "algoType": "CONDITIONAL",
        "symbol": symbol,
        "side": close_side,
        "type": order_type,
        "triggerPrice": trigger_price,
        "closePosition": "true",
        "workingType": "MARK_PRICE",
        "priceProtect": "TRUE",
    }
    params.update(_position_close_params(position_side))
    return place_algo_order(**params)


def place_stop_loss_only(
    symbol,
    side,
    entry_price,
    confirm_df,
    signal_type=None,
    position_side=None,
):
    signal_type = str(signal_type or "").upper().strip()
    details = {
        "ok": False,
        "symbol": symbol,
        "signal_type": signal_type or "UNKNOWN",
        "sl_enabled": is_stop_loss_enabled_for_signal(signal_type),
        "sl_price": None,
        "sl_order": None,
    }

    if not details["sl_enabled"]:
        return details

    try:
        precision = get_price_precision(symbol)
        market_price = get_mark_price(symbol)

        if market_price is None:
            return details

        sl_price = get_signal_stop_loss(
            side,
            entry_price,
            confirm_df,
            signal_type,
            precision,
        )
        sl_price = normalize_trigger_price(
            symbol,
            side,
            "STOP_MARKET",
            sl_price,
        )
        details["sl_price"] = sl_price

        if sl_price is None:
            log_warning(f"{symbol} standalone SL unavailable")
            return details

        if side == SIDE_BUY:
            valid = sl_price < market_price
            close_side = SIDE_SELL
        else:
            valid = sl_price > market_price
            close_side = SIDE_BUY

        if not valid:
            log_warning(
                f"{symbol} standalone SL invalid | "
                f"SL={sl_price} | MARKET={market_price}"
            )
            return details

        sl_order = place_close_position_protection(
            symbol,
            side,
            "STOP_MARKET",
            sl_price,
            position_side=position_side,
        )
        details["sl_order"] = sl_order
        details["ok"] = bool(
            sl_order and
            (sl_order.get("algoId") or sl_order.get("orderId"))
        )

        if details["ok"]:
            log_info(
                f"{symbol} standalone SL created | "
                f"TYPE={signal_type or 'UNKNOWN'} | SL={sl_price}"
            )

        return details

    except Exception as e:
        log_error(f"{symbol} standalone SL error: {e}")
        return details


# =========================
# TP/SL EXECUTION (CLEAN VERSION)
# =========================
def place_tp_sl(
    symbol,
    side,
    entry_price,
    quantity,
    confirm_df,
    structure_tp=None,
    roi_override=None,
    roi_mode_label=None,
    signal_type=None,
    enable_multi_tp=False,
    position_side=None,
    return_details=False
):
    signal_type = str(signal_type or "").upper().strip()
    sl_enabled = is_stop_loss_enabled_for_signal(signal_type)
    details = {
        "ok": False,
        "symbol": symbol,
        "side": side,
        "signal_type": signal_type or "UNKNOWN",
        "entry_price": entry_price,
        "quantity": quantity,
        "tp_price": None,
        "tp_mode": "",
        "sl_price": None,
        "sl_enabled": sl_enabled,
        "sl_created": False,
        "sl_order": None,
        "tp_order": None,
        "multi_tp_active": False,
        "tp1_close_pct": None,
        "tp1_requested_close_pct": None,
        "tp1_quantity": None,
        "tp1_base_quantity": quantity,
    }

    try:
        precision = get_price_precision(symbol)

        market_price = get_mark_price(symbol)

        if market_price is None:
            return details if return_details else False

        if side == SIDE_BUY:
            sl_price = get_signal_stop_loss(
                SIDE_BUY,
                entry_price,
                confirm_df,
                signal_type,
                precision
            )

        # ================= SELL =================
        else:
            sl_price = get_signal_stop_loss(
                SIDE_SELL,
                entry_price,
                confirm_df,
                signal_type,
                precision
            )

        reversal_tp = signal_type == "REVERSAL"
        reversal_max_roi = max(
            float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
            0,
        )
        fallback_roi = float(
            getattr(config, "REVERSAL_TP_FALLBACK_ROI", 35)
            if reversal_tp
            else config.STRUCTURE_TP_FALLBACK_ROI
        )

        if reversal_tp and reversal_max_roi > 0:
            fallback_roi = min(fallback_roi, reversal_max_roi)

        if roi_override is not None:
            effective_roi = float(roi_override)

            if reversal_tp and reversal_max_roi > 0:
                effective_roi = min(effective_roi, reversal_max_roi)

            tp_mode = (
                f"REVERSAL_CAPPED_ROI_{effective_roi}%"
                if effective_roi != float(roi_override)
                else roi_mode_label or f"ROI_{effective_roi}%"
            )
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                effective_roi,
                precision
            )
        elif config.STATIC_TP_ENABLED:
            effective_roi = float(config.STATIC_TP_ROI)

            if reversal_tp and reversal_max_roi > 0:
                effective_roi = min(effective_roi, reversal_max_roi)

            tp_mode = (
                f"REVERSAL_STATIC_CAPPED_ROI_{effective_roi}%"
                if effective_roi != float(config.STATIC_TP_ROI)
                else f"STATIC_ROI_{effective_roi}%"
            )
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                effective_roi,
                precision
            )
        elif structure_tp and structure_tp.get("target_price"):
            target_roi = float(structure_tp.get("target_roi") or 0)

            if (
                reversal_tp and
                reversal_max_roi > 0 and
                target_roi > reversal_max_roi
            ):
                tp_mode = (
                    f"REVERSAL_STRUCTURE_CAPPED_ROI_{reversal_max_roi}%"
                )
                tp_price = get_roi_take_profit(
                    side,
                    entry_price,
                    reversal_max_roi,
                    precision,
                )
            else:
                tp_mode = (
                    f"STRUCTURE_{structure_tp['source']} "
                    f"ROI={structure_tp['target_roi']}%"
                )
                tp_price = round(structure_tp["target_price"], precision)
        else:
            tp_mode = (
                f"REVERSAL_FALLBACK_ROI_{fallback_roi}%"
                if reversal_tp
                else f"FALLBACK_ROI_{fallback_roi}%"
            )
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                fallback_roi,
                precision
            )

        if (
            not config.STATIC_TP_ENABLED
            and structure_tp
            and structure_tp.get("target_price")
            and not is_valid_take_profit(side, tp_price, market_price)
        ):
            log_warning(f"{symbol} STRUCTURE TP INVALID | USING FALLBACK ROI")
            tp_mode = (
                f"REVERSAL_FALLBACK_ROI_{fallback_roi}%"
                if reversal_tp
                else f"FALLBACK_ROI_{fallback_roi}%"
            )
            tp_price = get_roi_take_profit(
                side,
                entry_price,
                fallback_roi,
                precision
            )

        tp_price = normalize_trigger_price(
            symbol,
            side,
            "TAKE_PROFIT_MARKET",
            tp_price,
        )

        if sl_enabled and sl_price is not None:
            sl_price = normalize_trigger_price(
                symbol,
                side,
                "STOP_MARKET",
                sl_price,
            )

        details.update({
            "tp_price": tp_price,
            "tp_mode": tp_mode,
            "sl_price": sl_price if sl_enabled else None,
        })

        # ================= VALIDATION ONLY =================
        if not is_valid_take_profit(side, tp_price, market_price):
            log_warning(
                f"{symbol} TP invalid | SIDE={side} | "
                f"TP={tp_price} | MARKET={market_price} | MODE={tp_mode}"
            )
            return details if return_details else False

        if sl_enabled and sl_price is None:
            log_warning(
                f"{symbol} SL unavailable | TYPE={signal_type or 'UNKNOWN'}"
            )

            if getattr(config, "SL_INVALID_FAILS_PROTECTION_ORDER", False):
                return details if return_details else False

            sl_enabled = False
            details["sl_enabled"] = False
            details["sl_price"] = None

        if side == SIDE_BUY and sl_enabled and sl_price >= market_price:
            log_warning(
                f"{symbol} SL invalid for BUY | "
                f"SL={sl_price} | MARKET={market_price}"
            )

            if getattr(config, "SL_INVALID_FAILS_PROTECTION_ORDER", False):
                return details if return_details else False

            sl_enabled = False
            details["sl_enabled"] = False
            details["sl_price"] = None

        if side == SIDE_SELL and sl_enabled and sl_price <= market_price:
            log_warning(
                f"{symbol} SL invalid for SELL | "
                f"SL={sl_price} | MARKET={market_price}"
            )

            if getattr(config, "SL_INVALID_FAILS_PROTECTION_ORDER", False):
                return details if return_details else False

            sl_enabled = False
            details["sl_enabled"] = False
            details["sl_price"] = None

        log_info(
            f"{symbol}\nENTRY: {entry_price}\nTP: {tp_price}\n"
            f"TP_MODE: {tp_mode}\n"
            f"SL: {sl_price if sl_enabled else 'DISABLED'} | "
            f"TYPE={signal_type or 'UNKNOWN'}"
        )

        # TAKE PROFIT
        tp_order = None
        partial_quantity = 0.0
        close_pct = float(
            getattr(config, "TP1_CLOSE_POSITION_PCT", 50)
        )

        if enable_multi_tp and getattr(config, "MULTI_TP_ENABLED", False):
            try:
                tp_order, partial_quantity = place_partial_take_profit(
                    symbol,
                    side,
                    quantity,
                    close_pct,
                    tp_price,
                    position_side=position_side,
                )
            except Exception as e:
                details["multi_tp_fallback_reason"] = str(e)
                log_warning(
                    f"{symbol} partial TP placement failed; "
                    f"using full-position TP | ERROR={e}"
                )
                tp_order = None
                partial_quantity = 0.0

        if tp_order and _accepted_order_id(tp_order):
            effective_close_pct = (
                (float(partial_quantity) / abs(float(quantity))) * 100
                if abs(float(quantity or 0)) > 0
                else 0
            )
            details.update({
                "multi_tp_active": True,
                "tp1_close_pct": effective_close_pct,
                "tp1_requested_close_pct": close_pct,
                "tp1_quantity": partial_quantity,
            })
            log_info(
                f"{symbol} TP1 partial order created | "
                f"REQUESTED_CLOSE_PCT={close_pct}% | "
                f"EFFECTIVE_CLOSE_PCT={round(effective_close_pct, 4)}% | "
                f"QTY={partial_quantity}"
            )
        else:
            if enable_multi_tp and getattr(config, "MULTI_TP_ENABLED", False):
                log_warning(
                    f"{symbol} partial TP unavailable; using full-position TP"
                )

            tp_order = place_close_position_protection(
                symbol,
                side,
                "TAKE_PROFIT_MARKET",
                tp_price,
                position_side=position_side,
            )

        if not _accepted_order_id(tp_order):
            raise RuntimeError(
                f"{symbol} TP order was not accepted | RESPONSE={tp_order}"
            )

        log_info(
            f"{symbol} TP order response | "
            f"ALGO_ID={tp_order.get('algoId')} | "
            f"STATUS={tp_order.get('algoStatus')} | "
            f"TRIGGER={tp_order.get('triggerPrice')} | "
            f"TYPE={tp_order.get('orderType')}"
        )
        details["tp_order"] = tp_order

        if sl_enabled:
            time.sleep(config.PROTECTION_ORDER_DELAY_SECONDS)

            # STOP LOSS
            sl_order = place_close_position_protection(
                symbol,
                side,
                "STOP_MARKET",
                sl_price,
                position_side=position_side,
            )
            details["sl_order"] = sl_order
            details["sl_created"] = bool(_accepted_order_id(sl_order))

            if not details["sl_created"]:
                raise RuntimeError(
                    f"{symbol} SL order was not accepted | RESPONSE={sl_order}"
                )

            log_info(
                f"{symbol} SL order response | "
                f"ALGO_ID={sl_order.get('algoId')} | "
                f"STATUS={sl_order.get('algoStatus')} | "
                f"TRIGGER={sl_order.get('triggerPrice')} | "
                f"TYPE={sl_order.get('orderType')}"
            )
        else:
            log_warning(
                f"{symbol} SL DISABLED | TYPE={signal_type or 'UNKNOWN'}"
            )

        log_info(f"{symbol} TP CREATED")
        details["ok"] = True
        return details if return_details else True

    except Exception as e:
        tp_order = details.get("tp_order") or {}
        tp_order_id = _accepted_order_id(tp_order)

        if tp_order_id:
            cleanup_ok = cancel_algo_order(symbol, tp_order_id)
            details["protection_cleanup_failed"] = not cleanup_ok

            if cleanup_ok:
                details["tp_order"] = None
                details["multi_tp_active"] = False
                details["tp1_close_pct"] = None
                details["tp1_requested_close_pct"] = None
                details["tp1_quantity"] = None
            else:
                details["uncancelled_tp_order_id"] = tp_order_id
                log_error(
                    f"{symbol} TP cleanup failed after protection error; "
                    "automatic retry must remain blocked"
                )

        log_error(f"{symbol} TP/SL error: {e}")
        return details if return_details else False


# =========================
# BTC CORRELATION
# =========================

def get_btc_correlation(symbol):

    try:

        if symbol == "BTCUSDT":
            return 1.0

        coin_df = get_klines(symbol, config.TREND_TIMEFRAME, 100)
        btc_df = get_klines("BTCUSDT", config.TREND_TIMEFRAME, 100)

        if coin_df is None or btc_df is None:
            return 0

        coin_ret = coin_df['close'].pct_change().dropna()
        btc_ret = btc_df['close'].pct_change().dropna()

        return round(float(np.corrcoef(coin_ret, btc_ret)[0, 1]), 2)

    except Exception as e:
        log_error(f"{symbol} corr error: {e}")
        return 0


# =========================
# BTC TREND
# =========================
def get_btc_trend():

    try:

        btc_df = get_klines("BTCUSDT", config.TREND_TIMEFRAME)
        btc_df = apply_indicators(btc_df)

        btc = btc_df.iloc[-2]

        if btc['ema50'] > btc['ema200']:
            return "BULLISH"
        elif btc['ema50'] < btc['ema200']:
            return "BEARISH"

        return "NEUTRAL"

    except Exception as e:
        log_error(f"BTC trend error: {e}")
        return None


# =========================
# RELATIVE STRENGTH
# =========================
def get_relative_strength(symbol):

    try:

        if symbol == "BTCUSDT":
            return 0

        coin = get_klines(symbol, config.TREND_TIMEFRAME, 50)
        btc = get_klines("BTCUSDT", config.TREND_TIMEFRAME, 50)

        if coin is None or btc is None:
            return 0

        coin_r = (coin['close'].iloc[-1] - coin['close'].iloc[-10]) / coin['close'].iloc[-10] * 100
        btc_r = (btc['close'].iloc[-1] - btc['close'].iloc[-10]) / btc['close'].iloc[-10] * 100

        return round(coin_r - btc_r, 2)

    except Exception as e:
        log_error(f"{symbol} RS error: {e}")
        return 0
    
def validate_min_notional(symbol, quantity, price):

    try:

        notional = quantity * price

        # Binance futures minimum notional (safe default buffer)
        MIN_NOTIONAL = 5.0

        if notional < MIN_NOTIONAL:
            return False, notional

        return True, notional

    except Exception:
        return False, 0
