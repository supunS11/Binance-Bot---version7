from binance.client import Client
from binance.enums import *

from collections import deque
from decimal import Decimal, InvalidOperation, ROUND_DOWN
import pandas as pd
import re
import threading
import time
import numpy as np

import config
from indicators import apply_indicators
from logger import log_info, log_warning, log_error


client = Client(config.API_KEY, config.SECRET_KEY)
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

# =========================
# SYNC TIME
# =========================
server_time = client.get_server_time()
client.timestamp_offset = server_time['serverTime'] - int(time.time() * 1000)


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
        taker = _latest_item(_get_taker_longshort_ratio(params))
        data["taker_buy_sell_ratio"] = _to_float(
            taker.get("buySellRatio") if taker else None
        )
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


def get_symbol_quantity_rules(symbol):
    try:
        for item in get_exchange_info().get("symbols", []):
            if item.get("symbol") != symbol:
                continue

            filters = {
                entry.get("filterType"): entry
                for entry in item.get("filters", [])
            }
            lot_size = (
                filters.get("MARKET_LOT_SIZE") or
                filters.get("LOT_SIZE") or
                {}
            )
            return {
                "step_size": lot_size.get("stepSize", "1"),
                "min_qty": lot_size.get("minQty", "0"),
                "max_qty": lot_size.get("maxQty", "0"),
                "precision": int(item.get("quantityPrecision", 3)),
            }
    except Exception as e:
        log_warning(f"{symbol} quantity rule lookup warning: {e}")

    precision = get_symbol_precision(symbol)
    return {
        "step_size": str(10 ** -precision),
        "min_qty": "0",
        "max_qty": "0",
        "precision": precision,
    }


def normalize_order_quantity(symbol, quantity, round_down=True):
    try:
        rules = get_symbol_quantity_rules(symbol)
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
def place_market_order(symbol, side, quantity):

    try:

        order = _private_rest_call(
            f"futures_create_order:{symbol}",
            client.futures_create_order,
            symbol=symbol,
            side=side,
            type=FUTURE_ORDER_TYPE_MARKET,
            quantity=quantity,
            newOrderRespType="RESULT"
        )

        _clear_position_cache(symbol)
        log_info(f"{symbol} MARKET ORDER: {side}")
        return order

    except Exception as e:
        log_error(f"{symbol} order error: {e}")
        return None


def _is_position_side_error(error):
    message = str(error).lower()
    return (
        "position side" in message
        or "hedge" in message
        or "reduceonly" in message
        or "reduce only" in message
    )


def _submit_close_order(symbol, side, quantity, position_side=None, reduce_only=True):
    params = {
        "symbol": symbol,
        "side": side,
        "type": FUTURE_ORDER_TYPE_MARKET,
        "quantity": quantity,
        "newOrderRespType": "RESULT",
    }

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


def close_position_market(symbol, amount, position_side=None):

    try:
        amount = float(amount)
        quantity = abs(amount)

        if quantity <= 0:
            return None

        position_side = (position_side or "").upper()

        if position_side in ("LONG", "SHORT"):
            side = SIDE_SELL if position_side == "LONG" else SIDE_BUY
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                position_side=position_side,
                reduce_only=False
            )
            log_warning(
                f"{symbol} HEDGE CLOSE ORDER: {side} | POSITION_SIDE={position_side}"
            )
            return order

        side = SIDE_SELL if amount > 0 else SIDE_BUY

        try:
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                reduce_only=True
            )
            log_warning(f"{symbol} REDUCE-ONLY CLOSE ORDER: {side}")
            return order

        except Exception as e:
            if not _is_position_side_error(e):
                raise

            inferred_position_side = "LONG" if amount > 0 else "SHORT"
            log_warning(
                f"{symbol} reduce-only close failed; retrying hedge close | "
                f"POSITION_SIDE={inferred_position_side} | ERROR={e}"
            )
            order = _submit_close_order(
                symbol,
                side,
                quantity,
                position_side=inferred_position_side,
                reduce_only=False
            )
            log_warning(
                f"{symbol} HEDGE CLOSE ORDER: {side} | "
                f"POSITION_SIDE={inferred_position_side}"
            )
            return order

    except Exception as e:
        log_error(f"{symbol} close position error: {e}")
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


def place_partial_take_profit(
    symbol,
    side,
    total_quantity,
    close_pct,
    trigger_price,
    position_side=None,
):
    close_pct = min(max(float(close_pct), 0), 100)

    if close_pct <= 0 or close_pct >= 100:
        return None, 0.0

    total_quantity = normalize_order_quantity(symbol, total_quantity)
    partial_quantity = normalize_order_quantity(
        symbol,
        total_quantity * close_pct / 100,
    )
    remaining_quantity = normalize_order_quantity(
        symbol,
        total_quantity - partial_quantity,
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

    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
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


def place_close_position_protection(
    symbol,
    side,
    order_type,
    trigger_price,
    position_side=None,
):
    close_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
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

        sl_order = place_algo_order(
            algoType="CONDITIONAL",
            symbol=symbol,
            side=close_side,
            type="STOP_MARKET",
            triggerPrice=sl_price,
            closePosition="true",
            workingType="MARK_PRICE",
            priceProtect="TRUE"
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
            details.update({
                "multi_tp_active": True,
                "tp1_close_pct": close_pct,
                "tp1_quantity": partial_quantity,
            })
            log_info(
                f"{symbol} TP1 partial order created | "
                f"CLOSE_PCT={close_pct}% | QTY={partial_quantity}"
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
            details["sl_created"] = bool(
                _accepted_order_id(sl_order)
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
