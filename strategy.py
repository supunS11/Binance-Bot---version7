from collections import OrderedDict
from copy import deepcopy
import threading

import numpy as np

import config
from logger import log_info, log_error, log_warning


_signal_analysis_cache = OrderedDict()
_signal_analysis_cache_lock = threading.RLock()


def score_to_confidence(score, max_score=None):
    if score <= 0:
        return 0

    if max_score is None:
        max_score = get_config_float("LONG_TERM_CONFIDENCE_MAX_SCORE", 42)

    return round(min((score / max(max_score, 1)) * 100, 100), 2)


def score_to_uncapped_index(score, max_score=None):
    if score <= 0:
        return 0

    if max_score is None:
        max_score = get_config_float("LONG_TERM_CONFIDENCE_MAX_SCORE", 42)

    return round((score / max(max_score, 1)) * 100, 2)


def get_config_float(name, default):
    try:
        return float(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def get_config_int(name, default):
    try:
        return int(getattr(config, name, default))
    except (TypeError, ValueError):
        return default


def latest_closed(df):
    return df.iloc[-2] if len(df) > 1 else df.iloc[-1]


def previous_closed(df):
    return df.iloc[-3] if len(df) > 2 else df.iloc[-2]


def pct_distance(a, b):
    if not b:
        return 0

    return abs(a - b) / b * 100


def add_score(score, condition, points):
    return score + points if condition else score


def _safe_float(value, default=0):
    try:
        result = float(value)

        if result != result:
            return default

        return result
    except (TypeError, ValueError):
        return default


def get_structure_stop_loss(df, side):
    try:
        candle = latest_closed(df)
        atr = candle["atr"]

        if atr <= 0:
            return None

        if side == "BUY":
            swing_low = df["low"].iloc[-20:-1].min()
            return swing_low - (atr * 1.5)

        swing_high = df["high"].iloc[-20:-1].max()
        return swing_high + (atr * 1.5)

    except Exception as e:
        log_error(f"STRUCTURE REFERENCE ERROR: {e}")
        return None


def detect_market_structure(df):
    try:
        recent_high = df["high"].iloc[-30:-5].max()
        recent_low = df["low"].iloc[-30:-5].min()
        prev_high = df["high"].iloc[-60:-30].max()
        prev_low = df["low"].iloc[-60:-30].min()
        close = latest_closed(df)["close"]

        return {
            "bullish_structure": recent_high > prev_high and recent_low > prev_low,
            "bearish_structure": recent_high < prev_high and recent_low < prev_low,
            "bullish_breakout": close > recent_high,
            "bearish_breakdown": close < recent_low,
        }

    except Exception:
        return {
            "bullish_structure": False,
            "bearish_structure": False,
            "bullish_breakout": False,
            "bearish_breakdown": False,
        }


def _level_tolerance(df):
    latest = latest_closed(df)
    price = latest["close"]
    atr = latest["atr"] if "atr" in latest.index else 0
    pct_tolerance = price * (
        get_config_float("LONG_TERM_SR_TOLERANCE_PCT", 1.0) / 100
    )
    atr_tolerance = atr * get_config_float("LONG_TERM_SR_ATR_TOLERANCE", 0.75)
    max_tolerance = price * (
        get_config_float("LONG_TERM_SR_MAX_TOLERANCE_PCT", 1.25) / 100
    )
    raw_tolerance = max(pct_tolerance, atr_tolerance, price * 0.002)

    return min(raw_tolerance, max_tolerance) if max_tolerance > 0 else raw_tolerance


def _touch_positions(data, column, level, tolerance):
    values = np.asarray(data[column], dtype=float)
    mask = np.isfinite(values) & (np.abs(values - level) <= tolerance)
    return np.flatnonzero(mask).tolist()


def _distinct_touch_count(positions, min_gap):
    touches = 0
    last_pos = None

    for pos in positions:
        if last_pos is None or pos - last_pos >= min_gap:
            touches += 1
            last_pos = pos

    return touches


def _level_reaction_score(data, side, level, tolerance):
    column = "low" if side == "BUY" else "high"
    touch_values = np.asarray(data[column], dtype=float)
    touch_mask = (
        np.isfinite(touch_values) &
        (np.abs(touch_values - level) <= tolerance)
    )

    if not np.any(touch_mask):
        return 0

    high = np.asarray(data["high"], dtype=float)[touch_mask]
    low = np.asarray(data["low"], dtype=float)[touch_mask]
    close = np.asarray(data["close"], dtype=float)[touch_mask]
    open_price = np.asarray(data["open"], dtype=float)[touch_mask]

    high = np.nan_to_num(high, nan=0.0)
    low = np.nan_to_num(low, nan=0.0)
    close = np.nan_to_num(close, nan=0.0)
    open_price = np.nan_to_num(open_price, nan=0.0)
    candle_range = high - low
    valid_range = candle_range > 0

    if not np.any(valid_range):
        return 0

    high = high[valid_range]
    low = low[valid_range]
    close = close[valid_range]
    open_price = open_price[valid_range]
    candle_range = candle_range[valid_range]

    if side == "BUY":
        directional_close = (close - low) / candle_range
        wick_rejection = (np.minimum(open_price, close) - low) / candle_range
    else:
        directional_close = (high - close) / candle_range
        wick_rejection = (high - np.maximum(open_price, close)) / candle_range

    reactions = np.maximum(np.maximum(directional_close, wick_rejection), 0)

    if reactions.size > 5:
        reactions = np.partition(reactions, reactions.size - 5)[-5:]

    return min(float(np.mean(reactions)), 1)


def _volume_touch_score(data, column, level, tolerance):
    if "volume_sma" not in data.columns or "volume" not in data.columns:
        return 0

    touch_values = np.asarray(data[column], dtype=float)
    touch_mask = (
        np.isfinite(touch_values) &
        (np.abs(touch_values - level) <= tolerance)
    )
    volume = np.nan_to_num(np.asarray(data["volume"], dtype=float), nan=0.0)
    volume_sma = np.nan_to_num(
        np.asarray(data["volume_sma"], dtype=float),
        nan=0.0,
    )
    strong_touches = int(np.count_nonzero(touch_mask & (volume > volume_sma)))

    return min(strong_touches / 3, 1)


def _recent_break_penalty(data, side, level, tolerance):
    recent_close = np.asarray(data["close"].tail(5), dtype=float)

    if side == "BUY":
        broken = bool(np.any(recent_close < level - tolerance))
    else:
        broken = bool(np.any(recent_close > level + tolerance))

    if not broken:
        return 0

    return get_config_float("LONG_TERM_SR_RECENT_BREAK_PENALTY", 1.5)


def _level_strength_score(
    data,
    side,
    column,
    level,
    positions,
    timeframe_weight,
):
    min_gap = max(get_config_int("LONG_TERM_SR_TOUCH_MIN_GAP", 5), 1)
    distinct_touches = _distinct_touch_count(positions, min_gap)
    recency_score = (max(positions) / len(data)) if positions else 0
    tolerance = _level_tolerance(data)
    reaction_score = _level_reaction_score(data, side, level, tolerance)
    volume_score = _volume_touch_score(data, column, level, tolerance)
    penalty = _recent_break_penalty(data, side, level, tolerance)
    score = (
        distinct_touches * timeframe_weight +
        recency_score +
        reaction_score * get_config_float("LONG_TERM_SR_REACTION_BONUS", 1.0) +
        volume_score * get_config_float("LONG_TERM_SR_VOLUME_BONUS", 0.5) -
        penalty
    )

    return {
        "score": round(max(score, 0), 2),
        "touches": distinct_touches,
        "reaction_score": round(float(reaction_score), 2),
        "volume_score": round(float(volume_score), 2),
        "recent_break_penalty": round(float(penalty), 2),
    }


def _collect_pivot_levels(df, side, label, timeframe_weight):
    lookback = get_config_int("LONG_TERM_SR_LOOKBACK", 160)
    min_touches = get_config_int("LONG_TERM_SR_MIN_TOUCHES", 2)
    data = df.tail(lookback).copy()

    if len(data) < 20:
        return []

    tolerance = _level_tolerance(data)
    swing = max(get_config_int("LONG_TERM_SR_SWING", 3), 2)
    column = "low" if side == "BUY" else "high"
    levels = []
    values = np.asarray(data[column], dtype=float)

    for pos in range(swing, len(data) - swing):
        level = float(values[pos])
        window_values = values[pos - swing:pos + swing + 1]

        if side == "BUY" and level > np.nanmin(window_values):
            continue

        if side == "SELL" and level < np.nanmax(window_values):
            continue

        positions = _touch_positions(data, column, level, tolerance)
        strength = _level_strength_score(
            data,
            side,
            column,
            level,
            positions,
            timeframe_weight,
        )

        if strength["touches"] < min_touches:
            continue

        levels.append({
            "level": level,
            "score": strength["score"],
            "touches": strength["touches"],
            "reaction_score": strength["reaction_score"],
            "volume_score": strength["volume_score"],
            "recent_break_penalty": strength["recent_break_penalty"],
            "source": f"{label}_pivot",
        })

    return levels


def _collect_ema_levels(df, side, label, timeframe_weight):
    lookback = get_config_int("LONG_TERM_SR_LOOKBACK", 160)
    data = df.tail(lookback).copy()
    latest = latest_closed(data)
    levels = []
    min_respects = max(get_config_int("LONG_TERM_SR_EMA_MIN_RESPECTS", 2), 0)
    tolerance = _level_tolerance(data)

    for ema_name, bonus in (("ema50", 0.75), ("ema200", 1.25)):
        if ema_name not in latest.index:
            continue

        level = float(latest[ema_name])
        close = float(latest["close"])

        if side == "BUY" and level >= close:
            continue

        if side == "SELL" and level <= close:
            continue

        if side == "BUY":
            respects = data[
                (data["low"] <= level + tolerance) &
                (data["close"] >= level)
            ]
            column = "low"
        else:
            respects = data[
                (data["high"] >= level - tolerance) &
                (data["close"] <= level)
            ]
            column = "high"

        if len(respects) < min_respects:
            continue

        reaction_score = _level_reaction_score(data, side, level, tolerance)
        volume_score = _volume_touch_score(data, column, level, tolerance)
        penalty = _recent_break_penalty(data, side, level, tolerance)
        score = (
            timeframe_weight +
            bonus +
            min(len(respects) * 0.25, 1.25) +
            reaction_score * 0.50 +
            volume_score * get_config_float("LONG_TERM_SR_VOLUME_BONUS", 0.5) -
            penalty
        )

        levels.append({
            "level": level,
            "score": round(max(score, 0), 2),
            "touches": int(len(respects)),
            "reaction_score": round(float(reaction_score), 2),
            "volume_score": round(float(volume_score), 2),
            "recent_break_penalty": round(float(penalty), 2),
            "source": f"{label}_{ema_name}",
        })

    return levels


def _collect_range_levels(df, side, label, timeframe_weight):
    lookback = get_config_int("LONG_TERM_SR_LOOKBACK", 160)
    data = df.iloc[:-1].tail(lookback).copy() if len(df) > 1 else df.tail(lookback)

    if len(data) < 20:
        return []

    tolerance = _level_tolerance(data)
    column = "low" if side == "BUY" else "high"
    levels = []

    for window in (20, 50, 100):
        if len(data) < window:
            continue

        recent = data.tail(window)
        level = recent[column].min() if side == "BUY" else recent[column].max()
        positions = _touch_positions(data, column, float(level), tolerance)
        strength = _level_strength_score(
            data,
            side,
            column,
            float(level),
            positions,
            timeframe_weight,
        )
        min_touches = max(get_config_int("LONG_TERM_SR_RANGE_MIN_TOUCHES", 2), 1)

        if strength["touches"] < min_touches:
            continue

        score = (
            timeframe_weight +
            min(window / 100, 1) +
            strength["touches"] * 0.35 +
            strength["reaction_score"] * get_config_float(
                "LONG_TERM_SR_REACTION_BONUS",
                1.0,
            ) +
            strength["volume_score"] * get_config_float(
                "LONG_TERM_SR_VOLUME_BONUS",
                0.5,
            ) -
            strength["recent_break_penalty"]
        )

        levels.append({
            "level": float(level),
            "score": round(max(score, 0), 2),
            "touches": strength["touches"],
            "reaction_score": strength["reaction_score"],
            "volume_score": strength["volume_score"],
            "recent_break_penalty": strength["recent_break_penalty"],
            "source": f"{label}_{window}_range",
        })

    return levels


def _dedupe_levels(levels, tolerance):
    deduped = []
    confluence_bonus = get_config_float("LONG_TERM_SR_CONFLUENCE_BONUS", 0.75)

    for level in sorted(levels, key=lambda item: item["score"], reverse=True):
        match = next(
            (
                item for item in deduped
                if abs(level["level"] - item["level"]) <= tolerance
            ),
            None
        )

        if match:
            match["score"] = round(
                match["score"] +
                min(level["score"] * 0.20, confluence_bonus),
                2
            )
            match["touches"] = max(
                int(match.get("touches", 0)),
                int(level.get("touches", 0)),
            )
            match["source"] = f"{match['source']}+{level['source']}"
            match["confluence_count"] = int(match.get("confluence_count", 1)) + 1
            continue

        item = level.copy()
        item["confluence_count"] = 1
        deduped.append(item)

    return deduped


def _take_profit_buffer(entry_price, trend_df, confirm_df):
    pct_buffer = entry_price * (
        get_config_float("STRUCTURE_TP_BUFFER_PCT", 0.15) / 100
    )
    atr_values = []

    for df in (confirm_df, trend_df):
        try:
            atr = latest_closed(df)["atr"]

            if atr > 0:
                atr_values.append(float(atr))
        except Exception:
            continue

    atr_buffer = 0

    if atr_values:
        atr_buffer = min(atr_values) * get_config_float(
            "STRUCTURE_TP_ATR_BUFFER_MULT",
            0.25
        )

    return max(pct_buffer, atr_buffer, entry_price * 0.0005)


def find_structure_take_profit(side, entry_price, trend_df, confirm_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    target_side = "SELL" if side == "BUY" else "BUY"
    min_roi = get_config_float("STRUCTURE_TP_MIN_ROI", 8)
    max_roi = get_config_float("STRUCTURE_TP_MAX_ROI", 120)
    min_score = get_config_float("STRUCTURE_TP_MIN_SCORE", 2.0)
    buffer = _take_profit_buffer(entry_price, trend_df, confirm_df)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_pivot_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_range_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_range_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_ema_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_ema_levels(confirm_df, target_side, "4h", 1.25))

    tolerance = max(_level_tolerance(trend_df), _level_tolerance(confirm_df))
    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = candidate["level"]

        if candidate["score"] < min_score:
            continue

        if side == "BUY":
            if level <= entry_price:
                continue

            target_price = level - buffer

            if target_price <= entry_price:
                continue

            roi = ((target_price - entry_price) / entry_price) * leverage_to_use * 100
        else:
            if level >= entry_price:
                continue

            target_price = level + buffer

            if target_price >= entry_price:
                continue

            roi = ((entry_price - target_price) / entry_price) * leverage_to_use * 100

        if roi < min_roi or roi > max_roi:
            continue

        item = candidate.copy()
        item["target_price"] = float(target_price)
        item["raw_level"] = float(level)
        item["target_roi"] = round(float(roi), 2)
        item["buffer"] = round(float(buffer), 8)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["target_roi"], -item["score"]))
    return valid[0]


def validate_structure_take_profit(side, entry_price, trend_df, confirm_df, leverage=None):
    target = find_structure_take_profit(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )

    if target:
        return True, target

    return False, {
        "reason": "NO VALID STRUCTURE TP LEVEL FOUND"
    }


def find_nearest_profit_room_level(side, entry_price, trend_df, confirm_df, leverage=None):
    if entry_price <= 0:
        return None

    leverage_to_use = leverage or config.LEVERAGE
    target_side = "SELL" if side == "BUY" else "BUY"
    min_score = get_config_float(
        "ENTRY_TP_ROOM_MIN_LEVEL_SCORE",
        get_config_float("STRUCTURE_TP_MIN_SCORE", 2.0)
    )
    buffer = _take_profit_buffer(entry_price, trend_df, confirm_df)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_pivot_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_range_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_range_levels(confirm_df, target_side, "4h", 1.25))
    candidates.extend(_collect_ema_levels(trend_df, target_side, "1d", 2.0))
    candidates.extend(_collect_ema_levels(confirm_df, target_side, "4h", 1.25))

    tolerance = max(_level_tolerance(trend_df), _level_tolerance(confirm_df))
    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = float(candidate["level"])

        if candidate["score"] < min_score:
            continue

        if side == "BUY":
            if level <= entry_price:
                continue

            target_price = level - buffer
            raw_roi = ((level - entry_price) / entry_price) * leverage_to_use * 100
            target_roi = (
                ((target_price - entry_price) / entry_price) *
                leverage_to_use *
                100
            )
        else:
            if level >= entry_price:
                continue

            target_price = level + buffer
            raw_roi = ((entry_price - level) / entry_price) * leverage_to_use * 100
            target_roi = (
                ((entry_price - target_price) / entry_price) *
                leverage_to_use *
                100
            )

        item = candidate.copy()
        item["raw_level"] = level
        item["target_price"] = float(target_price)
        item["raw_level_roi"] = round(float(raw_roi), 2)
        item["target_roi"] = round(float(target_roi), 2)
        item["buffer"] = round(float(buffer), 8)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["raw_level_roi"], -item["score"]))
    return valid[0]


def validate_entry_profit_room(
    side,
    entry_price,
    trend_df,
    confirm_df,
    leverage=None,
    min_roi_override=None
):
    if not getattr(config, "ENTRY_TP_ROOM_CHECK_ENABLED", True):
        return True, {"reason": "ENTRY_TP_ROOM_CHECK_DISABLED"}

    min_roi = (
        min_roi_override
        if min_roi_override is not None
        else get_config_float(
            "ENTRY_MIN_TP_ROOM_ROI",
            get_config_float("STRUCTURE_TP_MIN_ROI", 8)
        )
    )
    level = find_nearest_profit_room_level(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )
    label = "RESISTANCE" if side == "BUY" else "SUPPORT"

    if not level:
        if getattr(config, "ENTRY_TP_ROOM_BLOCK_IF_NO_LEVEL", False):
            return False, {
                "reason": f"NO {label} PROFIT ROOM LEVEL FOUND"
            }

        return True, {
            "reason": f"NO CLEAR {label} FOUND; PROFIT ROOM ALLOWED"
        }

    if level["target_roi"] < min_roi:
        return False, {
            "reason": (
                f"TOO CLOSE TO {label} | "
                f"ROOM={level['target_roi']}% < MIN={min_roi}%"
            ),
            **level
        }

    return True, {
        "reason": "ENTRY_PROFIT_ROOM_OK",
        **level
    }


def _closed_data(df, lookback=None):
    data = df.iloc[:-1].copy() if len(df) > 1 else df.copy()

    if lookback:
        data = data.tail(lookback)

    return data


def _body(candle):
    return abs(float(candle["close"]) - float(candle["open"]))


def _is_bullish(candle):
    return candle["close"] > candle["open"]


def _is_bearish(candle):
    return candle["close"] < candle["open"]


def _candle_atr(candle):
    try:
        return max(float(candle["atr"]), 1e-10)
    except Exception:
        return max(float(candle["high"] - candle["low"]), 1e-10)


def _average_range(df, period=14):
    data = _closed_data(df, period)

    if len(data) == 0:
        return 0

    ranges = data["high"] - data["low"]
    value = ranges.mean()

    return max(float(value), 1e-10)


def _close_position(candle):
    high = float(candle["high"])
    low = float(candle["low"])
    close = float(candle["close"])
    candle_range = high - low

    if candle_range <= 0:
        return 0.5

    return (close - low) / candle_range


def _adverse_zone(side, entry_price, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    max_adverse_roi = abs(get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50))
    max_price_move = (max_adverse_roi / max(leverage_to_use, 1)) / 100

    if side == "BUY":
        return entry_price * (1 - max_price_move), entry_price

    return entry_price, entry_price * (1 + max_price_move)


def _live_entry_timeframe_check(side, df, mark_price, label, confidence=None):
    data = _closed_data(df)
    lookback = get_config_int("LIVE_ENTRY_STRUCTURE_LOOKBACK", 12)

    if len(data) < lookback + 2:
        return {
            "label": label,
            "block": False,
            "structure_break": False,
            "opposite_reversal": False,
            "ema_wrong_side": False,
            "ema_chase": False,
            "close_chase": False,
            "ema20": None,
            "ema_distance_pct": 0,
            "ema_chase_atr": 0,
            "close_position": 0,
            "body_atr": 0,
            "support_score": 0,
            "supports_direction": False,
            "opposes_direction": False,
            "mark_price": round(float(mark_price), 8) if mark_price else None,
            "latest_close": None,
            "reason": "INSUFFICIENT_DATA",
        }

    latest = data.iloc[-1]
    previous = data.iloc[-lookback - 1:-1]
    atr = max(_safe_float(_average_range(df, 14)), 1e-10)
    structure_buffer = atr * get_config_float(
        "LIVE_ENTRY_STRUCTURE_BUFFER_ATR",
        0.08
    )
    retrace_atr = get_config_float("MAX_LIVE_ENTRY_RETRACE_ATR", 0.20)
    min_body_atr = get_config_float("LIVE_ENTRY_MIN_REVERSAL_BODY_ATR", 0.35)
    close_pos_limit = get_config_float("LIVE_ENTRY_REVERSAL_CLOSE_POSITION", 0.30)
    max_chase_atr = get_config_float("MAX_LIVE_ENTRY_CHASE_ATR", 0.50)

    if (
        getattr(config, "LIVE_ENTRY_CHASE_RELAX_ENABLED", False)
        and confidence is not None
        and _safe_float(confidence) >= get_config_float(
            "LIVE_ENTRY_CHASE_RELAX_MIN_CONFIDENCE",
            95.0
        )
    ):
        max_chase_atr = max(
            max_chase_atr,
            get_config_float("LIVE_ENTRY_CHASE_RELAXED_MAX_ATR", 1.2)
        )

    ema_tolerance_pct = get_config_float("LIVE_ENTRY_EMA_TOLERANCE_PCT", 0.08)
    max_close_position = get_config_float("MAX_LIVE_ENTRY_CLOSE_POSITION", 0.88)
    close_position = _close_position(latest)
    body_atr = _body(latest) / atr
    ema20 = _safe_float(latest.get("ema20"))
    ema_distance_pct = pct_distance(mark_price, ema20) if ema20 else 0
    ema_chase_atr = abs(mark_price - ema20) / atr if ema20 else 0
    ema_tolerance = ema20 * (ema_tolerance_pct / 100) if ema20 else 0
    macd = _safe_float(latest.get("macd"))
    macd_signal = _safe_float(latest.get("macd_signal"))
    rsi = _safe_float(latest.get("rsi"), 50)
    ema_wrong_side = False
    ema_chase = False
    close_chase = False

    if side == "BUY":
        recent_low = float(previous["low"].min())
        structure_break = mark_price < recent_low - structure_buffer
        opposite_reversal = (
            _is_bearish(latest)
            and body_atr >= min_body_atr
            and close_position <= close_pos_limit
            and mark_price <= float(latest["close"]) - (atr * retrace_atr)
        )
        if ema20:
            ema_wrong_side = mark_price < ema20 - ema_tolerance
            ema_chase = (
                max_chase_atr > 0 and
                mark_price > ema20 and
                ema_chase_atr > max_chase_atr
            )
        close_chase = (
            max_close_position > 0 and
            close_position > max_close_position
        )
        direction_ok = _is_bullish(latest)
        opposite_direction = _is_bearish(latest)
        directional_close = close_position
        ema_support = bool(ema20 and mark_price >= ema20 - ema_tolerance)
        oscillator_support = macd > macd_signal or rsi >= 50
        oscillator_opposes = macd < macd_signal and rsi < 48
        mark_support = mark_price >= float(latest["close"])
        reason = "BUY_GUARD"
    else:
        recent_high = float(previous["high"].max())
        structure_break = mark_price > recent_high + structure_buffer
        opposite_reversal = (
            _is_bullish(latest)
            and body_atr >= min_body_atr
            and close_position >= 1 - close_pos_limit
            and mark_price >= float(latest["close"]) + (atr * retrace_atr)
        )
        if ema20:
            ema_wrong_side = mark_price > ema20 + ema_tolerance
            ema_chase = (
                max_chase_atr > 0 and
                mark_price < ema20 and
                ema_chase_atr > max_chase_atr
            )
        close_chase = (
            max_close_position > 0 and
            close_position < 1 - max_close_position
        )
        direction_ok = _is_bearish(latest)
        opposite_direction = _is_bullish(latest)
        directional_close = 1 - close_position
        ema_support = bool(ema20 and mark_price <= ema20 + ema_tolerance)
        oscillator_support = macd < macd_signal or rsi <= 50
        oscillator_opposes = macd > macd_signal and rsi > 52
        mark_support = mark_price <= float(latest["close"])
        reason = "SELL_GUARD"

    support_score = 0
    support_score += 0.50 if direction_ok else 0
    support_score -= 0.50 if opposite_direction else 0
    support_score += 0.75 if ema_support else 0
    support_score -= 0.75 if ema_wrong_side else 0
    support_score += 0.35 if oscillator_support else 0
    support_score -= 0.35 if oscillator_opposes else 0
    support_score += 0.35 if directional_close >= 0.50 else 0
    support_score -= 0.35 if directional_close < 0.40 else 0
    support_score += 0.25 if mark_support else 0
    support_score -= 0.50 if opposite_reversal else 0

    min_support_score = get_config_float("LIVE_ENTRY_MIN_SUPPORT_SCORE", 1.0)
    opposition_score = get_config_float(
        "LIVE_ENTRY_DUAL_OPPOSITION_BLOCK_SCORE",
        -0.75
    )
    supports_direction = support_score >= min_support_score
    opposes_direction = support_score <= opposition_score

    return {
        "label": label,
        "block": structure_break or opposite_reversal or ema_wrong_side or ema_chase,
        "structure_break": structure_break,
        "opposite_reversal": opposite_reversal,
        "ema_wrong_side": ema_wrong_side,
        "ema_chase": ema_chase,
        "close_chase": close_chase,
        "ema20": round(float(ema20), 8) if ema20 else None,
        "ema_distance_pct": round(float(ema_distance_pct), 3),
        "ema_chase_atr": round(float(ema_chase_atr), 2),
        "close_position": round(float(close_position), 2),
        "body_atr": round(float(body_atr), 2),
        "support_score": round(float(support_score), 2),
        "supports_direction": supports_direction,
        "opposes_direction": opposes_direction,
        "mark_price": round(float(mark_price), 8),
        "latest_close": round(float(latest["close"]), 8),
        "reason": reason,
    }


def validate_live_entry_guard(
    side,
    fast_df,
    slow_df,
    mark_price,
    require_both_override=None,
    confidence=None
):
    if not config.LIVE_ENTRY_CONFIRMATION_ENABLED:
        return True, {"reason": "LIVE_ENTRY_GUARD_DISABLED"}

    if fast_df is None or slow_df is None or mark_price is None:
        if config.LIVE_ENTRY_REQUIRE_DATA:
            return False, {"reason": "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE"}

        return True, {"reason": "LIVE_ENTRY_GUARD_DATA_UNAVAILABLE_ALLOWED"}

    fast = _live_entry_timeframe_check(
        side,
        fast_df,
        mark_price,
        config.LIVE_ENTRY_FAST_TIMEFRAME,
        confidence=confidence
    )
    slow = _live_entry_timeframe_check(
        side,
        slow_df,
        mark_price,
        config.LIVE_ENTRY_SLOW_TIMEFRAME,
        confidence=confidence
    )
    structure_break = fast["structure_break"] or slow["structure_break"]
    dual_reversal = fast["opposite_reversal"] and slow["opposite_reversal"]
    dual_ema_wrong_side = fast["ema_wrong_side"] and slow["ema_wrong_side"]
    dual_ema_chase = fast["ema_chase"] and slow["ema_chase"]
    dual_close_chase = fast["close_chase"] and slow["close_chase"]
    live_ema_block = (
        dual_ema_wrong_side or
        dual_ema_chase or
        (dual_close_chase and (fast["ema_chase"] or slow["ema_chase"]))
    )
    support_count = int(fast.get("supports_direction", False)) + int(
        slow.get("supports_direction", False)
    )
    opposition_count = int(fast.get("opposes_direction", False)) + int(
        slow.get("opposes_direction", False)
    )

    if structure_break or dual_reversal or live_ema_block:
        if structure_break:
            reason = "OPPOSITE_STRUCTURE_BREAK"
        elif dual_reversal:
            reason = "DUAL_OPPOSITE_REVERSAL"
        elif dual_ema_wrong_side:
            reason = "DUAL_LIVE_EMA_WRONG_SIDE"
        elif dual_ema_chase:
            reason = "DUAL_LIVE_EMA_CHASE"
        else:
            reason = "DUAL_LIVE_CLOSE_CHASE"
        return False, {
            "reason": reason,
            "fast": fast,
            "slow": slow,
            "mark_price": mark_price,
        }

    if getattr(config, "LIVE_ENTRY_REQUIRE_DIRECTION_SUPPORT", True):
        require_both = (
            bool(require_both_override)
            if require_both_override is not None
            else bool(
                getattr(config, "LIVE_ENTRY_REQUIRE_BOTH_TIMEFRAMES", False)
            )
        )

        if opposition_count >= 2:
            return False, {
                "reason": "DUAL_LIVE_DIRECTION_OPPOSITION",
                "fast": fast,
                "slow": slow,
                "mark_price": mark_price,
                "support_count": support_count,
                "opposition_count": opposition_count,
            }

        if require_both and support_count < 2:
            return False, {
                "reason": "LIVE_DIRECTION_SUPPORT_MISSING_BOTH",
                "fast": fast,
                "slow": slow,
                "mark_price": mark_price,
                "support_count": support_count,
                "opposition_count": opposition_count,
            }

        if not require_both and support_count < 1:
            return False, {
                "reason": "LIVE_DIRECTION_SUPPORT_MISSING",
                "fast": fast,
                "slow": slow,
                "mark_price": mark_price,
                "support_count": support_count,
                "opposition_count": opposition_count,
            }

    return True, {
        "reason": "LIVE_ENTRY_GUARD_OK",
        "fast": fast,
        "slow": slow,
        "mark_price": mark_price,
        "support_count": support_count,
        "opposition_count": opposition_count,
    }


def validate_dca_recovery_confirmation(side, fast_df, slow_df, mark_price):
    """Require an actual lower-timeframe recovery before the one DCA add."""
    if not getattr(config, "DCA_RECOVERY_CONFIRMATION_ENABLED", True):
        return True, {"reason": "DCA_RECOVERY_CONFIRMATION_DISABLED"}

    if fast_df is None or slow_df is None or mark_price is None:
        allowed = not getattr(config, "DCA_RECOVERY_REQUIRE_DATA", True)
        return allowed, {
            "reason": (
                "DCA_RECOVERY_DATA_UNAVAILABLE_ALLOWED"
                if allowed
                else "DCA_RECOVERY_DATA_UNAVAILABLE"
            )
        }

    fast = _live_entry_timeframe_check(
        side,
        fast_df,
        mark_price,
        config.LIVE_ENTRY_FAST_TIMEFRAME,
    )
    slow = _live_entry_timeframe_check(
        side,
        slow_df,
        mark_price,
        config.LIVE_ENTRY_SLOW_TIMEFRAME,
    )
    support_count = int(fast.get("supports_direction", False)) + int(
        slow.get("supports_direction", False)
    )
    opposition_count = int(fast.get("opposes_direction", False)) + int(
        slow.get("opposes_direction", False)
    )
    hard_block = bool(
        fast.get("structure_break") or
        slow.get("structure_break") or
        fast.get("opposite_reversal") or
        slow.get("opposite_reversal") or
        opposition_count > 0
    )
    require_both = bool(
        getattr(config, "DCA_RECOVERY_REQUIRE_BOTH_TIMEFRAMES", True)
    )
    required_support = 2 if require_both else 1
    allowed = not hard_block and support_count >= required_support

    return allowed, {
        "reason": (
            "DCA_RECOVERY_CONFIRMED"
            if allowed
            else "DCA_RECOVERY_NOT_CONFIRMED"
        ),
        "fast": fast,
        "slow": slow,
        "support_count": support_count,
        "required_support": required_support,
        "opposition_count": opposition_count,
        "hard_block": hard_block,
    }


def evaluate_time_exit_weakness(side, trend_df, confirm_df):
    """Score 4h deterioration, with the 1d frame used only as extra evidence."""
    unavailable = {
        "should_exit": False,
        "reason": "TIME_EXIT_DATA_UNAVAILABLE",
        "weakness_score": 0,
        "evidence": [],
    }

    try:
        if trend_df is None or confirm_df is None:
            return unavailable

        trend = latest_closed(trend_df)
        confirm = latest_closed(confirm_df)
        confirm_previous = previous_closed(confirm_df)
        evidence = []
        confirm_weak = False

        if side == "BUY":
            checks = (
                ("4H_CLOSE_BELOW_EMA20", confirm["close"] < confirm["ema20"], True),
                ("4H_EMA20_BELOW_EMA50", confirm["ema20"] < confirm["ema50"], True),
                ("4H_MACD_BEARISH", confirm["macd"] < confirm["macd_signal"], True),
                ("4H_STRUCTURE_BREAK", confirm["close"] < confirm_previous["low"], True),
                ("1D_CLOSE_BELOW_EMA50", trend["close"] < trend["ema50"], False),
                ("1D_MACD_BEARISH", trend["macd"] < trend["macd_signal"], False),
            )
        else:
            checks = (
                ("4H_CLOSE_ABOVE_EMA20", confirm["close"] > confirm["ema20"], True),
                ("4H_EMA20_ABOVE_EMA50", confirm["ema20"] > confirm["ema50"], True),
                ("4H_MACD_BULLISH", confirm["macd"] > confirm["macd_signal"], True),
                ("4H_STRUCTURE_BREAK", confirm["close"] > confirm_previous["high"], True),
                ("1D_CLOSE_ABOVE_EMA50", trend["close"] > trend["ema50"], False),
                ("1D_MACD_BULLISH", trend["macd"] > trend["macd_signal"], False),
            )

        for label, active, confirmation_evidence in checks:
            if active:
                evidence.append(label)
                confirm_weak = confirm_weak or confirmation_evidence

        score = len(evidence)
        required_score = max(
            get_config_float("TIME_EXIT_MIN_WEAKNESS_SCORE", 2),
            1,
        )
        should_exit = bool(confirm_weak and score >= required_score)
        return {
            "should_exit": should_exit,
            "reason": (
                "TIME_EXIT_TREND_WEAKENED"
                if should_exit
                else "TIME_EXIT_WEAKNESS_INCOMPLETE"
            ),
            "weakness_score": score,
            "required_score": required_score,
            "evidence": evidence,
        }
    except Exception:
        return unavailable


def evaluate_route_early_invalidation(
    side,
    fast_df,
    slow_df,
    mark_price,
    confirmation_type=None,
    reference_price=None,
):
    route = (
        "REVERSAL"
        if str(confirmation_type or "").upper() == "REVERSAL"
        else "TREND"
    )
    unavailable = {
        "should_exit": False,
        "reason": "EARLY_INVALIDATION_DATA_UNAVAILABLE",
        "route": route,
        "reference_broken": False,
    }

    if side not in ("BUY", "SELL") or mark_price is None:
        return unavailable

    if fast_df is None or slow_df is None:
        return unavailable

    fast = _live_entry_timeframe_check(
        side,
        fast_df,
        mark_price,
        config.LIVE_ENTRY_FAST_TIMEFRAME,
    )
    slow = _live_entry_timeframe_check(
        side,
        slow_df,
        mark_price,
        config.LIVE_ENTRY_SLOW_TIMEFRAME,
    )

    if fast.get("latest_close") is None or slow.get("latest_close") is None:
        return {
            **unavailable,
            "fast": fast,
            "slow": slow,
        }

    ema_tolerance = max(
        get_config_float(
            "EARLY_FLOW_EXIT_EMA_TOLERANCE_PCT",
            get_config_float("LIVE_ENTRY_EMA_TOLERANCE_PCT", 0.08),
        ),
        0,
    ) / 100
    fast_ema20 = _safe_float(fast.get("ema20"))
    slow_ema20 = _safe_float(slow.get("ema20"))
    fast_ema_wrong = bool(fast.get("ema_wrong_side"))
    slow_ema_wrong = bool(slow.get("ema_wrong_side"))

    if fast_ema20 > 0:
        fast_ema_wrong = (
            mark_price < fast_ema20 * (1 - ema_tolerance)
            if side == "BUY"
            else mark_price > fast_ema20 * (1 + ema_tolerance)
        )

    if slow_ema20 > 0:
        slow_ema_wrong = (
            mark_price < slow_ema20 * (1 - ema_tolerance)
            if side == "BUY"
            else mark_price > slow_ema20 * (1 + ema_tolerance)
        )

    fast_adverse = bool(
        fast.get("opposes_direction") or
        fast.get("opposite_reversal") or
        (fast.get("structure_break") and fast_ema_wrong)
    )
    slow_adverse = bool(
        slow.get("opposes_direction") or
        slow.get("opposite_reversal") or
        (slow.get("structure_break") and slow_ema_wrong)
    )
    fast_failure = bool(fast.get("structure_break") and fast_adverse)
    slow_failure = bool(slow.get("structure_break") and slow_adverse)
    dual_opposition = bool(
        fast.get("opposes_direction") and slow.get("opposes_direction")
    )

    reference = _safe_float(reference_price)
    reference_buffer = max(
        get_config_float("EARLY_FLOW_EXIT_REFERENCE_BUFFER_PCT", 0.15),
        0,
    ) / 100
    reference_broken = False

    if reference > 0:
        if side == "BUY":
            reference_broken = mark_price < reference * (1 - reference_buffer)
        else:
            reference_broken = mark_price > reference * (1 + reference_buffer)

    if route == "REVERSAL":
        should_exit = bool(
            slow_failure and
            (fast_adverse or reference_broken)
        )
        reason = "REVERSAL_THESIS_INVALIDATED"
    else:
        should_exit = bool(
            (fast_failure and slow_failure) or
            (reference_broken and slow_failure and fast_adverse)
        )
        reason = "TREND_THESIS_INVALIDATED"

    if not should_exit:
        reason = "EARLY_INVALIDATION_EVIDENCE_INCOMPLETE"

    return {
        "should_exit": should_exit,
        "reason": reason,
        "route": route,
        "mark_price": round(float(mark_price), 8),
        "reference_price": round(reference, 8) if reference > 0 else None,
        "reference_broken": reference_broken,
        "fast_failure": fast_failure,
        "slow_failure": slow_failure,
        "fast_adverse": fast_adverse,
        "slow_adverse": slow_adverse,
        "fast_ema_wrong_side": fast_ema_wrong,
        "slow_ema_wrong_side": slow_ema_wrong,
        "dual_opposition": dual_opposition,
        "fast": fast,
        "slow": slow,
    }


def detect_liquidity_sweep(side, df, label):
    if not config.SMC_ENABLED or not config.SMC_SWEEP_ENABLED:
        return None

    lookback = get_config_int("SMC_SWEEP_LOOKBACK", 24)
    max_age = get_config_int("SMC_SWEEP_MAX_AGE", 5)
    data = _closed_data(df)

    if len(data) < lookback + 2:
        return None

    start = max(lookback, len(data) - max_age)
    best = None

    for pos in range(start, len(data)):
        prior = data.iloc[pos - lookback:pos]
        candle = data.iloc[pos]
        atr = _candle_atr(candle)

        if side == "BUY":
            swept_level = prior["low"].min()
            swept = candle["low"] < swept_level and candle["close"] > swept_level
            direction_ok = _is_bullish(candle)
            depth = (swept_level - candle["low"]) / atr
        else:
            swept_level = prior["high"].max()
            swept = candle["high"] > swept_level and candle["close"] < swept_level
            direction_ok = _is_bearish(candle)
            depth = (candle["high"] - swept_level) / atr

        if not swept or not direction_ok:
            continue

        recency = 1 - ((len(data) - 1 - pos) / max(max_age, 1))
        volume_bonus = 0.25 if candle.get("volume", 0) > candle.get("volume_sma", 0) else 0
        score = round(1 + max(depth, 0) + recency + volume_bonus, 2)
        item = {
            "type": "liquidity_sweep",
            "source": label,
            "level": float(swept_level),
            "score": score,
            "age": len(data) - 1 - pos,
        }

        if not best or item["score"] > best["score"]:
            best = item

    return best


def _collect_order_blocks(df, side, label, timeframe_weight):
    if not config.SMC_ENABLED or not config.SMC_OB_ENABLED:
        return []

    lookback = get_config_int("SMC_OB_LOOKBACK", 120)
    min_displacement = get_config_float("SMC_OB_DISPLACEMENT_ATR", 0.8)
    max_zone_pct = get_config_float("SMC_OB_MAX_ZONE_PCT", 4.0)
    data = _closed_data(df, lookback)
    blocks = []

    if len(data) < 10:
        return blocks

    open_values = np.asarray(data["open"], dtype=float)
    high_values = np.asarray(data["high"], dtype=float)
    low_values = np.asarray(data["low"], dtype=float)
    close_values = np.asarray(data["close"], dtype=float)
    atr_values = np.asarray(data["atr"], dtype=float)
    volume_values = np.asarray(data["volume"], dtype=float)
    volume_sma_values = np.asarray(data["volume_sma"], dtype=float)

    for pos in range(2, len(data) - 3):
        atr = max(float(atr_values[pos]), 1e-10)
        zone_low = float(low_values[pos])
        zone_high = float(high_values[pos])
        zone_width_pct = ((zone_high - zone_low) / max(zone_high, 1e-10)) * 100

        if zone_width_pct > max_zone_pct:
            continue

        if side == "BUY":
            if not close_values[pos] < open_values[pos]:
                continue

            next_close = np.nanmax(close_values[pos + 1:pos + 4])
            next_high = np.nanmax(high_values[pos + 1:pos + 4])
            displacement = (next_close - zone_high) / atr
            broke_structure = next_high > zone_high

            if displacement < min_displacement or not broke_structure:
                continue

            anchor = zone_high
        else:
            if not close_values[pos] > open_values[pos]:
                continue

            next_close = np.nanmin(close_values[pos + 1:pos + 4])
            next_low = np.nanmin(low_values[pos + 1:pos + 4])
            displacement = (zone_low - next_close) / atr
            broke_structure = next_low < zone_low

            if displacement < min_displacement or not broke_structure:
                continue

            anchor = zone_low

        recency_score = pos / len(data)
        volume_bonus = (
            0.25
            if volume_values[pos] > volume_sma_values[pos]
            else 0
        )
        score = timeframe_weight + min(max(displacement, 0), 2.5) + recency_score + volume_bonus
        blocks.append({
            "type": "order_block",
            "source": f"{label}_ob",
            "zone_low": zone_low,
            "zone_high": zone_high,
            "level": float(anchor),
            "score": round(score, 2),
            "displacement": round(float(displacement), 2),
        })

    return blocks


def find_order_block_confirmation(side, entry_price, trend_df, confirm_df, leverage=None):
    zone_min, zone_max = _adverse_zone(side, entry_price, leverage)
    candidates = []
    candidates.extend(_collect_order_blocks(trend_df, side, "1d", 2.0))
    candidates.extend(_collect_order_blocks(confirm_df, side, "4h", 1.25))
    valid = []

    for candidate in candidates:
        if side == "BUY":
            if candidate["level"] >= entry_price:
                continue

            if not (zone_min <= candidate["level"] <= zone_max):
                continue
        else:
            if candidate["level"] <= entry_price:
                continue

            if not (zone_min <= candidate["level"] <= zone_max):
                continue

        distance_pct = pct_distance(entry_price, candidate["level"])
        item = candidate.copy()
        item["distance_pct"] = round(distance_pct, 2)
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["score"], -item["distance_pct"]), reverse=True)
    return valid[0]


def _collect_fvgs(df, label):
    if not config.SMC_ENABLED or not config.SMC_FVG_ENABLED:
        return []

    lookback = get_config_int("SMC_FVG_LOOKBACK", 120)
    min_gap_atr = get_config_float("SMC_FVG_MIN_GAP_ATR", 0.12)
    data = _closed_data(df, lookback)
    fvgs = []

    if len(data) < 5:
        return fvgs

    high_values = np.asarray(data["high"], dtype=float)
    low_values = np.asarray(data["low"], dtype=float)
    atr_values = np.asarray(data["atr"], dtype=float)

    for pos in range(2, len(data)):
        left_high = high_values[pos - 2]
        left_low = low_values[pos - 2]
        right_high = high_values[pos]
        right_low = low_values[pos]
        atr = max(float(atr_values[pos]), 1e-10)

        if left_high < right_low:
            gap_low = float(left_high)
            gap_high = float(right_low)
            gap_atr = (gap_high - gap_low) / atr
            after_low = low_values[pos + 1:]

            if gap_atr >= min_gap_atr and not (
                after_low.size and np.nanmin(after_low) <= gap_low
            ):
                fvgs.append({
                    "type": "bullish_fvg",
                    "source": f"{label}_bullish_fvg",
                    "zone_low": gap_low,
                    "zone_high": gap_high,
                    "level": (gap_low + gap_high) / 2,
                    "score": round(1 + min(gap_atr, 2), 2),
                })

        if left_low > right_high:
            gap_low = float(right_high)
            gap_high = float(left_low)
            gap_atr = (gap_high - gap_low) / atr
            after_high = high_values[pos + 1:]

            if gap_atr >= min_gap_atr and not (
                after_high.size and np.nanmax(after_high) >= gap_high
            ):
                fvgs.append({
                    "type": "bearish_fvg",
                    "source": f"{label}_bearish_fvg",
                    "zone_low": gap_low,
                    "zone_high": gap_high,
                    "level": (gap_low + gap_high) / 2,
                    "score": round(1 + min(gap_atr, 2), 2),
                })

    return fvgs


def _estimated_tp_price(side, entry_price, trend_df, confirm_df):
    if config.STATIC_TP_ENABLED:
        roi = config.STATIC_TP_ROI
        if side == "BUY":
            return entry_price * (1 + (roi / config.LEVERAGE) / 100)

        return entry_price * (1 - (roi / config.LEVERAGE) / 100)

    target = find_structure_take_profit(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )

    if target:
        return target["target_price"]

    roi = config.STRUCTURE_TP_FALLBACK_ROI

    if side == "BUY":
        return entry_price * (1 + (roi / config.LEVERAGE) / 100)

    return entry_price * (1 - (roi / config.LEVERAGE) / 100)


def find_fvg_confirmation(side, entry_price, trend_df, confirm_df, leverage=None):
    zone_min, zone_max = _adverse_zone(side, entry_price, leverage)
    target_price = _estimated_tp_price(side, entry_price, trend_df, confirm_df)
    fvgs = []
    fvgs.extend(_collect_fvgs(trend_df, "1d"))
    fvgs.extend(_collect_fvgs(confirm_df, "4h"))
    supportive = []
    blocking = []

    for fvg in fvgs:
        if side == "BUY":
            if (
                fvg["type"] == "bullish_fvg"
                and fvg["zone_high"] < entry_price
                and zone_min <= fvg["level"] <= zone_max
            ):
                supportive.append(fvg)

            if (
                fvg["type"] == "bearish_fvg"
                and entry_price < fvg["zone_low"] < target_price
            ):
                blocking.append(fvg)
        else:
            if (
                fvg["type"] == "bearish_fvg"
                and fvg["zone_low"] > entry_price
                and zone_min <= fvg["level"] <= zone_max
            ):
                supportive.append(fvg)

            if (
                fvg["type"] == "bullish_fvg"
                and target_price < fvg["zone_high"] < entry_price
            ):
                blocking.append(fvg)

    supportive.sort(key=lambda item: item["score"], reverse=True)
    blocking.sort(key=lambda item: item["score"], reverse=True)

    return {
        "supportive": supportive[0] if supportive else None,
        "blocking": blocking[0] if blocking else None,
        "target_price": target_price,
    }


def _smc_context_score(side, trend_df, confirm_df, entry_df):
    if not config.SMC_ENABLED:
        return 0, {"enabled": False}

    entry_price = latest_closed(entry_df)["close"]
    score = 0
    context = {
        "enabled": True,
        "liquidity_sweep": None,
        "order_block": None,
        "fvg_support": None,
        "fvg_block": None,
    }

    entry_sweep = detect_liquidity_sweep(side, entry_df, "1h")
    confirm_sweep = detect_liquidity_sweep(side, confirm_df, "4h")
    opposite_side = "SELL" if side == "BUY" else "BUY"
    opposite_sweep = detect_liquidity_sweep(opposite_side, entry_df, "1h")

    if entry_sweep:
        context["liquidity_sweep"] = entry_sweep
        score += min(entry_sweep["score"], 2.0)
    elif confirm_sweep:
        context["liquidity_sweep"] = confirm_sweep
        score += min(confirm_sweep["score"], 1.25)

    if opposite_sweep:
        score -= min(opposite_sweep["score"], 1.25)

    order_block = find_order_block_confirmation(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )

    if order_block:
        context["order_block"] = order_block
        score += min(order_block["score"] / 2, 2.0)

    fvg = find_fvg_confirmation(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE
    )
    context["fvg_support"] = fvg["supportive"]
    context["fvg_block"] = fvg["blocking"]

    if fvg["supportive"]:
        score += min(fvg["supportive"]["score"] / 2, 1.0)

    if fvg["blocking"]:
        score -= get_config_float("SMC_TP_PATH_BLOCK_PENALTY", 1.5)

    return round(score, 2), context


def find_adverse_zone_level(side, entry_price, trend_df, confirm_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    max_adverse_roi = abs(
        get_config_float(
            "ADVERSE_REVERSAL_MAX_ROI",
            get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50)
        )
    )
    max_adverse_roi = max(max_adverse_roi, 0.01)
    max_price_move = (max_adverse_roi / max(leverage_to_use, 1)) / 100
    use_1d_only = bool(getattr(config, "ADVERSE_REVERSAL_USE_1D_ONLY", True))
    include_range = bool(getattr(config, "ADVERSE_REVERSAL_INCLUDE_RANGE", True))
    include_ema = bool(getattr(config, "ADVERSE_REVERSAL_INCLUDE_EMA", True))
    trend_label = str(
        getattr(
            config,
            "ADVERSE_REVERSAL_TIMEFRAME",
            getattr(config, "TREND_TIMEFRAME", "1d")
        )
    )

    if side == "BUY":
        zone_min = entry_price * (1 - max_price_move)
        zone_max = entry_price
    else:
        zone_min = entry_price
        zone_max = entry_price * (1 + max_price_move)

    candidates = []
    candidates.extend(_collect_pivot_levels(trend_df, side, trend_label, 2.0))

    if include_range:
        candidates.extend(_collect_range_levels(trend_df, side, trend_label, 2.0))

    if include_ema:
        candidates.extend(_collect_ema_levels(trend_df, side, trend_label, 2.0))

    if not use_1d_only and confirm_df is not None:
        confirm_label = str(getattr(config, "CONFIRMATION_TIMEFRAME", "4h"))
        candidates.extend(_collect_pivot_levels(confirm_df, side, confirm_label, 1.25))

        if include_range:
            candidates.extend(_collect_range_levels(confirm_df, side, confirm_label, 1.25))

        if include_ema:
            candidates.extend(_collect_ema_levels(confirm_df, side, confirm_label, 1.25))

    tolerance = _level_tolerance(trend_df)

    if not use_1d_only and confirm_df is not None:
        tolerance = max(tolerance, _level_tolerance(confirm_df))

    candidates = _dedupe_levels(candidates, tolerance)
    valid = []

    for candidate in candidates:
        level = candidate["level"]

        if not (zone_min <= level <= zone_max):
            continue

        if side == "BUY" and level >= entry_price:
            continue

        if side == "SELL" and level <= entry_price:
            continue

        if side == "BUY":
            adverse_roi = ((level - entry_price) / entry_price) * leverage_to_use * 100
        else:
            adverse_roi = ((entry_price - level) / entry_price) * leverage_to_use * 100

        proximity_score = 1 - min(abs(adverse_roi) / max_adverse_roi, 1)
        item = candidate.copy()
        item["adverse_roi"] = round(adverse_roi, 2)
        item["score"] = round(candidate["score"] + proximity_score, 2)
        item["zone_min"] = zone_min
        item["zone_max"] = zone_max
        item["max_adverse_roi"] = round(max_adverse_roi, 2)
        item["safety_timeframe"] = trend_label if use_1d_only else "multi_tf"
        valid.append(item)

    if not valid:
        return None

    valid.sort(key=lambda item: (item["score"], -abs(item["adverse_roi"])), reverse=True)
    best = valid[0]

    min_score = get_config_float(
        "ADVERSE_REVERSAL_MIN_SCORE",
        get_config_float("LONG_TERM_SR_MIN_SCORE", 2.5)
    )

    if best["score"] < min_score:
        return None

    return best


def validate_adverse_zone_level(side, entry_price, trend_df, confirm_df, leverage=None):
    enabled = bool(
        getattr(
            config,
            "ADVERSE_REVERSAL_LEVEL_CHECK_ENABLED",
            getattr(config, "LONG_TERM_ADVERSE_ZONE_CHECK_ENABLED", True)
        )
    )

    if not enabled:
        label = "support" if side == "BUY" else "resistance"
        return True, {
            "reason": f"{label.upper()} ADVERSE-ZONE CHECK DISABLED",
            "level": float(entry_price),
            "adverse_roi": 0,
            "source": "disabled",
            "score": 0,
            "max_adverse_roi": get_config_float(
                "ADVERSE_REVERSAL_MAX_ROI",
                get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50)
            ),
            "safety_timeframe": str(
                getattr(
                    config,
                    "ADVERSE_REVERSAL_TIMEFRAME",
                    getattr(config, "TREND_TIMEFRAME", "1d")
                )
            ),
            "level_check_disabled": True,
        }

    level = find_adverse_zone_level(
        side,
        entry_price,
        trend_df,
        confirm_df,
        leverage=leverage
    )

    if level:
        return True, level

    zone_roi = get_config_float(
        "ADVERSE_REVERSAL_MAX_ROI",
        get_config_float("LONG_TERM_MAX_ADVERSE_ROI", 50)
    )
    label = "support" if side == "BUY" else "resistance"
    timeframe = str(
        getattr(
            config,
            "ADVERSE_REVERSAL_TIMEFRAME",
            getattr(config, "TREND_TIMEFRAME", "1d")
        )
    ).upper()
    return False, {
        "reason": (
            f"NO STRONG {timeframe} {label.upper()} "
            f"WITHIN -{zone_roi:.0f}% ROI SAFETY ZONE"
        )
    }


def _dca_structure_tolerance(current_price, entry_df, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    atr_tolerance = _average_range(entry_df, 14) * get_config_float(
        "DCA_STRUCTURE_MAX_DISTANCE_ATR",
        0.6
    )
    roi_tolerance = current_price * (
        get_config_float("DCA_STRUCTURE_MAX_DISTANCE_ROI", 6) /
        max(leverage_to_use, 1) /
        100
    )

    return max(atr_tolerance, roi_tolerance, current_price * 0.001)


def _normalise_dca_level(side, current_price, candidate, tolerance, leverage=None):
    leverage_to_use = leverage or config.LEVERAGE
    level = float(candidate.get("level", 0) or 0)

    if level <= 0:
        return None

    zone_low = float(candidate.get("zone_low", level) or level)
    zone_high = float(candidate.get("zone_high", level) or level)

    if zone_low > zone_high:
        zone_low, zone_high = zone_high, zone_low

    if side == "BUY":
        if current_price < zone_low - tolerance:
            return None

        distance = max(0, current_price - zone_high)
    else:
        if current_price > zone_high + tolerance:
            return None

        distance = max(0, zone_low - current_price)

    if distance > tolerance:
        return None

    proximity_score = 1 - min(distance / max(tolerance, 1e-10), 1)
    distance_roi = (distance / current_price) * leverage_to_use * 100
    item = candidate.copy()
    item["level"] = level
    item["zone_low"] = zone_low
    item["zone_high"] = zone_high
    item["distance"] = round(float(distance), 8)
    item["distance_roi"] = round(float(distance_roi), 2)
    item["score"] = round(float(candidate.get("score", 0)) + proximity_score, 2)
    item["max_distance"] = round(float(tolerance), 8)

    return item


def _collect_dca_structure_candidates(side, trend_df, confirm_df):
    candidates = []

    for item in _collect_pivot_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "pivot"
        candidates.append(candidate)

    for item in _collect_pivot_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "pivot"
        candidates.append(candidate)

    for item in _collect_range_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "range"
        candidates.append(candidate)

    for item in _collect_range_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "range"
        candidates.append(candidate)

    for item in _collect_ema_levels(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "ema"
        candidates.append(candidate)

    for item in _collect_ema_levels(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "ema"
        candidates.append(candidate)

    for item in _collect_order_blocks(trend_df, side, "1d", 2.0):
        candidate = item.copy()
        candidate["kind"] = "order_block"
        candidate["score"] = float(candidate.get("score", 0)) + 0.5
        candidates.append(candidate)

    for item in _collect_order_blocks(confirm_df, side, "4h", 1.25):
        candidate = item.copy()
        candidate["kind"] = "order_block"
        candidate["score"] = float(candidate.get("score", 0)) + 0.5
        candidates.append(candidate)

    for fvg in _collect_fvgs(trend_df, "1d"):
        if side == "BUY" and fvg["type"] != "bullish_fvg":
            continue

        if side == "SELL" and fvg["type"] != "bearish_fvg":
            continue

        candidate = fvg.copy()
        candidate["kind"] = "fvg"
        candidate["score"] = float(candidate.get("score", 0)) + 0.25
        candidates.append(candidate)

    for fvg in _collect_fvgs(confirm_df, "4h"):
        if side == "BUY" and fvg["type"] != "bullish_fvg":
            continue

        if side == "SELL" and fvg["type"] != "bearish_fvg":
            continue

        candidate = fvg.copy()
        candidate["kind"] = "fvg"
        candidate["score"] = float(candidate.get("score", 0)) + 0.25
        candidates.append(candidate)

    return candidates


def _level_touched_by_candle(side, candle, level, tolerance):
    zone_low = float(level.get("zone_low", level.get("level")))
    zone_high = float(level.get("zone_high", level.get("level")))

    if side == "BUY":
        return float(candle["low"]) <= zone_high + tolerance

    return float(candle["high"]) >= zone_low - tolerance


def _dca_reaction_confirmation(side, level, entry_df, tolerance):
    if not getattr(config, "DCA_STRUCTURE_REQUIRE_REACTION", True):
        return True, {
            "reaction": "REACTION_NOT_REQUIRED"
        }

    lookback = get_config_int("DCA_STRUCTURE_REACTION_LOOKBACK", 3)
    min_body_atr = get_config_float("DCA_STRUCTURE_REACTION_MIN_BODY_ATR", 0.05)
    close_position_threshold = get_config_float(
        "DCA_STRUCTURE_REACTION_CLOSE_POSITION",
        0.55
    )
    data = _closed_data(entry_df, lookback)

    if len(data) == 0:
        return False, {
            "reason": "DCA_STRUCTURE_REACTION_DATA_UNAVAILABLE"
        }

    best = None

    for _, candle in data.iterrows():
        if not _level_touched_by_candle(side, candle, level, tolerance):
            continue

        atr = _candle_atr(candle)
        body_atr = _body(candle) / atr
        close_position = _close_position(candle)
        close = float(candle["close"])
        zone_low = float(level.get("zone_low", level.get("level")))
        zone_high = float(level.get("zone_high", level.get("level")))

        if side == "BUY":
            reclaimed_level = close >= zone_low
            reaction_ok = (
                reclaimed_level
                and
                close_position >= close_position_threshold
                and (
                    _is_bullish(candle)
                    or body_atr >= min_body_atr
                )
            )
        else:
            reclaimed_level = close <= zone_high
            reaction_ok = (
                reclaimed_level
                and
                close_position <= 1 - close_position_threshold
                and (
                    _is_bearish(candle)
                    or body_atr >= min_body_atr
                )
            )

        item = {
            "close_position": round(float(close_position), 2),
            "body_atr": round(float(body_atr), 2),
            "candle_close": close,
            "reclaimed_level": reclaimed_level,
            "reaction": "OK" if reaction_ok else "WEAK",
        }

        if reaction_ok:
            return True, item

        best = item

    if best:
        best["reason"] = "DCA_STRUCTURE_REACTION_WEAK"
        return False, best

    return False, {
        "reason": "DCA_STRUCTURE_LEVEL_NOT_TOUCHED"
    }


def validate_dca_structure_level(side, current_price, trend_df, confirm_df, entry_df, leverage=None):
    if not getattr(config, "DCA_STRUCTURE_LEVEL_ENABLED", True):
        return True, {
            "reason": "DCA_STRUCTURE_LEVEL_CHECK_DISABLED",
            "level": current_price,
            "source": "disabled",
            "score": 0,
        }

    if current_price <= 0:
        return False, {
            "reason": "DCA_STRUCTURE_INVALID_PRICE"
        }

    min_score = get_config_float("DCA_STRUCTURE_MIN_SCORE", 2.0)
    tolerance = _dca_structure_tolerance(
        current_price,
        entry_df,
        leverage=leverage
    )
    levels = []

    for candidate in _collect_dca_structure_candidates(side, trend_df, confirm_df):
        level = _normalise_dca_level(
            side,
            current_price,
            candidate,
            tolerance,
            leverage=leverage
        )

        if not level:
            continue

        if level["score"] < min_score:
            continue

        levels.append(level)

    if not levels:
        label = "SUPPORT" if side == "BUY" else "RESISTANCE"
        return False, {
            "reason": f"NO DCA {label} LEVEL NEAR CURRENT PRICE"
        }

    levels.sort(key=lambda item: (item["score"], -item["distance"]), reverse=True)
    best = levels[0]
    reaction_ok, reaction = _dca_reaction_confirmation(
        side,
        best,
        entry_df,
        tolerance
    )

    if not reaction_ok:
        return False, {
            "reason": reaction.get("reason", "DCA_STRUCTURE_REACTION_FAILED"),
            **best,
            **reaction,
        }

    return True, {
        "reason": "DCA_STRUCTURE_LEVEL_OK",
        **best,
        **reaction,
    }


def _candle_quality_score(side, candle, max_ema_distance):
    open_price = _safe_float(candle.get("open"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    atr = max(_safe_float(candle.get("atr")), 1e-10)
    ema20 = _safe_float(candle.get("ema20"))
    volume = _safe_float(candle.get("volume"))
    volume_sma = _safe_float(candle.get("volume_sma"))
    rsi = _safe_float(candle.get("rsi"), 50)
    candle_range = max(high - low, 1e-10)
    body = abs(close - open_price)
    close_position = (close - low) / candle_range
    directional_close = close_position if side == "BUY" else 1 - close_position
    upper_wick = max(high - max(open_price, close), 0)
    lower_wick = max(min(open_price, close) - low, 0)
    rejection_wick = upper_wick if side == "BUY" else lower_wick
    rejection_wick_ratio = rejection_wick / candle_range
    body_ratio = body / candle_range
    directional_body = close - open_price if side == "BUY" else open_price - close
    momentum_atr = directional_body / atr
    candle_atr = candle_range / atr
    volume_mult = volume / volume_sma if volume_sma > 0 else 0
    ema_distance = pct_distance(close, ema20) if ema20 else 0
    chase_atr = abs(close - ema20) / atr if ema20 else 0
    min_body = get_config_float("MIN_SIGNAL_BODY_RATIO", 0.16)
    min_close = get_config_float("MIN_SIGNAL_CLOSE_POSITION", 0.50)
    max_close = get_config_float("MAX_SIGNAL_CLOSE_POSITION", 0.88)
    max_candle_atr = get_config_float("MAX_SIGNAL_CANDLE_ATR", 2.0)
    min_volume_mult = get_config_float("MIN_VOLUME_SMA_MULT", 1.05)
    max_wick_ratio = get_config_float("MAX_SIGNAL_REJECTION_WICK_RATIO", 0.45)
    min_momentum_atr = get_config_float("MIN_SIGNAL_MOMENTUM_ATR", 0.03)
    max_chase_pct = get_config_float("MAX_CHASE_DISTANCE_PCT", max_ema_distance)
    max_late_entry_atr = get_config_float("MAX_LATE_ENTRY_ATR", 2.2)
    late_penalty = get_config_float("LATE_ENTRY_SCORE_PENALTY", 2.0)
    wick_filter = bool(getattr(config, "SIGNAL_WICK_FILTER_ENABLED", True))
    score = 0

    score += 0.5 if body_ratio >= min_body else -0.5
    score += 0.5 if directional_close >= min_close else -0.75
    score += 0.25 if directional_close <= max_close else -0.25

    if wick_filter:
        score += 0.5 if rejection_wick_ratio <= max_wick_ratio else -1.0

    score += 0.25 if candle_atr <= max_candle_atr else -1.0
    score += 0.5 if momentum_atr >= min_momentum_atr else -0.25
    score += 0.5 if volume_mult >= min_volume_mult else -0.25
    score += 0.25 if ema_distance <= max_ema_distance else -0.75

    if max_chase_pct > 0:
        score += 0.25 if ema_distance <= max_chase_pct else -0.5

    if max_late_entry_atr > 0:
        score += 0.25 if chase_atr <= max_late_entry_atr else -late_penalty

    overheat = (
        (side == "BUY" and rsi > get_config_float("BUY_RSI_OVERHEAT", 72)) or
        (side == "SELL" and rsi < get_config_float("SELL_RSI_OVERHEAT", 28))
    )

    if overheat:
        score -= 1.0

    quality_ok = (
        not overheat and
        candle_atr <= max(max_candle_atr * 1.75, max_candle_atr + 1) and
        (
            not wick_filter or
            rejection_wick_ratio <= max(max_wick_ratio * 1.75, 0.80)
        )
    )

    return round(score, 2), {
        "score": round(score, 2),
        "body_ratio": round(float(body_ratio), 3),
        "directional_close": round(float(directional_close), 3),
        "rejection_wick_ratio": round(float(rejection_wick_ratio), 3),
        "volume_mult": round(float(volume_mult), 2),
        "candle_atr": round(float(candle_atr), 2),
        "momentum_atr": round(float(momentum_atr), 3),
        "ema_distance": round(float(ema_distance), 3),
        "chase_atr": round(float(chase_atr), 2),
        "rsi": round(float(rsi), 2),
        "overheat": overheat,
        "quality_ok": quality_ok,
    }


def _market_regime_score(side, trend_df, confirm_df, entry_df):
    trend = latest_closed(trend_df)
    confirm = latest_closed(confirm_df)
    entry = latest_closed(entry_df)
    confirm_structure = detect_market_structure(confirm_df)
    trend_adx = _safe_float(trend.get("adx"))
    confirm_adx = _safe_float(confirm.get("adx"))
    sideways_adx = get_config_float("SIDEWAYS_ADX", 15)
    trending_adx = get_config_float("TRENDING_ADX", 25)
    atr = max(_safe_float(entry.get("atr")), 1e-10)
    entry_close = _safe_float(entry.get("close"))
    entry_ema20 = _safe_float(entry.get("ema20"))
    chase_atr = abs(entry_close - entry_ema20) / atr if entry_ema20 else 0
    max_late_entry_atr = get_config_float("MAX_LATE_ENTRY_ATR", 2.2)

    if side == "BUY":
        trend_aligned = (
            trend["close"] > trend["ema50"] and trend["ema50"] >= trend["ema200"]
        )
        confirm_aligned = (
            confirm["close"] > confirm["ema50"] and confirm["ema20"] >= confirm["ema50"]
        )
        breakout = confirm_structure["bullish_breakout"]
    else:
        trend_aligned = (
            trend["close"] < trend["ema50"] and trend["ema50"] <= trend["ema200"]
        )
        confirm_aligned = (
            confirm["close"] < confirm["ema50"] and confirm["ema20"] <= confirm["ema50"]
        )
        breakout = confirm_structure["bearish_breakdown"]

    if chase_atr > max_late_entry_atr:
        regime = "late_entry"
    elif trend_adx < sideways_adx and confirm_adx < sideways_adx:
        regime = "sideways"
    elif trend_adx >= trending_adx or confirm_adx >= trending_adx:
        regime = "trending"
    elif breakout:
        regime = "breakout"
    else:
        regime = "transition"

    score = 0

    if regime == "late_entry":
        score -= get_config_float("LATE_ENTRY_SCORE_PENALTY", 2.0)
    elif regime == "sideways":
        score += 0.5 if breakout else -1.0
    elif regime == "trending":
        score += 1.0 if trend_aligned and confirm_aligned else -1.25
    elif regime == "breakout":
        score += 0.75 if confirm_aligned else 0
    elif trend_aligned and confirm_aligned:
        score += 0.25

    return round(score, 2), {
        "regime": regime,
        "trend_adx": round(float(trend_adx), 2),
        "confirm_adx": round(float(confirm_adx), 2),
        "trend_aligned": trend_aligned,
        "confirm_aligned": confirm_aligned,
        "breakout": breakout,
        "chase_atr": round(float(chase_atr), 2),
    }


def _ema_gap_score(side, candle):
    enabled = bool(getattr(config, "EMA_GAP_FILTER_ENABLED", True))
    context = {
        "enabled": enabled,
        "score": 0,
    }

    if not enabled:
        context["reason"] = "EMA_GAP_DISABLED"
        return 0, context

    ema20 = _safe_float(candle.get("ema20"))
    ema50 = _safe_float(candle.get("ema50"))
    ema200 = _safe_float(candle.get("ema200"))

    if ema20 <= 0 or ema50 <= 0 or ema200 <= 0:
        context["reason"] = "EMA_GAP_DATA_UNAVAILABLE"
        return 0, context

    gap20_50 = pct_distance(ema20, ema50)
    gap50_200 = pct_distance(ema50, ema200)
    min20_50 = get_config_float("MIN_EMA20_EMA50_GAP_PCT", 0.03)
    min50_200 = get_config_float("MIN_EMA50_EMA200_GAP_PCT", 0.05)
    max20_50 = get_config_float("MAX_EMA20_EMA50_GAP_PCT", 0)
    max50_200 = get_config_float("MAX_EMA50_EMA200_GAP_PCT", 0)
    bonus = get_config_float("EMA_GAP_SCORE_BONUS", 0.75)
    penalty = get_config_float("EMA_GAP_SCORE_PENALTY", 0.5)

    if side == "BUY":
        order_ok = ema20 >= ema50 >= ema200
    else:
        order_ok = ema20 <= ema50 <= ema200

    min_ok = gap20_50 >= min20_50 and gap50_200 >= min50_200
    max20_ok = max20_50 <= 0 or gap20_50 <= max20_50
    max50_ok = max50_200 <= 0 or gap50_200 <= max50_200
    max_ok = max20_ok and max50_ok

    if order_ok and min_ok and max_ok:
        score = bonus
        reason = "EMA_GAP_HEALTHY"
    elif not order_ok:
        score = -penalty
        reason = "EMA_ORDER_NOT_ALIGNED"
    elif not min_ok:
        score = -penalty
        reason = "EMA_GAP_TOO_TIGHT"
    else:
        score = -(penalty / 2)
        reason = "EMA_GAP_TOO_WIDE"

    score = round(float(score), 2)
    context.update({
        "reason": reason,
        "score": score,
        "order_ok": order_ok,
        "min_ok": min_ok,
        "max_ok": max_ok,
        "gap20_50": round(float(gap20_50), 3),
        "gap50_200": round(float(gap50_200), 3),
    })
    return score, context


def _trend_bias_score(side, trend_df):
    trend = latest_closed(trend_df)
    prev = previous_closed(trend_df)
    structure = detect_market_structure(trend_df)
    min_adx = get_config_float("LONG_TERM_MIN_ADX", 14)
    score = 0
    hard_ok = False

    if side == "BUY":
        score = add_score(score, trend["close"] > trend["ema200"], 3)
        score = add_score(score, trend["ema50"] > trend["ema200"], 3)
        score = add_score(score, trend["ema20"] > trend["ema50"], 2)
        score = add_score(score, trend["close"] > trend["ema50"], 2)
        score = add_score(score, trend["ema20"] > prev["ema20"], 1)
        score = add_score(score, structure["bullish_structure"], 2)
        score = add_score(score, structure["bullish_breakout"], 2)
        score = add_score(score, trend["adx"] >= min_adx, 1)
        hard_ok = (
            trend["close"] > trend["ema50"]
            and (trend["ema20"] > trend["ema50"] or trend["ema50"] > trend["ema200"])
        )
    else:
        score = add_score(score, trend["close"] < trend["ema200"], 3)
        score = add_score(score, trend["ema50"] < trend["ema200"], 3)
        score = add_score(score, trend["ema20"] < trend["ema50"], 2)
        score = add_score(score, trend["close"] < trend["ema50"], 2)
        score = add_score(score, trend["ema20"] < prev["ema20"], 1)
        score = add_score(score, structure["bearish_structure"], 2)
        score = add_score(score, structure["bearish_breakdown"], 2)
        score = add_score(score, trend["adx"] >= min_adx, 1)
        hard_ok = (
            trend["close"] < trend["ema50"]
            and (trend["ema20"] < trend["ema50"] or trend["ema50"] < trend["ema200"])
        )

    ema_gap_score, _ = _ema_gap_score(side, trend)
    score += ema_gap_score

    return score, hard_ok


def _confirmation_score(side, confirm_df):
    confirm = latest_closed(confirm_df)
    prev = previous_closed(confirm_df)
    structure = detect_market_structure(confirm_df)
    min_adx = get_config_float("LONG_TERM_MIN_ADX", 14)
    max_ema_distance = get_config_float("MAX_SIGNAL_EMA20_DISTANCE_PCT", 1.2)
    score = 0
    hard_ok = False

    if side == "BUY":
        score = add_score(score, confirm["close"] > confirm["ema50"], 3)
        score = add_score(score, confirm["ema20"] > confirm["ema50"], 2)
        score = add_score(score, confirm["macd"] > confirm["macd_signal"], 2)
        score = add_score(score, 45 <= confirm["rsi"] <= 72, 2)
        score = add_score(score, confirm["adx"] >= min_adx, 1)
        score = add_score(score, structure["bullish_breakout"], 2)
        score = add_score(score, confirm["close"] > prev["high"], 1)
        score = add_score(score, confirm["volume"] > confirm["volume_sma"], 1)
        hard_ok = (
            confirm["close"] > confirm["ema20"]
            and (confirm["macd"] > confirm["macd_signal"] or confirm["rsi"] > 50)
        )
    else:
        score = add_score(score, confirm["close"] < confirm["ema50"], 3)
        score = add_score(score, confirm["ema20"] < confirm["ema50"], 2)
        score = add_score(score, confirm["macd"] < confirm["macd_signal"], 2)
        score = add_score(score, 28 <= confirm["rsi"] <= 55, 2)
        score = add_score(score, confirm["adx"] >= min_adx, 1)
        score = add_score(score, structure["bearish_breakdown"], 2)
        score = add_score(score, confirm["close"] < prev["low"], 1)
        score = add_score(score, confirm["volume"] > confirm["volume_sma"], 1)
        hard_ok = (
            confirm["close"] < confirm["ema20"]
            and (confirm["macd"] < confirm["macd_signal"] or confirm["rsi"] < 50)
        )

    quality_score, quality = _candle_quality_score(
        side,
        confirm,
        max_ema_distance
    )
    ema_gap_score, ema_gap = _ema_gap_score(side, confirm)
    score += quality_score
    score += ema_gap_score
    quality["ema_gap"] = ema_gap
    quality["ema_gap_score"] = ema_gap_score
    quality["score"] = round(float(quality.get("score", 0)) + ema_gap_score, 2)
    quality["direction_ok"] = bool(hard_ok)
    hard_ok = hard_ok and quality["quality_ok"]

    return round(score, 2), hard_ok, quality


def _entry_score(side, entry_df):
    entry = latest_closed(entry_df)
    prev = previous_closed(entry_df)
    ema_distance = pct_distance(entry["close"], entry["ema20"])
    atr = max(_safe_float(entry.get("atr")), 1e-10)
    ema20 = _safe_float(entry.get("ema20"))
    chase_atr = abs(_safe_float(entry.get("close")) - ema20) / atr if ema20 else 0
    late_block_enabled = bool(getattr(config, "LATE_ENTRY_HARD_BLOCK_ENABLED", True))
    hard_late_limit = get_config_float("MAX_LATE_ENTRY_HARD_BLOCK_ATR", 2.6)
    late_entry_ok = (
        not late_block_enabled or
        hard_late_limit <= 0 or
        chase_atr <= hard_late_limit
    )
    max_ema_distance = get_config_float(
        "MAX_ENTRY_EMA20_DISTANCE_PCT",
        get_config_float("LONG_TERM_ENTRY_MAX_EMA_DISTANCE_PCT", 6)
    )
    score = 0
    hard_ok = False

    if side == "BUY":
        bullish_candle = entry["close"] > entry["open"]
        score = add_score(score, entry["close"] > entry["ema20"], 2)
        score = add_score(score, entry["macd"] > entry["macd_signal"], 1)
        score = add_score(score, entry["rsi"] > 50, 1)
        score = add_score(score, bullish_candle, 1)
        score = add_score(score, entry["close"] > prev["high"], 1)
        score = add_score(score, ema_distance <= max_ema_distance, 1)
        score = add_score(score, entry["volume"] > entry["volume_sma"], 1)
        hard_ok = entry["close"] > entry["ema20"] and ema_distance <= max_ema_distance
    else:
        bearish_candle = entry["close"] < entry["open"]
        score = add_score(score, entry["close"] < entry["ema20"], 2)
        score = add_score(score, entry["macd"] < entry["macd_signal"], 1)
        score = add_score(score, entry["rsi"] < 50, 1)
        score = add_score(score, bearish_candle, 1)
        score = add_score(score, entry["close"] < prev["low"], 1)
        score = add_score(score, ema_distance <= max_ema_distance, 1)
        score = add_score(score, entry["volume"] > entry["volume_sma"], 1)
        hard_ok = entry["close"] < entry["ema20"] and ema_distance <= max_ema_distance

    quality_score, quality = _candle_quality_score(
        side,
        entry,
        max_ema_distance
    )
    score += quality_score
    quality["direction_ok"] = bool(hard_ok)
    hard_ok = hard_ok and quality["quality_ok"] and late_entry_ok
    quality["late_entry_ok"] = late_entry_ok
    quality["hard_late_limit_atr"] = round(float(hard_late_limit), 2)

    return round(score, 2), hard_ok, ema_distance, quality


def _btc_context_score(side, btc_trend, btc_corr, rs):
    score = 0
    corr_threshold = get_config_float("LONG_TERM_BTC_CORR_THRESHOLD", 0.65)

    if btc_corr is not None and btc_corr >= corr_threshold:
        if side == "BUY":
            score = add_score(score, btc_trend == "BULLISH", 2)
            score -= 2 if btc_trend == "BEARISH" else 0
        else:
            score = add_score(score, btc_trend == "BEARISH", 2)
            score -= 2 if btc_trend == "BULLISH" else 0

    if rs is not None:
        if side == "BUY":
            score = add_score(score, rs > 1, 1)
            score -= 1 if rs < -2 else 0
        else:
            score = add_score(score, rs < -1, 1)
            score -= 1 if rs > 2 else 0

    return score


def _closed_frame_return_pct(df, periods):
    try:
        closes = df["close"].iloc[:-1]
        periods = max(int(periods), 1)

        if len(closes) < periods + 1:
            return None

        start = float(closes.iloc[-(periods + 1)])
        end = float(closes.iloc[-1])

        if start <= 0:
            return None

        return round(((end - start) / start) * 100, 3)

    except Exception:
        return None


def _oi_price_state(oi_change, price_change, oi_min, price_min):
    if oi_change is None or price_change is None:
        return "UNAVAILABLE"

    if oi_change >= oi_min:
        if price_change >= price_min:
            return "LONG_BUILD"
        if price_change <= -price_min:
            return "SHORT_BUILD"
        return "OI_BUILD_FLAT_PRICE"

    if oi_change <= -oi_min:
        if price_change >= price_min:
            return "SHORT_COVERING"
        if price_change <= -price_min:
            return "LONG_LIQUIDATION"
        return "OI_UNWIND_FLAT_PRICE"

    return "OI_NEUTRAL"


def _futures_participation_score(side, participation, price_change_pct=None):
    if not participation or not participation.get("available"):
        return 0

    score = 0
    oi_change = participation.get("oi_change_pct")
    taker_ratio = participation.get("taker_buy_sell_ratio")
    global_ratio = participation.get("global_long_short_ratio")
    top_ratio = participation.get("top_long_short_ratio")
    funding_rate = participation.get("funding_rate")
    oi_min = get_config_float("FUTURES_CONTEXT_OI_MIN_CHANGE_PCT", 1.0)
    price_min = get_config_float("FUTURES_CONTEXT_PRICE_MIN_CHANGE_PCT", 0.5)
    taker_buy_min = get_config_float("FUTURES_CONTEXT_TAKER_BUY_MIN", 1.05)
    taker_sell_max = get_config_float("FUTURES_CONTEXT_TAKER_SELL_MAX", 0.95)
    crowd_long_max = get_config_float("FUTURES_CONTEXT_CROWD_LONG_MAX", 2.2)
    crowd_short_min = get_config_float("FUTURES_CONTEXT_CROWD_SHORT_MIN", 0.45)
    funding_abs_max = get_config_float("FUTURES_CONTEXT_FUNDING_ABS_MAX", 0.001)

    oi_state = _oi_price_state(
        _safe_float(oi_change, None),
        _safe_float(price_change_pct, None),
        oi_min,
        price_min,
    )
    participation["price_change_pct"] = price_change_pct
    participation["oi_price_state"] = oi_state

    if oi_state == "LONG_BUILD":
        score += 1.5 if side == "BUY" else -1
    elif oi_state == "SHORT_BUILD":
        score += 1.5 if side == "SELL" else -1
    elif oi_state == "SHORT_COVERING":
        score += 0.5 if side == "BUY" else -0.25
    elif oi_state == "LONG_LIQUIDATION":
        score += 0.5 if side == "SELL" else -0.25

    if taker_ratio is not None:
        if side == "BUY":
            score = add_score(score, taker_ratio >= taker_buy_min, 2)
            score -= 2 if taker_ratio <= taker_sell_max else 0
        else:
            score = add_score(score, taker_ratio <= taker_sell_max, 2)
            score -= 2 if taker_ratio >= taker_buy_min else 0

    crowd_ratio = top_ratio if top_ratio is not None else global_ratio

    if crowd_ratio is not None:
        if side == "BUY":
            score -= 1.5 if crowd_ratio >= crowd_long_max else 0
            score = add_score(score, crowd_ratio <= 1, 0.5)
        else:
            score -= 1.5 if crowd_ratio <= crowd_short_min else 0
            score = add_score(score, crowd_ratio >= 1, 0.5)

    if funding_rate is not None:
        if side == "BUY":
            score -= 1 if funding_rate >= funding_abs_max else 0
            score = add_score(score, funding_rate <= -funding_abs_max, 0.5)
        else:
            score -= 1 if funding_rate <= -funding_abs_max else 0
            score = add_score(score, funding_rate >= funding_abs_max, 0.5)

    weight = get_config_float("FUTURES_CONTEXT_SCORE_WEIGHT", 1.15)
    return round(score * weight, 2)


def _futures_context_gate(participation_score, participation):
    if not getattr(config, "FUTURES_CONTEXT_BLOCK_CONFLICT_ENABLED", True):
        return True, []

    if not participation or not participation.get("available"):
        return True, []

    minimum = get_config_float("FUTURES_CONTEXT_MIN_SIGNAL_SCORE", -0.5)
    score = _safe_float(participation_score)

    if score < minimum:
        return False, [
            f"FUTURES_CONTEXT_CONFLICT={round(score, 2)} < {minimum}"
        ]

    return True, []


def _module_gates_check(
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    regime_score
):
    if not getattr(config, "SIGNAL_MODULE_GATES_ENABLED", True):
        return True, []

    checks = (
        ("TREND", trend_score, get_config_float("SIGNAL_MIN_TREND_SCORE", 7)),
        ("CONFIRM", confirm_score, get_config_float("SIGNAL_MIN_CONFIRM_SCORE", 7)),
        ("ENTRY", entry_score, get_config_float("SIGNAL_MIN_ENTRY_SCORE", 4)),
        ("QUALITY", quality_score, get_config_float("SIGNAL_MIN_QUALITY_SCORE", 0)),
        ("REGIME", regime_score, get_config_float("SIGNAL_MIN_REGIME_SCORE", -1.5)),
    )
    failures = []

    for label, value, minimum in checks:
        value = _safe_float(value)
        minimum = _safe_float(minimum)

        if value < minimum:
            failures.append(f"{label}={round(value, 2)} < {minimum}")

    return not failures, failures


def _trend_timing_rescue_context(
    trend_ok,
    confirm_ok,
    entry_ok,
    level_ok,
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    regime_score,
    trend_confidence,
    confirm_quality,
    entry_quality,
    participation_score,
    participation,
    futures_ok
):
    enabled = bool(getattr(config, "TREND_TIMING_RESCUE_ENABLED", True))
    context = {
        "enabled": enabled,
        "eligible": False,
        "active": False,
        "missed_module": None,
        "reasons": [],
    }

    if not enabled:
        context["reasons"].append("TREND_TIMING_RESCUE_DISABLED")
        return context

    tolerance = max(
        get_config_float("TREND_TIMING_RESCUE_SCORE_TOLERANCE", 0.5),
        0
    )
    min_confidence = get_config_float(
        "TREND_TIMING_RESCUE_MIN_CONFIDENCE",
        78
    )
    min_quality = get_config_float(
        "TREND_TIMING_RESCUE_MIN_QUALITY_SCORE",
        1.0
    )
    min_regime = get_config_float(
        "TREND_TIMING_RESCUE_MIN_REGIME_SCORE",
        -1.25
    )
    timing_misses = set()
    reasons = []

    if not getattr(config, "LIVE_ENTRY_CONFIRMATION_ENABLED", True):
        reasons.append("LIVE_ENTRY_GUARD_REQUIRED")

    if not getattr(config, "LIVE_ENTRY_REQUIRE_DIRECTION_SUPPORT", True):
        reasons.append("LIVE_DIRECTION_SUPPORT_REQUIRED")

    if not trend_ok:
        reasons.append("DAILY_TREND_HARD_CHECK_FAILED")

    if not level_ok:
        reasons.append("ADVERSE_LEVEL_CHECK_FAILED")

    trend_min = get_config_float("SIGNAL_MIN_TREND_SCORE", 7.5)

    if _safe_float(trend_score) < trend_min:
        reasons.append(
            f"TREND={round(_safe_float(trend_score), 2)} < {trend_min}"
        )

    if _safe_float(trend_confidence) < min_confidence:
        reasons.append(
            f"CONFIDENCE={round(_safe_float(trend_confidence), 2)} "
            f"< {min_confidence}"
        )

    if _safe_float(quality_score) < min_quality:
        reasons.append(
            f"QUALITY={round(_safe_float(quality_score), 2)} < {min_quality}"
        )

    if _safe_float(regime_score) < min_regime:
        reasons.append(
            f"REGIME={round(_safe_float(regime_score), 2)} < {min_regime}"
        )

    if not bool(entry_quality.get("late_entry_ok", True)):
        reasons.append("LATE_ENTRY_HARD_BLOCK")

    timing_checks = (
        (
            "CONFIRM",
            confirm_ok,
            bool(confirm_quality.get("direction_ok")),
            _safe_float(confirm_score),
            get_config_float("SIGNAL_MIN_CONFIRM_SCORE", 7.0),
        ),
        (
            "ENTRY",
            entry_ok,
            bool(entry_quality.get("direction_ok")),
            _safe_float(entry_score),
            get_config_float("SIGNAL_MIN_ENTRY_SCORE", 4.0),
        ),
    )

    for label, hard_ok, direction_ok, score, minimum in timing_checks:
        if score < minimum:
            if minimum - score <= tolerance:
                timing_misses.add(label)
            else:
                reasons.append(
                    f"{label}={round(score, 2)} < "
                    f"{round(minimum - tolerance, 2)} RESCUE_FLOOR"
                )

        if not hard_ok:
            if direction_ok:
                timing_misses.add(label)
            else:
                reasons.append(f"{label}_DIRECTION_HARD_CHECK_FAILED")

    if len(timing_misses) != 1:
        reasons.append(
            f"TIMING_MISSES={len(timing_misses)} REQUIRED=1"
        )

    eligible = not reasons and len(timing_misses) == 1
    participation_available = bool(
        participation and participation.get("available")
    )
    require_futures = bool(
        getattr(config, "TREND_TIMING_RESCUE_REQUIRE_FUTURES", True)
    )
    min_futures = get_config_float(
        "TREND_TIMING_RESCUE_MIN_FUTURES_SCORE",
        0
    )
    futures_score = _safe_float(participation_score)
    futures_supports = futures_ok and futures_score >= min_futures
    active = eligible

    if require_futures:
        if not participation_available:
            active = False
            reasons.append("FUTURES_CONTEXT_REQUIRED")
        elif not futures_supports:
            active = False
            reasons.append(
                f"FUTURES_SCORE={round(futures_score, 2)} < {min_futures}"
            )
    elif participation_available and not futures_supports:
        active = False
        reasons.append(
            f"FUTURES_SCORE={round(futures_score, 2)} < {min_futures}"
        )

    context.update({
        "eligible": eligible,
        "active": active,
        "missed_module": next(iter(timing_misses), None),
        "reasons": reasons,
        "score_tolerance": tolerance,
        "min_confidence": min_confidence,
        "min_quality_score": min_quality,
        "participation_available": participation_available,
        "futures_score": round(futures_score, 2),
        "min_futures_score": min_futures,
        "reason": (
            "TREND_TIMING_RESCUE_ACTIVE"
            if active
            else (
                "TREND_TIMING_RESCUE_AWAITING_FUTURES"
                if eligible and not participation_available
                else "TREND_TIMING_RESCUE_BLOCKED"
            )
        ),
    })
    return context


def _continuation_pullback_context(
    side,
    entry_df,
    trend_ok,
    confirm_ok,
    entry_ok,
    level_ok,
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    regime_score,
    trend_confidence,
    entry_quality,
    participation_score,
    participation,
    futures_ok
):
    enabled = bool(getattr(config, "CONTINUATION_PULLBACK_ENABLED", True))
    context = {
        "enabled": enabled,
        "eligible": False,
        "active": False,
        "reasons": [],
    }

    if not enabled:
        context["reasons"].append("CONTINUATION_PULLBACK_DISABLED")
        context["reason"] = "CONTINUATION_PULLBACK_DISABLED"
        return context

    reasons = []

    if entry_ok:
        reasons.append("NORMAL_ENTRY_ALREADY_VALID")
    if not trend_ok:
        reasons.append("DAILY_TREND_HARD_CHECK_FAILED")
    if not confirm_ok:
        reasons.append("CONFIRMATION_HARD_CHECK_FAILED")
    if not level_ok:
        reasons.append("ADVERSE_LEVEL_CHECK_FAILED")
    if not bool(entry_quality.get("late_entry_ok", True)):
        reasons.append("LATE_ENTRY_HARD_BLOCK")

    minimum_checks = (
        (
            "CONFIDENCE",
            trend_confidence,
            get_config_float("CONTINUATION_PULLBACK_MIN_CONFIDENCE", 80),
        ),
        (
            "TREND",
            trend_score,
            get_config_float("CONTINUATION_PULLBACK_MIN_TREND_SCORE", 8),
        ),
        (
            "CONFIRM",
            confirm_score,
            get_config_float("CONTINUATION_PULLBACK_MIN_CONFIRM_SCORE", 8),
        ),
        (
            "ENTRY",
            entry_score,
            get_config_float("CONTINUATION_PULLBACK_MIN_ENTRY_SCORE", 2.5),
        ),
        (
            "QUALITY",
            quality_score,
            get_config_float("CONTINUATION_PULLBACK_MIN_QUALITY_SCORE", 1),
        ),
        (
            "REGIME",
            regime_score,
            get_config_float("CONTINUATION_PULLBACK_MIN_REGIME_SCORE", 0),
        ),
    )

    for label, value, minimum in minimum_checks:
        value = _safe_float(value)
        minimum = _safe_float(minimum)

        if value < minimum:
            reasons.append(f"{label}={round(value, 2)} < {minimum}")

    lookback = max(
        get_config_int("CONTINUATION_PULLBACK_STRUCTURE_LOOKBACK", 8),
        2,
    )
    closed = entry_df.iloc[:-1] if len(entry_df) > 1 else entry_df

    if len(closed) < lookback + 1:
        reasons.append(
            f"INSUFFICIENT_ENTRY_HISTORY={len(closed)} < {lookback + 1}"
        )
        candle = latest_closed(entry_df)
        prior = closed.iloc[0:0]
    else:
        candle = closed.iloc[-1]
        prior = closed.iloc[-lookback - 1:-1]

    close = _safe_float(candle.get("close"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    atr = max(_safe_float(candle.get("atr")), 1e-10)
    ema20 = _safe_float(candle.get("ema20"))
    ema50 = _safe_float(candle.get("ema50"))
    rsi = _safe_float(candle.get("rsi"), 50)
    volume = _safe_float(candle.get("volume"))
    volume_sma = _safe_float(candle.get("volume_sma"))
    candle_atr = max(high - low, 0) / atr
    volume_mult = volume / volume_sma if volume_sma > 0 else 0
    ema20_distance_atr = abs(close - ema20) / atr if ema20 else float("inf")
    touch_buffer = max(
        get_config_float("CONTINUATION_PULLBACK_EMA20_TOUCH_ATR", 0.20),
        0,
    ) * atr
    ema50_buffer = max(
        get_config_float(
            "CONTINUATION_PULLBACK_EMA50_BREAK_BUFFER_ATR",
            0.15,
        ),
        0,
    ) * atr
    structure_buffer = max(
        get_config_float(
            "CONTINUATION_PULLBACK_STRUCTURE_BREAK_BUFFER_ATR",
            0.15,
        ),
        0,
    ) * atr
    ema20_touched = low <= ema20 + touch_buffer and high >= ema20 - touch_buffer

    if side == "BUY":
        ema_stack_ok = ema20 > ema50
        ema50_hold_ok = close >= ema50 - ema50_buffer
        structure_level = (
            _safe_float(prior["low"].min())
            if not prior.empty and "low" in prior
            else 0
        )
        structure_ok = bool(structure_level) and close >= structure_level - structure_buffer
        rsi_ok = (
            get_config_float("CONTINUATION_PULLBACK_BUY_MIN_RSI", 44)
            <= rsi <=
            get_config_float("CONTINUATION_PULLBACK_BUY_MAX_RSI", 70)
        )
    else:
        ema_stack_ok = ema20 < ema50
        ema50_hold_ok = close <= ema50 + ema50_buffer
        structure_level = (
            _safe_float(prior["high"].max())
            if not prior.empty and "high" in prior
            else 0
        )
        structure_ok = bool(structure_level) and close <= structure_level + structure_buffer
        rsi_ok = (
            get_config_float("CONTINUATION_PULLBACK_SELL_MIN_RSI", 30)
            <= rsi <=
            get_config_float("CONTINUATION_PULLBACK_SELL_MAX_RSI", 56)
        )

    max_ema20_distance_atr = get_config_float(
        "CONTINUATION_PULLBACK_MAX_EMA20_DISTANCE_ATR",
        0.75,
    )
    max_candle_atr = get_config_float(
        "CONTINUATION_PULLBACK_MAX_CANDLE_ATR",
        1.25,
    )
    max_volume_mult = get_config_float(
        "CONTINUATION_PULLBACK_MAX_VOLUME_MULT",
        1.35,
    )
    setup_checks = (
        ("EMA_STACK_NOT_ALIGNED", ema_stack_ok),
        ("EMA20_NOT_TOUCHED", ema20_touched),
        ("EMA50_HOLD_FAILED", ema50_hold_ok),
        ("ENTRY_STRUCTURE_BROKEN", structure_ok),
        ("PULLBACK_RSI_OUT_OF_RANGE", rsi_ok),
        (
            f"EMA20_DISTANCE_ATR={round(ema20_distance_atr, 2)} > "
            f"{max_ema20_distance_atr}",
            ema20_distance_atr <= max_ema20_distance_atr,
        ),
        (
            f"CANDLE_ATR={round(candle_atr, 2)} > {max_candle_atr}",
            candle_atr <= max_candle_atr,
        ),
        (
            f"VOLUME_MULT={round(volume_mult, 2)} > {max_volume_mult}",
            volume_sma > 0 and volume_mult <= max_volume_mult,
        ),
    )

    for reason, check_ok in setup_checks:
        if not check_ok:
            reasons.append(reason)

    eligible = not reasons
    participation_available = bool(
        participation and participation.get("available")
    )
    require_futures = bool(
        getattr(config, "CONTINUATION_PULLBACK_REQUIRE_FUTURES", True)
    )
    min_futures = get_config_float(
        "CONTINUATION_PULLBACK_MIN_FUTURES_SCORE",
        0.5,
    )
    futures_score = _safe_float(participation_score)
    futures_supports = futures_ok and futures_score >= min_futures
    active = eligible

    if require_futures:
        if not participation_available:
            active = False
            reasons.append("FUTURES_CONTEXT_REQUIRED")
        elif not futures_supports:
            active = False
            reasons.append(
                f"FUTURES_SCORE={round(futures_score, 2)} < {min_futures}"
            )
    elif participation_available and not futures_supports:
        active = False
        reasons.append(
            f"FUTURES_SCORE={round(futures_score, 2)} < {min_futures}"
        )

    context.update({
        "eligible": eligible,
        "active": active,
        "reasons": reasons,
        "ema_stack_ok": ema_stack_ok,
        "ema20_touched": ema20_touched,
        "ema50_hold_ok": ema50_hold_ok,
        "structure_ok": structure_ok,
        "structure_level": round(structure_level, 8),
        "rsi": round(rsi, 2),
        "ema20_distance_atr": round(ema20_distance_atr, 2),
        "candle_atr": round(candle_atr, 2),
        "volume_mult": round(volume_mult, 2),
        "participation_available": participation_available,
        "futures_score": round(futures_score, 2),
        "min_futures_score": min_futures,
        "reason": (
            "CONTINUATION_PULLBACK_ACTIVE"
            if active
            else (
                "CONTINUATION_PULLBACK_AWAITING_FUTURES"
                if eligible and not participation_available
                else "CONTINUATION_PULLBACK_BLOCKED"
            )
        ),
    })
    return context


def _counter_trend_context(side, trend_df, confirm_df):
    trend = latest_closed(trend_df)
    confirm = latest_closed(confirm_df)

    if side == "BUY":
        checks = {
            "trend_close_below_ema50": trend["close"] < trend["ema50"],
            "trend_ema20_below_ema50": trend["ema20"] < trend["ema50"],
            "trend_ema50_below_ema200": trend["ema50"] < trend["ema200"],
            "confirm_close_below_ema50": confirm["close"] < confirm["ema50"],
        }
    else:
        checks = {
            "trend_close_above_ema50": trend["close"] > trend["ema50"],
            "trend_ema20_above_ema50": trend["ema20"] > trend["ema50"],
            "trend_ema50_above_ema200": trend["ema50"] > trend["ema200"],
            "confirm_close_above_ema50": confirm["close"] > confirm["ema50"],
        }

    count = sum(1 for value in checks.values() if value)
    return count >= 2, {"count": count, "checks": checks}


def _reversal_smc_check(smc_score, smc_context):
    if not getattr(config, "SMC_ENABLED", True):
        return False, {"reason": "SMC_DISABLED"}

    smc_context = smc_context or {}
    supportive = {
        "liquidity_sweep": bool(smc_context.get("liquidity_sweep")),
        "order_block": bool(smc_context.get("order_block")),
        "fvg_support": bool(smc_context.get("fvg_support")),
    }
    has_support = any(supportive.values())
    min_smc = get_config_float("REVERSAL_MIN_SMC_SCORE", 1.0)

    return (
        has_support and _safe_float(smc_score) >= min_smc,
        {
            "supportive": supportive,
            "has_block": bool(smc_context.get("fvg_block")),
            "min_smc": min_smc,
        }
    )


def _directional_candle_context(side, candle):
    open_price = _safe_float(candle.get("open"))
    high = _safe_float(candle.get("high"))
    low = _safe_float(candle.get("low"))
    close = _safe_float(candle.get("close"))
    atr = _candle_atr(candle)
    ema20 = _safe_float(candle.get("ema20"))
    macd = _safe_float(candle.get("macd"))
    macd_signal = _safe_float(candle.get("macd_signal"))
    rsi = _safe_float(candle.get("rsi"), 50)
    candle_range = max(high - low, 1e-10)
    close_position = (close - low) / candle_range
    directional_close = close_position if side == "BUY" else 1 - close_position
    directional_body = close - open_price if side == "BUY" else open_price - close
    body_atr = directional_body / atr
    tolerance = get_config_float("REVERSAL_MOMENTUM_EMA_TOLERANCE_PCT", 0.25) / 100

    if side == "BUY":
        direction_ok = close > open_price
        ema_reclaimed = ema20 > 0 and close >= ema20
        ema_near = ema20 > 0 and close >= ema20 * (1 - tolerance)
        oscillator_ok = macd > macd_signal or rsi > 50
    else:
        direction_ok = close < open_price
        ema_reclaimed = ema20 > 0 and close <= ema20
        ema_near = ema20 > 0 and close <= ema20 * (1 + tolerance)
        oscillator_ok = macd < macd_signal or rsi < 50

    return {
        "direction_ok": direction_ok,
        "directional_close": round(float(directional_close), 3),
        "body_atr": round(float(body_atr), 3),
        "ema_reclaimed": ema_reclaimed,
        "ema_near": ema_near,
        "oscillator_ok": oscillator_ok,
        "close": close,
        "rsi": round(float(rsi), 2),
    }


def _reversal_momentum_context(side, confirm_df, entry_df, smc_ok):
    enabled = bool(getattr(config, "REVERSAL_MOMENTUM_SCORE_ENABLED", True))
    context = {"enabled": enabled, "ok": False, "score": 0}

    if not enabled:
        context["reason"] = "REVERSAL_MOMENTUM_DISABLED"
        return 0, context

    lookback = max(get_config_int("REVERSAL_MOMENTUM_LOOKBACK", 3), 2)
    min_candles = max(get_config_int("REVERSAL_MOMENTUM_MIN_CANDLES", 2), 1)
    min_body_atr = get_config_float("REVERSAL_MOMENTUM_MIN_BODY_ATR", 0.20)
    min_close_position = get_config_float(
        "REVERSAL_MOMENTUM_MIN_CLOSE_POSITION",
        0.58
    )
    entry = latest_closed(entry_df)
    confirm = latest_closed(confirm_df)
    prev_confirm = previous_closed(confirm_df)
    entry_context = _directional_candle_context(side, entry)
    confirm_context = _directional_candle_context(side, confirm)
    entry_data = _closed_data(entry_df, lookback + 1)
    recent = entry_data.tail(lookback)

    if side == "BUY":
        direction_count = sum(1 for _, candle in recent.iterrows() if _is_bullish(candle))
        structure_break = (
            len(entry_data) > 1 and
            entry_context["close"] > entry_data.iloc[:-1]["high"].tail(lookback).max()
        )
        confirm_shift = (
            confirm["close"] > prev_confirm["close"] or
            confirm["rsi"] > prev_confirm["rsi"] or
            confirm["macd"] > prev_confirm["macd"]
        )
    else:
        direction_count = sum(1 for _, candle in recent.iterrows() if _is_bearish(candle))
        structure_break = (
            len(entry_data) > 1 and
            entry_context["close"] < entry_data.iloc[:-1]["low"].tail(lookback).min()
        )
        confirm_shift = (
            confirm["close"] < prev_confirm["close"] or
            confirm["rsi"] < prev_confirm["rsi"] or
            confirm["macd"] < prev_confirm["macd"]
        )

    score = 0
    score += 1.0 if entry_context["direction_ok"] else 0
    score += 1.0 if entry_context["body_atr"] >= min_body_atr else 0
    score += 0.75 if entry_context["directional_close"] >= min_close_position else 0
    score += 1.0 if direction_count >= min_candles else 0
    score += 1.25 if entry_context["ema_reclaimed"] else 0.5 if entry_context["ema_near"] else 0
    score += 1.0 if structure_break else 0
    score += 0.75 if entry_context["oscillator_ok"] else 0
    score += 0.75 if confirm_context["direction_ok"] else 0
    score += 0.75 if confirm_shift else 0
    score += 0.75 if confirm_context["ema_reclaimed"] else 0.35 if confirm_context["ema_near"] else 0
    score += 0.5 if smc_ok else 0
    score = round(float(score), 2)

    ok = (
        score >= get_config_float("REVERSAL_MOMENTUM_MIN_SCORE", 4.0)
        and entry_context["direction_ok"]
        and (
            entry_context["ema_near"] or
            structure_break or
            smc_ok
        )
        and (
            confirm_context["direction_ok"] or
            confirm_shift
        )
    )
    context.update({
        "ok": ok,
        "score": score,
        "entry": entry_context,
        "confirm": confirm_context,
        "direction_count": direction_count,
        "min_candles": min_candles,
        "structure_break": structure_break,
        "confirm_shift": bool(confirm_shift),
        "smc_ok": bool(smc_ok),
    })

    if not ok:
        context["reason"] = "REVERSAL_MOMENTUM_WEAK"

    return score, context


def _continuation_pressure_against_side(side, trend_df, confirm_df, entry_df):
    trend = latest_closed(trend_df)
    prev_trend = previous_closed(trend_df)
    confirm = latest_closed(confirm_df)
    prev_confirm = previous_closed(confirm_df)
    entry = latest_closed(entry_df)
    prev_entry = previous_closed(entry_df)
    min_adx = get_config_float("REVERSAL_INVALIDATION_MIN_ADX", 18)

    if side == "BUY":
        checks = (
            ("trend_close_below_ema50", trend["close"] < trend["ema50"], 1.0),
            ("trend_ema20_below_ema50", trend["ema20"] < trend["ema50"], 1.0),
            ("trend_ema50_below_ema200", trend["ema50"] < trend["ema200"], 1.25),
            ("trend_lower_close", trend["close"] < prev_trend["close"], 0.75),
            ("trend_strong_adx", trend["adx"] >= min_adx, 0.75),
            ("confirm_close_below_ema20", confirm["close"] < confirm["ema20"], 1.0),
            ("confirm_close_below_ema50", confirm["close"] < confirm["ema50"], 1.0),
            ("confirm_macd_bearish", confirm["macd"] < confirm["macd_signal"], 1.0),
            ("confirm_breakdown", confirm["close"] < prev_confirm["low"], 1.0),
            ("entry_close_below_ema20", entry["close"] < entry["ema20"], 0.75),
            ("entry_bearish", _is_bearish(entry), 0.75),
            ("entry_breakdown", entry["close"] < prev_entry["low"], 0.75),
        )
    else:
        checks = (
            ("trend_close_above_ema50", trend["close"] > trend["ema50"], 1.0),
            ("trend_ema20_above_ema50", trend["ema20"] > trend["ema50"], 1.0),
            ("trend_ema50_above_ema200", trend["ema50"] > trend["ema200"], 1.25),
            ("trend_higher_close", trend["close"] > prev_trend["close"], 0.75),
            ("trend_strong_adx", trend["adx"] >= min_adx, 0.75),
            ("confirm_close_above_ema20", confirm["close"] > confirm["ema20"], 1.0),
            ("confirm_close_above_ema50", confirm["close"] > confirm["ema50"], 1.0),
            ("confirm_macd_bullish", confirm["macd"] > confirm["macd_signal"], 1.0),
            ("confirm_breakout", confirm["close"] > prev_confirm["high"], 1.0),
            ("entry_close_above_ema20", entry["close"] > entry["ema20"], 0.75),
            ("entry_bullish", _is_bullish(entry), 0.75),
            ("entry_breakout", entry["close"] > prev_entry["high"], 0.75),
        )

    active = [name for name, ok, _ in checks if ok]
    score = sum(weight for _, ok, weight in checks if ok)

    return {
        "score": round(float(score), 2),
        "active": active,
        "min_adx": min_adx,
    }


def _reversal_recovery_score(side, confirm_df, entry_df, momentum_context, smc_ok):
    entry = latest_closed(entry_df)
    prev_entry = previous_closed(entry_df)
    confirm = latest_closed(confirm_df)
    prev_confirm = previous_closed(confirm_df)
    entry_context = _directional_candle_context(side, entry)
    confirm_context = _directional_candle_context(side, confirm)
    momentum_context = momentum_context or {}

    if side == "BUY":
        entry_break = entry["close"] > prev_entry["high"]
        confirm_break = confirm["close"] > prev_confirm["high"]
        confirm_shift = (
            confirm["close"] > prev_confirm["close"] or
            confirm["rsi"] > prev_confirm["rsi"] or
            confirm["macd"] > prev_confirm["macd"]
        )
    else:
        entry_break = entry["close"] < prev_entry["low"]
        confirm_break = confirm["close"] < prev_confirm["low"]
        confirm_shift = (
            confirm["close"] < prev_confirm["close"] or
            confirm["rsi"] < prev_confirm["rsi"] or
            confirm["macd"] < prev_confirm["macd"]
        )

    min_close_position = get_config_float(
        "REVERSAL_MOMENTUM_MIN_CLOSE_POSITION",
        0.58
    )
    score = 0
    score += 0.75 if entry_context["direction_ok"] else 0
    score += 0.75 if entry_context["directional_close"] >= min_close_position else 0
    score += 1.0 if entry_context["ema_reclaimed"] else 0.35 if entry_context["ema_near"] else 0
    score += 1.0 if entry_break else 0
    score += 0.75 if confirm_context["direction_ok"] else 0
    score += 1.0 if confirm_context["ema_reclaimed"] else 0.35 if confirm_context["ema_near"] else 0
    score += 0.75 if confirm_shift else 0
    score += 0.75 if confirm_break else 0
    score += 0.75 if momentum_context.get("ok") else 0
    score += 0.75 if momentum_context.get("structure_break") else 0
    score += 0.5 if smc_ok else 0

    return {
        "score": round(float(score), 2),
        "entry": entry_context,
        "confirm": confirm_context,
        "entry_break": bool(entry_break),
        "confirm_break": bool(confirm_break),
        "confirm_shift": bool(confirm_shift),
        "momentum_ok": bool(momentum_context.get("ok")),
        "smc_ok": bool(smc_ok),
    }


def _adaptive_dca_atr_multiplier(dca_level):
    multipliers = list(
        getattr(config, "DCA_ADAPTIVE_ATR_MULTIPLIERS", []) or []
    )

    if not multipliers:
        return 1.0

    index = min(max(int(dca_level or 1) - 1, 0), len(multipliers) - 1)
    return max(_safe_float(multipliers[index], 1.0), 0)


def _adaptive_dca_trend_thesis(side, trend_df, trade_type):
    context = {
        "valid": True,
        "reason": "DCA_ADAPTIVE_THESIS_OK",
        "opposing_alignment": False,
        "ema200_broken": False,
        "momentum_opposing": False,
        "structure_break": False,
        "adx": 0.0,
    }

    if (
        trade_type != "TREND" or
        not getattr(
            config,
            "DCA_ADAPTIVE_BLOCK_TREND_THESIS_INVALIDATION",
            True,
        )
    ):
        context["reason"] = "DCA_ADAPTIVE_THESIS_NOT_REQUIRED"
        return context

    trend = latest_closed(trend_df)
    previous = previous_closed(trend_df)
    adx = _safe_float(trend.get("adx"))
    min_adx = get_config_float("DCA_ADAPTIVE_THESIS_MIN_ADX", 18)

    if side == "BUY":
        opposing_alignment = (
            trend["ema20"] < trend["ema50"] < trend["ema200"]
        )
        ema200_broken = trend["close"] < trend["ema200"]
        momentum_opposing = trend["macd"] < trend["macd_signal"]
        structure_break = trend["close"] < previous["low"]
    else:
        opposing_alignment = (
            trend["ema20"] > trend["ema50"] > trend["ema200"]
        )
        ema200_broken = trend["close"] > trend["ema200"]
        momentum_opposing = trend["macd"] > trend["macd_signal"]
        structure_break = trend["close"] > previous["high"]

    invalid = bool(
        adx >= min_adx and
        opposing_alignment and
        ema200_broken and
        (momentum_opposing or structure_break)
    )
    context.update({
        "valid": not invalid,
        "reason": (
            "DCA_ADAPTIVE_1D_THESIS_INVALIDATED"
            if invalid
            else "DCA_ADAPTIVE_THESIS_OK"
        ),
        "opposing_alignment": bool(opposing_alignment),
        "ema200_broken": bool(ema200_broken),
        "momentum_opposing": bool(momentum_opposing),
        "structure_break": bool(structure_break),
        "adx": round(adx, 2),
        "min_adx": min_adx,
    })
    return context


def validate_adaptive_dca_trigger(
    side,
    current_price,
    trend_df,
    confirm_df,
    dca_level,
    adverse_roi,
    trigger_roi,
    spacing_anchor_price,
    confirmation_type,
    recovery,
    structure_ok,
    structure_info,
):
    mode = str(getattr(config, "DCA_TRIGGER_MODE", "static_roi") or "").lower()
    info = {
        "enabled": mode == "adaptive_hybrid",
        "reason": "DCA_ADAPTIVE_DISABLED",
        "dca_level": int(dca_level or 1),
        "adverse_roi": round(_safe_float(adverse_roi), 2),
        "trigger_roi": round(_safe_float(trigger_roi), 2),
    }

    if mode != "adaptive_hybrid":
        return True, info

    current_price = _safe_float(current_price)
    spacing_anchor_price = _safe_float(spacing_anchor_price)
    trigger_roi = _safe_float(trigger_roi)
    adverse_roi = _safe_float(adverse_roi)
    max_adverse_roi = max(
        get_config_float("DCA_MAX_ADVERSE_ROI", 0),
        0,
    )

    if current_price <= 0 or spacing_anchor_price <= 0:
        info["reason"] = "DCA_ADAPTIVE_INVALID_PRICE"
        return False, info

    if adverse_roi < trigger_roi:
        info["reason"] = "DCA_ADAPTIVE_ROI_FLOOR_NOT_REACHED"
        return False, info

    if max_adverse_roi > 0 and adverse_roi > max_adverse_roi:
        info.update({
            "reason": "DCA_ADAPTIVE_MAX_RISK_EXCEEDED",
            "max_adverse_roi": max_adverse_roi,
        })
        return False, info

    confirm = latest_closed(confirm_df)
    atr = max(_safe_float(confirm.get("atr")), 0)
    multiplier = _adaptive_dca_atr_multiplier(dca_level)
    directional_gap = (
        spacing_anchor_price - current_price
        if side == "BUY"
        else current_price - spacing_anchor_price
    )
    gap_atr = directional_gap / atr if atr > 0 else 0
    info.update({
        "spacing_anchor_price": round(spacing_anchor_price, 10),
        "atr": round(atr, 10),
        "atr_multiplier": multiplier,
        "gap_atr": round(gap_atr, 2),
    })

    if atr <= 0:
        info["reason"] = "DCA_ADAPTIVE_ATR_UNAVAILABLE"
        return False, info

    if directional_gap <= 0 or gap_atr < multiplier:
        info["reason"] = "DCA_ADAPTIVE_ATR_SPACING_NOT_REACHED"
        return False, info

    thesis = _adaptive_dca_trend_thesis(
        side,
        trend_df,
        str(confirmation_type or "").upper(),
    )
    info["thesis"] = thesis

    if not thesis.get("valid"):
        info["reason"] = thesis.get(
            "reason",
            "DCA_ADAPTIVE_THESIS_INVALIDATED",
        )
        return False, info

    if (
        getattr(config, "DCA_ADAPTIVE_REQUIRE_STRUCTURE", True) and
        not structure_ok
    ):
        info.update({
            "reason": structure_info.get(
                "reason",
                "DCA_ADAPTIVE_STRUCTURE_REQUIRED",
            ),
            "structure": structure_info,
        })
        return False, info

    route = str(confirmation_type or "").upper()
    base_recovery = (
        get_config_float("DCA_ADAPTIVE_REVERSAL_MIN_RECOVERY_SCORE", 3.5)
        if route == "REVERSAL"
        else get_config_float("DCA_ADAPTIVE_MIN_RECOVERY_SCORE", 2.5)
    )
    recovery_step = max(
        get_config_float("DCA_ADAPTIVE_RECOVERY_STEP_PER_LEVEL", 0.25),
        0,
    )
    required_recovery = base_recovery + (
        max(int(dca_level or 1) - 1, 0) * recovery_step
    )
    recovery_score = _safe_float((recovery or {}).get("score"))
    info.update({
        "recovery_score": round(recovery_score, 2),
        "required_recovery": round(required_recovery, 2),
        "structure": structure_info,
    })

    if (
        getattr(config, "DCA_ADAPTIVE_REQUIRE_RECOVERY", True) and
        recovery_score < required_recovery
    ):
        info["reason"] = "DCA_ADAPTIVE_RECOVERY_NOT_CONFIRMED"
        return False, info

    info["reason"] = "DCA_ADAPTIVE_TRIGGER_OK"
    return True, info


def validate_reversal_invalidation(
    side,
    trend_df,
    confirm_df,
    entry_df,
    momentum_context=None,
    smc_ok=False,
):
    if not getattr(config, "REVERSAL_INVALIDATION_ENABLED", True):
        return True, {"reason": "REVERSAL_INVALIDATION_DISABLED"}

    pressure = _continuation_pressure_against_side(
        side,
        trend_df,
        confirm_df,
        entry_df,
    )
    recovery = _reversal_recovery_score(
        side,
        confirm_df,
        entry_df,
        momentum_context,
        smc_ok,
    )
    pressure_score = float(pressure.get("score", 0))
    recovery_score = float(recovery.get("score", 0))
    max_pressure = get_config_float("REVERSAL_INVALIDATION_MIN_PRESSURE_SCORE", 5.5)
    hard_pressure = get_config_float("REVERSAL_INVALIDATION_HARD_PRESSURE_SCORE", 7.0)
    min_recovery = get_config_float("REVERSAL_INVALIDATION_MIN_RECOVERY_SCORE", 3.0)
    require_recovery = bool(
        getattr(config, "REVERSAL_REQUIRE_RECOVERY_CONFIRMATION", True)
    )
    base_recovery_min = get_config_float("REVERSAL_RECOVERY_MIN_SCORE", 3.25)
    strong_pressure = get_config_float("REVERSAL_RECOVERY_STRONG_PRESSURE_SCORE", 4.75)
    strong_pressure_min = get_config_float(
        "REVERSAL_RECOVERY_STRONG_PRESSURE_MIN_SCORE",
        4.0
    )
    required_recovery = (
        max(base_recovery_min, strong_pressure_min)
        if pressure_score >= strong_pressure
        else base_recovery_min
    )

    info = {
        "reason": "REVERSAL_INVALIDATION_OK",
        "pressure": pressure,
        "recovery": recovery,
        "pressure_score": round(pressure_score, 2),
        "recovery_score": round(recovery_score, 2),
        "max_pressure": max_pressure,
        "hard_pressure": hard_pressure,
        "min_recovery": min_recovery,
        "required_recovery": round(float(required_recovery), 2),
    }

    if require_recovery and recovery_score < required_recovery:
        info["reason"] = (
            f"REVERSAL_RECOVERY_NOT_CONFIRMED "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score} "
            f"REQUIRED={required_recovery}"
        )
        return False, info

    if pressure_score >= hard_pressure and recovery_score < min_recovery + 0.75:
        info["reason"] = (
            f"REVERSAL_INVALIDATED_HARD_PRESSURE "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score}"
        )
        return False, info

    if pressure_score >= max_pressure and recovery_score < min_recovery:
        info["reason"] = (
            f"REVERSAL_INVALIDATED_PRESSURE "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score}"
        )
        return False, info

    return True, info


def validate_dca_continuation_guard(
    side,
    current_price,
    avg_entry,
    trend_df,
    confirm_df,
    entry_df,
    leverage=None,
    confirmation_type=None,
    dca_level=1,
    adverse_roi=0,
    position_adverse_roi=0,
    trigger_roi=None,
    spacing_anchor_price=None,
):
    strict_enabled = bool(
        getattr(config, "DCA_STRICT_GUARD_ENABLED", True)
    )
    adaptive_enabled = (
        str(getattr(config, "DCA_TRIGGER_MODE", "static_roi")).lower() ==
        "adaptive_hybrid"
    )

    if not strict_enabled and not adaptive_enabled:
        return True, {"reason": "DCA_STRICT_GUARD_DISABLED"}

    trade_type = str(confirmation_type or "").upper()

    if (
        getattr(config, "DCA_STRICT_GUARD_APPLY_TO_REVERSAL_ONLY", False)
        and trade_type != "REVERSAL"
        and not adaptive_enabled
    ):
        return True, {"reason": "DCA_STRICT_GUARD_NON_REVERSAL_SKIPPED"}

    if trend_df is None or confirm_df is None or entry_df is None:
        if getattr(config, "DCA_STRICT_GUARD_REQUIRE_DATA", True):
            return False, {"reason": "DCA_STRICT_GUARD_DATA_UNAVAILABLE"}

        return True, {"reason": "DCA_STRICT_GUARD_DATA_UNAVAILABLE_ALLOWED"}

    pressure = _continuation_pressure_against_side(
        side,
        trend_df,
        confirm_df,
        entry_df,
    )
    recovery = _reversal_recovery_score(
        side,
        confirm_df,
        entry_df,
        momentum_context=None,
        smc_ok=False,
    )
    pressure_score = float(pressure.get("score", 0))
    recovery_score = float(recovery.get("score", 0))
    max_pressure = get_config_float("DCA_STRICT_GUARD_MAX_PRESSURE_SCORE", 5.5)
    hard_pressure = get_config_float("DCA_STRICT_GUARD_HARD_PRESSURE_SCORE", 7.0)
    min_recovery = (
        get_config_float("DCA_STRICT_GUARD_REVERSAL_MIN_RECOVERY_SCORE", 3.25)
        if trade_type == "REVERSAL"
        else get_config_float("DCA_STRICT_GUARD_MIN_RECOVERY_SCORE", 2.5)
    )

    if trade_type == "REVERSAL" and int(dca_level or 1) <= 1:
        min_recovery = max(
            min_recovery,
            get_config_float(
                "DCA_STRICT_GUARD_REVERSAL_FIRST_DCA_MIN_RECOVERY_SCORE",
                4.25
            )
        )
        max_pressure = min(
            max_pressure,
            get_config_float(
                "DCA_STRICT_GUARD_REVERSAL_FIRST_DCA_MAX_PRESSURE_SCORE",
                4.75
            )
        )

    max_adverse_roi = get_config_float("DCA_STRICT_GUARD_MAX_ADVERSE_ROI", 0)
    structure_info = {"reason": "DCA_STRICT_GUARD_STRUCTURE_NOT_CHECKED"}

    if getattr(config, "DCA_STRICT_GUARD_STRUCTURE_CHECK_ENABLED", True):
        structure_ok, structure_info = validate_dca_structure_level(
            side,
            current_price,
            trend_df,
            confirm_df,
            entry_df,
            leverage=leverage,
        )
    else:
        structure_ok = True

    info = {
        "reason": "DCA_STRICT_GUARD_OK",
        "trade_type": trade_type or "UNKNOWN",
        "dca_level": dca_level,
        "current_price": current_price,
        "avg_entry": avg_entry,
        "adverse_roi": adverse_roi,
        "position_adverse_roi": position_adverse_roi,
        "pressure": pressure,
        "recovery": recovery,
        "pressure_score": round(pressure_score, 2),
        "recovery_score": round(recovery_score, 2),
        "max_pressure": max_pressure,
        "hard_pressure": hard_pressure,
        "min_recovery": min_recovery,
        "structure": structure_info,
    }

    adaptive_ok, adaptive_info = validate_adaptive_dca_trigger(
        side,
        current_price,
        trend_df,
        confirm_df,
        dca_level,
        adverse_roi,
        adverse_roi if trigger_roi is None else trigger_roi,
        avg_entry if spacing_anchor_price is None else spacing_anchor_price,
        trade_type,
        recovery,
        structure_ok,
        structure_info,
    )
    info["adaptive"] = adaptive_info

    if not adaptive_ok:
        info["reason"] = adaptive_info.get(
            "reason",
            "DCA_ADAPTIVE_TRIGGER_BLOCKED",
        )
        return False, info

    if trade_type == "REVERSAL" and int(dca_level or 1) <= 1 and recovery_score < min_recovery:
        info["reason"] = (
            f"DCA_STRICT_GUARD_REVERSAL_FIRST_DCA_WEAK_RECOVERY "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score} "
            f"REQUIRED={min_recovery}"
        )
        return False, info

    if (
        max_adverse_roi > 0
        and float(position_adverse_roi or adverse_roi or 0) >= max_adverse_roi
        and recovery_score < min_recovery + 0.5
    ):
        info["reason"] = (
            f"DCA_STRICT_GUARD_MAX_ADVERSE "
            f"ROI={position_adverse_roi} MAX={max_adverse_roi}"
        )
        return False, info

    if pressure_score >= hard_pressure and recovery_score < min_recovery + 0.75:
        info["reason"] = (
            f"DCA_STRICT_GUARD_HARD_PRESSURE "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score}"
        )
        return False, info

    if pressure_score >= max_pressure and recovery_score < min_recovery:
        info["reason"] = (
            f"DCA_STRICT_GUARD_OPPOSITE_PRESSURE "
            f"PRESSURE={pressure_score} RECOVERY={recovery_score}"
        )
        return False, info

    block_on_no_structure = (
        getattr(config, "DCA_STRICT_GUARD_BLOCK_REVERSAL_ON_NO_STRUCTURE", True)
        if trade_type == "REVERSAL"
        else getattr(config, "DCA_STRICT_GUARD_BLOCK_TREND_ON_NO_STRUCTURE", False)
    )

    if not structure_ok and block_on_no_structure:
        info["reason"] = structure_info.get(
            "reason",
            "DCA_STRICT_GUARD_STRUCTURE_FAILED"
        )
        return False, info

    return True, info


def _reversal_signal_check(
    side,
    trend_df,
    confirm_df,
    entry_df,
    trend_score,
    confirm_score,
    entry_score,
    quality_score,
    smc_score,
    smc_context,
    momentum_context,
    regime_score,
    confidence,
    confirm_ok,
    entry_ok,
    level_ok
):
    context = {"enabled": bool(getattr(config, "REVERSAL_MODE_ENABLED", True))}
    if not context["enabled"]:
        return False, ["REVERSAL_DISABLED"], context

    failures = []
    counter_ok, counter_context = _counter_trend_context(side, trend_df, confirm_df)
    smc_ok, smc_details = _reversal_smc_check(smc_score, smc_context)
    momentum_context = momentum_context or {}
    momentum_ok = bool(momentum_context.get("ok"))
    invalidation_ok, invalidation = validate_reversal_invalidation(
        side,
        trend_df,
        confirm_df,
        entry_df,
        momentum_context=momentum_context,
        smc_ok=smc_ok,
    )
    context.update({
        "counter_trend": counter_context,
        "smc": smc_details,
        "momentum": momentum_context,
        "invalidation": invalidation,
    })

    if getattr(config, "REVERSAL_REQUIRE_COUNTER_TREND", True) and not counter_ok:
        failures.append("COUNTER_TREND_CONTEXT_MISSING")

    if (
        getattr(config, "REVERSAL_REQUIRE_SMC", True)
        and not smc_ok
        and not (
            momentum_ok and
            getattr(config, "REVERSAL_ALLOW_MOMENTUM_WITHOUT_SMC", False)
        )
    ):
        failures.append("SMC_REVERSAL_EVIDENCE_MISSING")

    confidence_threshold = get_config_float("REVERSAL_SIGNAL_THRESHOLD", 78)
    confirm_min = get_config_float("REVERSAL_MIN_CONFIRM_SCORE", 8)
    entry_min = get_config_float("REVERSAL_MIN_ENTRY_SCORE", 5)
    quality_min = get_config_float("REVERSAL_MIN_QUALITY_SCORE", 0.75)
    regime_min = get_config_float("REVERSAL_MIN_REGIME_SCORE", -1.0)

    if momentum_ok:
        confidence_threshold = get_config_float(
            "REVERSAL_MOMENTUM_SIGNAL_THRESHOLD",
            confidence_threshold
        )
        confirm_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_CONFIRM_SCORE",
            confirm_min
        )
        entry_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_ENTRY_SCORE",
            entry_min
        )
        quality_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_QUALITY_SCORE",
            quality_min
        )
        regime_min = get_config_float(
            "REVERSAL_MOMENTUM_MIN_REGIME_SCORE",
            regime_min
        )

    recovery_score = _safe_float(invalidation.get("recovery_score", 0))
    pressure_score = _safe_float(invalidation.get("pressure_score", 0))
    required_recovery = _safe_float(
        invalidation.get(
            "required_recovery",
            get_config_float("REVERSAL_RECOVERY_MIN_SCORE", 3.25)
        )
    )
    confirmation_items = {
        "counter_trend": bool(counter_ok),
        "smc": bool(smc_ok),
        "momentum": bool(momentum_ok),
        "recovery": recovery_score >= required_recovery,
        "confirm_score": _safe_float(confirm_score) >= confirm_min,
        "entry_score": _safe_float(entry_score) >= entry_min,
        "quality_score": _safe_float(quality_score) >= quality_min,
        "level": bool(level_ok),
    }
    confirmation_points = sum(
        1 for ok in confirmation_items.values() if ok
    )
    min_confirmation_points = get_config_float(
        "REVERSAL_MIN_CONFIRMATION_POINTS",
        5
    )

    if pressure_score >= get_config_float(
        "REVERSAL_CONFIRMATION_STRONG_PRESSURE_SCORE",
        5.0
    ):
        min_confirmation_points = max(
            min_confirmation_points,
            get_config_float(
                "REVERSAL_STRONG_PRESSURE_MIN_CONFIRMATION_POINTS",
                6
            )
        )

    context["extra_confirmation"] = {
        "enabled": bool(getattr(config, "REVERSAL_EXTRA_CONFIRMATION_ENABLED", True)),
        "items": confirmation_items,
        "points": confirmation_points,
        "required": min_confirmation_points,
        "pressure_score": pressure_score,
        "recovery_score": recovery_score,
        "required_recovery": required_recovery,
    }

    if (
        getattr(config, "REVERSAL_REQUIRE_MOMENTUM_OR_RECOVERY", True)
        and not momentum_ok
        and recovery_score < required_recovery
    ):
        failures.append(
            f"REVERSAL_MOMENTUM_OR_RECOVERY_MISSING "
            f"RECOVERY={recovery_score} REQUIRED={required_recovery}"
        )

    if (
        getattr(config, "REVERSAL_EXTRA_CONFIRMATION_ENABLED", True)
        and confirmation_points < min_confirmation_points
    ):
        failures.append(
            f"REVERSAL_CONFIRMATIONS_LOW "
            f"POINTS={confirmation_points} REQUIRED={min_confirmation_points}"
        )

    checks = (
        (
            "CONFIDENCE",
            confidence,
            confidence_threshold,
            ">="
        ),
        (
            "CONFIRM",
            confirm_score,
            confirm_min,
            ">="
        ),
        (
            "ENTRY",
            entry_score,
            entry_min,
            ">="
        ),
        (
            "QUALITY",
            quality_score,
            quality_min,
            ">="
        ),
        (
            "REGIME",
            regime_score,
            regime_min,
            ">="
        ),
        (
            "TREND",
            trend_score,
            get_config_float("REVERSAL_MAX_TREND_SCORE", 8),
            "<="
        ),
    )

    for label, value, limit, operator in checks:
        value = _safe_float(value)
        limit = _safe_float(limit)

        if operator == ">=" and value < limit:
            failures.append(f"{label}={round(value, 2)} < {limit}")
        elif operator == "<=" and value > limit:
            failures.append(f"{label}={round(value, 2)} > {limit}")

    if not confirm_ok and not momentum_ok:
        failures.append("CONFIRMATION_HARD_CHECK_FAILED")

    if not entry_ok and not momentum_ok:
        failures.append("ENTRY_HARD_CHECK_FAILED")

    if not level_ok:
        failures.append("LEVEL_CHECK_FAILED")

    if not invalidation_ok:
        failures.append(invalidation.get("reason", "REVERSAL_INVALIDATED"))

    return not failures, failures, context


def _reversal_futures_confirmation_context(
    chart_reversal_ok,
    participation_score,
    participation,
    futures_ok,
):
    required = bool(
        getattr(config, "REVERSAL_REQUIRE_FUTURES_CONFIRMATION", True)
    )
    available = bool(participation and participation.get("available"))
    score = _safe_float(participation_score)
    minimum = get_config_float("REVERSAL_MIN_FUTURES_SCORE", 0.5)
    eligible = bool(chart_reversal_ok)
    active = bool(
        eligible and
        (
            not required or
            (available and futures_ok and score >= minimum)
        )
    )
    reasons = []

    if eligible and required and not available:
        reasons.append("REVERSAL_FUTURES_CONTEXT_REQUIRED")
    elif eligible and required and not futures_ok:
        reasons.append("REVERSAL_FUTURES_CONTEXT_CONFLICT")
    elif eligible and required and score < minimum:
        reasons.append(
            f"REVERSAL_FUTURES_SCORE={round(score, 2)} < {minimum}"
        )

    return {
        "required": required,
        "eligible": eligible,
        "active": active,
        "available": available,
        "score": round(score, 2),
        "minimum": minimum,
        "reasons": reasons,
        "reason": (
            "REVERSAL_FUTURES_CONFIRMED"
            if active
            else (
                "REVERSAL_FUTURES_AWAITING_CONTEXT"
                if eligible and required and not available
                else "REVERSAL_FUTURES_BLOCKED"
            )
        ),
    }


def evaluate_route_profit_protection(
    side,
    avg_entry,
    current_price,
    peak_roi=0,
    leverage=None,
    confirmation_type="REVERSAL",
):
    route = (
        "REVERSAL"
        if str(confirmation_type or "").upper() == "REVERSAL"
        else "TREND"
    )
    prefix = f"{route}_PROFIT_PROTECTION"
    enabled = bool(
        getattr(
            config,
            f"{prefix}_ENABLED",
            route == "REVERSAL",
        )
    )
    info = {
        "route": route,
        "enabled": enabled,
        "armed": False,
        "should_exit": False,
        "current_roi": 0.0,
        "peak_roi": max(_safe_float(peak_roi), 0),
        "floor_roi": 0.0,
        "reason": f"{prefix}_DISABLED",
    }

    if not enabled:
        return info

    avg_entry = _safe_float(avg_entry)
    current_price = _safe_float(current_price)
    leverage = max(
        _safe_float(leverage, getattr(config, "LEVERAGE", 1)),
        1,
    )

    if side not in ("BUY", "SELL") or avg_entry <= 0 or current_price <= 0:
        info["reason"] = f"{prefix}_INVALID_PRICE"
        return info

    if side == "BUY":
        current_roi = (
            (current_price - avg_entry) / avg_entry
        ) * leverage * 100
    else:
        current_roi = (
            (avg_entry - current_price) / avg_entry
        ) * leverage * 100

    peak_roi = max(_safe_float(peak_roi), current_roi, 0)
    trigger_roi = max(
        get_config_float(
            f"{prefix}_TRIGGER_ROI",
            12 if route == "REVERSAL" else 15,
        ),
        0,
    )
    lock_roi = max(
        get_config_float(
            f"{prefix}_LOCK_ROI",
            3 if route == "REVERSAL" else 5,
        ),
        0,
    )
    retrace_pct = min(
        max(
            get_config_float(
                f"{prefix}_RETRACE_PCT",
                50 if route == "REVERSAL" else 45,
            ),
            0,
        ),
        100,
    )
    armed = peak_roi >= trigger_roi
    floor_roi = (
        max(lock_roi, peak_roi * (1 - retrace_pct / 100))
        if armed
        else 0
    )
    should_exit = armed and current_roi <= floor_roi

    info.update({
        "armed": armed,
        "should_exit": should_exit,
        "current_roi": round(float(current_roi), 2),
        "peak_roi": round(float(peak_roi), 2),
        "floor_roi": round(float(floor_roi), 2),
        "trigger_roi": trigger_roi,
        "lock_roi": lock_roi,
        "retrace_pct": retrace_pct,
        "reason": (
            f"{route}_PROFIT_RETRACE_EXIT"
            if should_exit
            else (
                f"{prefix}_ARMED"
                if armed
                else f"{route}_PROFIT_TRIGGER_NOT_REACHED"
            )
        ),
    })
    return info


def evaluate_reversal_profit_protection(
    side,
    avg_entry,
    current_price,
    peak_roi=0,
    leverage=None,
):
    return evaluate_route_profit_protection(
        side,
        avg_entry,
        current_price,
        peak_roi=peak_roi,
        leverage=leverage,
        confirmation_type="REVERSAL",
    )


def evaluate_trend_profit_protection(
    side,
    avg_entry,
    current_price,
    peak_roi=0,
    leverage=None,
):
    return evaluate_route_profit_protection(
        side,
        avg_entry,
        current_price,
        peak_roi=peak_roi,
        leverage=leverage,
        confirmation_type="TREND",
    )


def _refresh_side_decision(side_data):
    trend_ok = bool(side_data.get("trend_following_ok"))
    reversal_ok = bool(side_data.get("reversal_ok"))
    range_reversion_ok = bool((side_data.get("range_reversion") or {}).get("active"))
    side_data["hard_ok"] = trend_ok or reversal_ok or range_reversion_ok

    if trend_ok:
        side_data["confirmation_type"] = "TREND"
        side_data["confidence"] = side_data.get("trend_confidence", 0)
        side_data["uncapped_score_index"] = side_data.get(
            "trend_uncapped_score_index",
            side_data.get("confidence", 0),
        )
    elif reversal_ok:
        side_data["confirmation_type"] = "REVERSAL"
        side_data["confidence"] = side_data.get("reversal_confidence", 0)
        side_data["uncapped_score_index"] = side_data.get(
            "reversal_uncapped_score_index",
            side_data.get("confidence", 0),
        )
    elif range_reversion_ok:
        side_data["confirmation_type"] = "RANGE_REVERSION"
        side_data["confidence"] = (side_data.get("range_reversion") or {}).get(
            "confidence",
            0,
        )
        side_data["uncapped_score_index"] = side_data["confidence"]
    else:
        side_data["confirmation_type"] = "NONE"
        side_data["confidence"] = side_data.get("trend_confidence", 0)
        side_data["uncapped_score_index"] = side_data.get(
            "trend_uncapped_score_index",
            side_data.get("confidence", 0),
        )

    return side_data


def _reversal_warning_context(side_data):
    enabled = bool(getattr(config, "TREND_EXHAUSTION_GUARD_ENABLED", True))
    context = {
        "enabled": enabled,
        "active": False,
        "stage": "NONE",
        "side": side_data.get("side"),
        "items": {},
        "points": 0,
        "required": 0,
        "reasons": [],
    }

    if not enabled:
        context["reasons"].append("TREND_EXHAUSTION_GUARD_DISABLED")
        return context

    reversal_context = side_data.get("reversal_context") or {}
    extra = reversal_context.get("extra_confirmation") or {}
    invalidation = reversal_context.get("invalidation") or {}
    counter_ok = (
        reversal_context.get("counter_trend", {}).get("count", 0) >= 2
    )
    smc_ok = bool(extra.get("items", {}).get("smc"))
    momentum_ok = bool(extra.get("items", {}).get("momentum"))
    recovery_score = _safe_float(invalidation.get("recovery_score", 0))
    pressure_score = _safe_float(invalidation.get("pressure_score", 0))
    recovery_ok = recovery_score >= get_config_float(
        "TREND_EXHAUSTION_MIN_RECOVERY_SCORE",
        2.5
    )
    futures_score = _safe_float(side_data.get("participation_score", 0))
    futures_available = bool(side_data.get("participation_available"))
    futures_support = (
        futures_available and
        futures_score >= get_config_float(
            "TREND_EXHAUSTION_FUTURES_SUPPORT_SCORE",
            0.5
        )
    )
    futures_conflict = (
        futures_available and
        futures_score < get_config_float(
            "FUTURES_CONTEXT_MIN_SIGNAL_SCORE",
            -0.5
        )
    )
    warning_items = {
        "smc": smc_ok,
        "momentum": momentum_ok,
        "recovery": recovery_ok,
        "confirm_score": _safe_float(side_data.get("confirm_score")) >= get_config_float(
            "REVERSAL_MOMENTUM_MIN_CONFIRM_SCORE",
            5
        ),
        "entry_score": _safe_float(side_data.get("entry_score")) >= get_config_float(
            "REVERSAL_MOMENTUM_MIN_ENTRY_SCORE",
            4
        ),
        "quality_score": _safe_float(side_data.get("quality_score")) >= get_config_float(
            "SIGNAL_MIN_QUALITY_SCORE",
            0.25
        ),
        "level": bool(side_data.get("level_ok")),
        "futures": futures_support,
    }
    points = sum(1 for item_ok in warning_items.values() if item_ok)
    required = get_config_float("TREND_EXHAUSTION_MIN_WARNING_POINTS", 4)

    if pressure_score >= get_config_float(
        "REVERSAL_CONFIRMATION_STRONG_PRESSURE_SCORE",
        5.0
    ):
        required = max(
            required,
            get_config_float(
                "TREND_EXHAUSTION_STRONG_PRESSURE_MIN_WARNING_POINTS",
                5
            )
        )

    directional_trigger = (
        momentum_ok or
        recovery_ok or
        (smc_ok and futures_support)
    )
    confidence = _safe_float(side_data.get("reversal_confidence"))
    min_confidence = get_config_float(
        "TREND_EXHAUSTION_MIN_REVERSAL_CONFIDENCE",
        65
    )
    active = counter_ok and points >= required and confidence >= min_confidence

    if (
        getattr(config, "TREND_EXHAUSTION_REQUIRE_DIRECTIONAL_TRIGGER", True)
        and not directional_trigger
    ):
        active = False
        context["reasons"].append("DIRECTIONAL_TRIGGER_MISSING")

    if (
        getattr(config, "TREND_EXHAUSTION_FUTURES_VETO_ENABLED", True)
        and futures_conflict
    ):
        active = False
        context["reasons"].append(
            f"FUTURES_CONFLICT={round(futures_score, 2)}"
        )

    if not counter_ok:
        context["reasons"].append("COUNTER_TREND_CONTEXT_MISSING")
    if points < required:
        context["reasons"].append(f"POINTS={points} < {required}")
    if confidence < min_confidence:
        context["reasons"].append(
            f"CONFIDENCE={round(confidence, 2)} < {min_confidence}"
        )

    if side_data.get("reversal_confirmed"):
        stage = (
            "CONFIRMED_ENTRY"
            if side_data.get("reversal_ok")
            else "CONFIRMED_WARNING_ONLY"
        )
    elif active:
        stage = "WARNING"
    else:
        stage = "NONE"

    context.update({
        "active": active,
        "stage": stage,
        "items": warning_items,
        "points": points,
        "required": required,
        "confidence": round(confidence, 2),
        "min_confidence": min_confidence,
        "pressure_score": round(pressure_score, 2),
        "recovery_score": round(recovery_score, 2),
        "directional_trigger": directional_trigger,
        "futures_available": futures_available,
        "futures_score": round(futures_score, 2),
    })
    return context


def _apply_trend_exhaustion_guard(buy, sell):
    for side_data in (buy, sell):
        warning = _reversal_warning_context(side_data)
        side_data["reversal_warning"] = warning
        side_data.setdefault("reversal_context", {})["stage"] = warning["stage"]
        side_data["trend_exhaustion"] = {
            "enabled": warning["enabled"],
            "blocked": False,
            "opposite_side": None,
            "reason": None,
        }

    if not getattr(config, "TREND_EXHAUSTION_GUARD_ENABLED", True):
        return buy, sell

    for trend_side, opposite_side in ((buy, sell), (sell, buy)):
        warning = opposite_side.get("reversal_warning") or {}

        if not trend_side.get("trend_following_ok") or not warning.get("active"):
            continue

        reason = (
            f"OPPOSITE_{opposite_side.get('side')}_REVERSAL_WARNING "
            f"POINTS={warning.get('points')} REQUIRED={warning.get('required')} "
            f"CONFIDENCE={warning.get('confidence')}"
        )
        trend_side["trend_following_ok"] = False
        trend_side["trend_exhaustion"] = {
            "enabled": True,
            "blocked": True,
            "opposite_side": opposite_side.get("side"),
            "reason": reason,
            "warning": warning,
        }
        _refresh_side_decision(trend_side)

    return buy, sell


def _range_reversion_context(
    side,
    entry_df,
    regime_context,
    confirm_score,
    quality_score,
    level_ok,
    level,
):
    """Mean-reversion fade at a validated range boundary.

    Deliberately scored from its own inputs (level quality + RSI extremity)
    rather than the trend/confirm/entry aggregate score: a sideways regime
    implies that aggregate is not diagnostic for a fade trade, and reusing
    it would conflate two different strategies' confidence numbers.
    """
    enabled = bool(getattr(config, "RANGE_REVERSION_ENABLED", False))
    context = {
        "enabled": enabled,
        "active": False,
        "reasons": [],
        "score": 0,
        "confidence": 0,
    }

    if not enabled:
        context["reasons"].append("RANGE_REVERSION_DISABLED")
        return context

    regime = regime_context.get("regime")

    if regime != "sideways":
        context["reasons"].append(f"REGIME_NOT_SIDEWAYS={regime}")
        return context

    if not level_ok or not level or "level" not in level:
        context["reasons"].append("NO_RANGE_BOUNDARY_LEVEL")
        return context

    max_level_roi = get_config_float("RANGE_REVERSION_MAX_LEVEL_ADVERSE_ROI", 8.0)
    level_adverse_roi = abs(_safe_float(level.get("adverse_roi"), 999))

    if level_adverse_roi > max_level_roi:
        context["reasons"].append(
            f"LEVEL_TOO_FAR ROI={round(level_adverse_roi, 2)} > {max_level_roi}"
        )
        return context

    min_level_score = get_config_float("RANGE_REVERSION_MIN_LEVEL_SCORE", 3.0)
    level_score = _safe_float(level.get("score"))

    if level_score < min_level_score:
        context["reasons"].append(
            f"LEVEL_SCORE_LOW {level_score} < {min_level_score}"
        )
        return context

    entry = latest_closed(entry_df)
    rsi = _safe_float(entry.get("rsi"), 50)
    oversold = get_config_float("RANGE_REVERSION_RSI_OVERSOLD", 32)
    overbought = get_config_float("RANGE_REVERSION_RSI_OVERBOUGHT", 68)
    rsi_extreme = rsi <= oversold if side == "BUY" else rsi >= overbought

    if not rsi_extreme:
        context["reasons"].append(f"RSI_NOT_EXTREME={round(rsi, 1)}")
        return context

    min_quality = get_config_float("RANGE_REVERSION_MIN_QUALITY_SCORE", 0.0)

    if quality_score < min_quality:
        context["reasons"].append(
            f"QUALITY_SCORE_LOW {quality_score} < {min_quality}"
        )
        return context

    min_confirm = get_config_float("RANGE_REVERSION_MIN_CONFIRM_SCORE", -3.0)

    if confirm_score < min_confirm:
        context["reasons"].append(
            f"CONFIRM_SCORE_TOO_WEAK {confirm_score} < {min_confirm}"
        )
        return context

    rsi_edge = (oversold - rsi) if side == "BUY" else (rsi - overbought)
    proximity_component = (
        (max_level_roi - level_adverse_roi) / max(max_level_roi, 0.01)
    ) * 2
    score = max(level_score + proximity_component + max(rsi_edge, 0) / 10, 0)
    max_score = get_config_float("RANGE_REVERSION_CONFIDENCE_MAX_SCORE", 12.0)
    confidence = score_to_confidence(score, max_score)

    context.update({
        "active": True,
        "score": round(score, 2),
        "confidence": confidence,
        "rsi": round(rsi, 2),
        "level_adverse_roi": round(level_adverse_roi, 2),
        "level_score": level_score,
    })
    return context


def _side_signal_score(
    side,
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    participation=None
):
    entry_price = latest_closed(entry_df)["close"]
    level_ok, level = validate_adverse_zone_level(
        side,
        entry_price,
        trend_df,
        confirm_df
    )
    trend_score, trend_ok = _trend_bias_score(side, trend_df)
    confirm_score, confirm_ok, confirm_quality = _confirmation_score(side, confirm_df)
    entry_score, entry_ok, ema_distance, entry_quality = _entry_score(side, entry_df)
    btc_score = _btc_context_score(side, btc_trend, btc_corr, rs)
    futures_price_change = _closed_frame_return_pct(
        confirm_df,
        get_config_float("FUTURES_CONTEXT_LIMIT", 8),
    )
    participation_score = _futures_participation_score(
        side,
        participation,
        futures_price_change,
    )
    futures_ok, futures_gate_reasons = _futures_context_gate(
        participation_score,
        participation
    )
    smc_score, smc_context = _smc_context_score(side, trend_df, confirm_df, entry_df)
    smc_ok, _ = _reversal_smc_check(smc_score, smc_context)
    momentum_score, momentum_context = _reversal_momentum_context(
        side,
        confirm_df,
        entry_df,
        smc_ok
    )
    regime_score, regime_context = _market_regime_score(
        side,
        trend_df,
        confirm_df,
        entry_df
    )
    quality_score = round(
        float(confirm_quality.get("score", 0)) +
        float(entry_quality.get("score", 0)),
        2
    )
    module_gates_ok, module_gate_reasons = _module_gates_check(
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        regime_score
    )
    level_check_disabled = bool(level.get("level_check_disabled")) if level else False
    level_score = 4 if level_ok and not level_check_disabled else 0
    range_reversion = _range_reversion_context(
        side,
        entry_df,
        regime_context,
        confirm_score,
        quality_score,
        level_ok,
        level,
    )

    total = (
        trend_score +
        confirm_score +
        entry_score +
        btc_score +
        level_score +
        smc_score +
        momentum_score +
        participation_score +
        regime_score
    )
    score = max(0, total)
    trend_confidence = score_to_confidence(score)
    trend_uncapped_score_index = score_to_uncapped_index(score)
    reversal_confidence = score_to_confidence(
        score,
        get_config_float("REVERSAL_CONFIDENCE_MAX_SCORE", 34)
    )
    reversal_uncapped_score_index = score_to_uncapped_index(
        score,
        get_config_float("REVERSAL_CONFIDENCE_MAX_SCORE", 34)
    )
    normal_trend_following_ok = (
        trend_ok and
        confirm_ok and
        entry_ok and
        level_ok and
        module_gates_ok and
        futures_ok
    )
    trend_timing_rescue = _trend_timing_rescue_context(
        trend_ok,
        confirm_ok,
        entry_ok,
        level_ok,
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        regime_score,
        trend_confidence,
        confirm_quality,
        entry_quality,
        participation_score,
        participation,
        futures_ok
    )
    continuation_pullback = _continuation_pullback_context(
        side,
        entry_df,
        trend_ok,
        confirm_ok,
        entry_ok,
        level_ok,
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        regime_score,
        trend_confidence,
        entry_quality,
        participation_score,
        participation,
        futures_ok
    )
    trend_following_ok = (
        normal_trend_following_ok or
        trend_timing_rescue.get("active", False) or
        continuation_pullback.get("active", False)
    )
    reversal_ok, reversal_reasons, reversal_context = _reversal_signal_check(
        side,
        trend_df,
        confirm_df,
        entry_df,
        trend_score,
        confirm_score,
        entry_score,
        quality_score,
        smc_score,
        smc_context,
        momentum_context,
        regime_score,
        reversal_confidence,
        confirm_ok,
        entry_ok,
        level_ok
    )
    chart_reversal_ok = reversal_ok
    reversal_futures_confirmation = _reversal_futures_confirmation_context(
        chart_reversal_ok,
        participation_score,
        participation,
        futures_ok,
    )
    reversal_context["futures_confirmation"] = (
        reversal_futures_confirmation
    )

    if not futures_ok:
        reversal_ok = False
        reversal_reasons = list(reversal_reasons) + futures_gate_reasons

    if (
        reversal_futures_confirmation.get("available") and
        reversal_futures_confirmation.get("required") and
        not reversal_futures_confirmation.get("active")
    ):
        reversal_ok = False
        reversal_reasons = (
            list(reversal_reasons) +
            list(reversal_futures_confirmation.get("reasons", []))
        )

    reversal_confirmed = chart_reversal_ok

    if reversal_ok and not getattr(config, "REVERSAL_ENTRY_ENABLED", True):
        reversal_ok = False
        reversal_reasons = list(reversal_reasons) + [
            "REVERSAL_ENTRY_DISABLED_WARNING_ONLY"
        ]

    side_data = {
        "side": side,
        "score": score,
        "trend_confidence": trend_confidence,
        "trend_uncapped_score_index": trend_uncapped_score_index,
        "reversal_confidence": reversal_confidence,
        "reversal_uncapped_score_index": reversal_uncapped_score_index,
        "base_score": trend_score + confirm_score + entry_score + btc_score + level_score,
        "trend_score": trend_score,
        "confirm_score": confirm_score,
        "entry_score": entry_score,
        "btc_score": btc_score,
        "level_score": level_score,
        "smc_score": smc_score,
        "smc_context": smc_context,
        "momentum_score": momentum_score,
        "momentum_context": momentum_context,
        "quality_score": quality_score,
        "confirm_quality": confirm_quality,
        "entry_quality": entry_quality,
        "regime_score": regime_score,
        "regime_context": regime_context,
        "participation_score": participation_score,
        "participation_available": bool(
            participation and participation.get("available")
        ),
        "futures_context_ok": futures_ok,
        "futures_gate_reasons": futures_gate_reasons,
        "trend_following_ok": trend_following_ok,
        "normal_trend_following_ok": normal_trend_following_ok,
        "trend_timing_rescue": trend_timing_rescue,
        "continuation_pullback": continuation_pullback,
        "reversal_ok": reversal_ok,
        "reversal_confirmed": reversal_confirmed,
        "reversal_reasons": reversal_reasons,
        "reversal_context": reversal_context,
        "range_reversion": range_reversion,
        "trend_ok": trend_ok,
        "confirm_ok": confirm_ok,
        "entry_ok": entry_ok,
        "level_ok": level_ok,
        "module_gates_ok": module_gates_ok,
        "module_gate_reasons": module_gate_reasons,
        "level": level,
        "ema_distance": ema_distance,
    }
    return _refresh_side_decision(side_data)


def _signal_threshold(side_data):
    if side_data.get("confirmation_type") == "RANGE_REVERSION":
        return get_config_float("RANGE_REVERSION_SIGNAL_THRESHOLD", 70)

    if side_data.get("confirmation_type") != "REVERSAL":
        return config.LONG_TERM_SIGNAL_THRESHOLD

    momentum = side_data.get("reversal_context", {}).get("momentum", {})

    if momentum.get("ok"):
        return get_config_float(
            "REVERSAL_MOMENTUM_SIGNAL_THRESHOLD",
            get_config_float("REVERSAL_SIGNAL_THRESHOLD", 78)
        )

    return get_config_float(
        "REVERSAL_SIGNAL_THRESHOLD",
        config.LONG_TERM_SIGNAL_THRESHOLD
    )


def _signal_edge(side_data):
    if side_data.get("confirmation_type") == "REVERSAL":
        return get_config_float("REVERSAL_MIN_SIGNAL_EDGE", 0)

    if side_data.get("confirmation_type") == "RANGE_REVERSION":
        return get_config_float("RANGE_REVERSION_MIN_SIGNAL_EDGE", 0)

    return config.LONG_TERM_MIN_SIGNAL_EDGE


def _signal_candidate(side_data):
    return (
        side_data.get("hard_ok")
        and side_data.get("confidence", 0) >= _signal_threshold(side_data)
    )


def _select_signal(buy, sell):
    buy_ok = _signal_candidate(buy)
    sell_ok = _signal_candidate(sell)

    if buy_ok and not sell_ok:
        return "BUY"

    if sell_ok and not buy_ok:
        return "SELL"

    if not buy_ok and not sell_ok:
        return None

    ignore_trend_confidence = bool(
        getattr(config, "REVERSAL_IGNORE_OPPOSITE_TREND_CONFIDENCE", True)
    )

    if ignore_trend_confidence:
        if (
            buy.get("confirmation_type") == "REVERSAL"
            and sell.get("confirmation_type") == "TREND"
        ):
            return "BUY"

        if (
            sell.get("confirmation_type") == "REVERSAL"
            and buy.get("confirmation_type") == "TREND"
        ):
            return "SELL"

    if buy["confidence"] >= sell["confidence"] + _signal_edge(buy):
        return "BUY"

    if sell["confidence"] >= buy["confidence"] + _signal_edge(sell):
        return "SELL"

    return None


def log_signal_analysis(analysis):
    buy = analysis["buy"]
    sell = analysis["sell"]

    log_info(
        f"BUY conf={buy.get('confidence', 0)}% hard={buy.get('hard_ok', False)} "
        f"type={buy.get('confirmation_type', 'NONE')} "
        f"level={buy.get('level_ok', False)} "
        f"gates={buy.get('module_gates_ok', True)} "
        f"quality={buy.get('quality_score', 0)} "
        f"regime={buy.get('regime_context', {}).get('regime', '')}:"
        f"{buy.get('regime_score', 0)} "
        f"smc={buy.get('smc_score', 0)} "
        f"momentum={buy.get('momentum_score', 0)} "
        f"futures={buy.get('participation_score', 0)} "
        f"futures_ok={buy.get('futures_context_ok', True)} | "
        f"SELL conf={sell.get('confidence', 0)}% "
        f"hard={sell.get('hard_ok', False)} "
        f"type={sell.get('confirmation_type', 'NONE')} "
        f"level={sell.get('level_ok', False)} "
        f"gates={sell.get('module_gates_ok', True)} "
        f"quality={sell.get('quality_score', 0)} "
        f"regime={sell.get('regime_context', {}).get('regime', '')}:"
        f"{sell.get('regime_score', 0)} "
        f"smc={sell.get('smc_score', 0)} "
        f"momentum={sell.get('momentum_score', 0)} "
        f"futures={sell.get('participation_score', 0)} "
        f"futures_ok={sell.get('futures_context_ok', True)}"
    )

    for side_data in (buy, sell):
        rescue = side_data.get("trend_timing_rescue") or {}
        pullback = side_data.get("continuation_pullback") or {}
        reversal_futures = (
            (side_data.get("reversal_context") or {}).get(
                "futures_confirmation",
                {},
            )
        )

        if rescue.get("active"):
            log_info(
                f"{side_data.get('side')} TREND TIMING RESCUE ACTIVE | "
                f"MISSED={rescue.get('missed_module')} | "
                f"CONFIDENCE={side_data.get('trend_confidence')} | "
                f"FUTURES={rescue.get('futures_score')}"
            )
        elif (
            rescue.get("reason") ==
            "TREND_TIMING_RESCUE_AWAITING_FUTURES"
        ):
            log_info(
                f"{side_data.get('side')} TREND TIMING RESCUE WAITING | "
                f"MISSED={rescue.get('missed_module')} | "
                f"REASON={rescue.get('reason')}"
            )

        if pullback.get("active"):
            log_info(
                f"{side_data.get('side')} CONTINUATION PULLBACK ACTIVE | "
                f"CONFIDENCE={side_data.get('trend_confidence')} | "
                f"EMA20_DISTANCE_ATR={pullback.get('ema20_distance_atr')} | "
                f"FUTURES={pullback.get('futures_score')}"
            )
        elif (
            pullback.get("reason") ==
            "CONTINUATION_PULLBACK_AWAITING_FUTURES"
        ):
            log_info(
                f"{side_data.get('side')} CONTINUATION PULLBACK WAITING | "
                f"EMA20_DISTANCE_ATR={pullback.get('ema20_distance_atr')} | "
                f"REASON={pullback.get('reason')}"
            )

        if (
            reversal_futures.get("reason") ==
            "REVERSAL_FUTURES_AWAITING_CONTEXT"
        ):
            log_info(
                f"{side_data.get('side')} REVERSAL FUTURES WAITING | "
                f"MIN_SCORE={reversal_futures.get('minimum')}"
            )
        elif (
            reversal_futures.get("eligible") and
            reversal_futures.get("available") and
            not reversal_futures.get("active")
        ):
            log_warning(
                f"{side_data.get('side')} REVERSAL FUTURES BLOCKED | "
                f"SCORE={reversal_futures.get('score')} | "
                f"MIN={reversal_futures.get('minimum')}"
            )

        exhaustion = side_data.get("trend_exhaustion") or {}

        if exhaustion.get("blocked"):
            log_warning(
                f"{side_data.get('side')} TREND ENTRY BLOCKED | "
                f"{exhaustion.get('reason')}"
            )

        warning = side_data.get("reversal_warning") or {}

        if warning.get("active") and not side_data.get("reversal_ok"):
            log_info(
                f"{side_data.get('side')} REVERSAL WARNING | "
                f"STAGE={warning.get('stage')} "
                f"POINTS={warning.get('points')}/{warning.get('required')} "
                f"CONFIDENCE={warning.get('confidence')} "
                f"PRESSURE={warning.get('pressure_score')} "
                f"RECOVERY={warning.get('recovery_score')}"
            )

    if not buy.get("futures_context_ok", True):
        log_warning(
            "BUY FUTURES CONTEXT BLOCKED | " +
            "; ".join(buy.get("futures_gate_reasons", []))
        )

    if not sell.get("futures_context_ok", True):
        log_warning(
            "SELL FUTURES CONTEXT BLOCKED | " +
            "; ".join(sell.get("futures_gate_reasons", []))
        )

    if buy.get("reversal_ok"):
        log_info(
            "BUY REVERSAL CONFIRMED | " +
            f"counter={buy.get('reversal_context', {}).get('counter_trend', {}).get('count')} "
            f"smc={buy.get('smc_score', 0)} "
            f"momentum={buy.get('momentum_score', 0)} "
            f"conf={buy.get('confidence', 0)}"
        )

    if sell.get("reversal_ok"):
        log_info(
            "SELL REVERSAL CONFIRMED | " +
            f"counter={sell.get('reversal_context', {}).get('counter_trend', {}).get('count')} "
            f"smc={sell.get('smc_score', 0)} "
            f"momentum={sell.get('momentum_score', 0)} "
            f"conf={sell.get('confidence', 0)}"
        )

    min_blocked_conf = get_config_float("REVERSAL_LOG_BLOCKED_MIN_CONFIDENCE", 55)

    if (
        getattr(config, "REVERSAL_MODE_ENABLED", True)
        and not buy.get("reversal_ok")
        and buy.get("reversal_confidence", 0) >= min_blocked_conf
    ):
        momentum = buy.get("momentum_context", {})
        log_info(
            "BUY REVERSAL BLOCKED | " +
            "; ".join(buy.get("reversal_reasons", [])) +
            f" | momentum={momentum.get('score', 0)} ok={momentum.get('ok')}"
        )

    if (
        getattr(config, "REVERSAL_MODE_ENABLED", True)
        and not sell.get("reversal_ok")
        and sell.get("reversal_confidence", 0) >= min_blocked_conf
    ):
        momentum = sell.get("momentum_context", {})
        log_info(
            "SELL REVERSAL BLOCKED | " +
            "; ".join(sell.get("reversal_reasons", [])) +
            f" | momentum={momentum.get('score', 0)} ok={momentum.get('ok')}"
        )

    if not buy.get("module_gates_ok", True) and not buy.get("reversal_ok"):
        log_warning(
            "BUY MODULE GATE BLOCKED | " +
            "; ".join(buy.get("module_gate_reasons", []))
        )

    if not sell.get("module_gates_ok", True) and not sell.get("reversal_ok"):
        log_warning(
            "SELL MODULE GATE BLOCKED | " +
            "; ".join(sell.get("module_gate_reasons", []))
        )

    if buy.get("level_ok"):
        log_info(
            f"BUY support {buy.get('level', {}).get('level')} "
            f"ROI={buy.get('level', {}).get('adverse_roi')}% "
            f"SRC={buy.get('level', {}).get('source')}"
        )
    else:
        log_warning(
            f"BUY BLOCKED | {buy.get('level', {}).get('reason', 'NO DETAILS')}"
        )

    if sell.get("level_ok"):
        log_info(
            f"SELL resistance {sell.get('level', {}).get('level')} "
            f"ROI={sell.get('level', {}).get('adverse_roi')}% "
            f"SRC={sell.get('level', {}).get('source')}"
        )
    else:
        log_warning(
            f"SELL BLOCKED | {sell.get('level', {}).get('reason', 'NO DETAILS')}"
        )

    if analysis["signal"]:
        signal_details = analysis[analysis["signal"].lower()]
        log_info(
            f"FINAL LONG-TERM {analysis['signal']} "
            f"TYPE={signal_details.get('confirmation_type', 'NONE')} "
            f"CONFIDENCE: "
            f"{signal_details.get('confidence', 0)}"
        )


def analyze_signal(
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    participation=None,
    log_details=True
):
    try:
        buy = _side_signal_score(
            "BUY",
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            participation=participation
        )
        sell = _side_signal_score(
            "SELL",
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            participation=participation
        )
        buy, sell = _apply_trend_exhaustion_guard(buy, sell)
        signal = _select_signal(buy, sell)
        best = buy if buy["confidence"] >= sell["confidence"] else sell
        exhaustion_blocks = [
            side_data.get("trend_exhaustion", {}).get("reason")
            for side_data in (buy, sell)
            if side_data.get("trend_exhaustion", {}).get("blocked")
        ]
        analysis = {
            "buy": buy,
            "sell": sell,
            "signal": signal,
            "best_side": best["side"],
            "best_confidence": best["confidence"],
            "threshold": config.LONG_TERM_SIGNAL_THRESHOLD,
            "min_edge": config.LONG_TERM_MIN_SIGNAL_EDGE,
            "participation_available": bool(
                participation and participation.get("available")
            ),
            "trend_exhaustion_blocked": bool(exhaustion_blocks),
            "trend_exhaustion_reasons": exhaustion_blocks,
        }

        if log_details:
            log_signal_analysis(analysis)

        return analysis

    except Exception as e:
        log_error(f"STRATEGY ERROR: {e}")
    return {
            "buy": {},
            "sell": {},
            "signal": None,
            "best_side": None,
            "best_confidence": 0,
            "threshold": config.LONG_TERM_SIGNAL_THRESHOLD,
            "min_edge": config.LONG_TERM_MIN_SIGNAL_EDGE,
            "participation_available": False,
            "error": str(e),
    }


def _analysis_signature_value(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)

    return None if value != value else value


def _analysis_frame_signature(df):
    if df is None or len(df) == 0:
        return None

    signature = [len(df)]

    for column in (
        "time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
    ):
        if column not in df.columns:
            continue

        signature.append((
            column,
            tuple(
                _analysis_signature_value(value)
                for value in df[column].tail(2)
            ),
        ))

    return tuple(signature)


def clear_signal_analysis_cache():
    with _signal_analysis_cache_lock:
        _signal_analysis_cache.clear()


def analyze_signal_cached(
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    participation=None,
    log_details=False,
    cache_namespace=None,
):
    cache_enabled = bool(
        getattr(config, "SIGNAL_ANALYSIS_CACHE_ENABLED", True)
    )

    if not cache_enabled or participation is not None or log_details:
        return analyze_signal(
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            participation=participation,
            log_details=log_details,
        )

    key = (
        str(cache_namespace or ""),
        _analysis_frame_signature(trend_df),
        _analysis_frame_signature(confirm_df),
        _analysis_frame_signature(entry_df),
        str(btc_trend),
        _analysis_signature_value(btc_corr),
        _analysis_signature_value(rs),
    )

    with _signal_analysis_cache_lock:
        cached = _signal_analysis_cache.get(key)

        if cached is not None:
            _signal_analysis_cache.move_to_end(key)
            return deepcopy(cached)

    analysis = analyze_signal(
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        participation=None,
        log_details=False,
    )
    max_items = max(
        int(getattr(config, "SIGNAL_ANALYSIS_CACHE_MAX_ITEMS", 1200)),
        1,
    )

    with _signal_analysis_cache_lock:
        _signal_analysis_cache[key] = deepcopy(analysis)
        _signal_analysis_cache.move_to_end(key)

        while len(_signal_analysis_cache) > max_items:
            _signal_analysis_cache.popitem(last=False)

    return analysis


def should_fetch_futures_context(analysis):
    if not config.FUTURES_CONTEXT_ENABLED:
        return False

    if analysis.get("best_confidence", 0) < config.FUTURES_CONTEXT_MIN_CONFIDENCE:
        return False

    buy = analysis.get("buy", {})
    sell = analysis.get("sell", {})

    if any(
        (side_data.get("continuation_pullback") or {}).get("eligible")
        and not (side_data.get("continuation_pullback") or {}).get(
            "participation_available"
        )
        for side_data in (buy, sell)
    ):
        return True

    if any(
        (
            (side_data.get("reversal_context") or {}).get(
                "futures_confirmation",
                {},
            )
        ).get("eligible")
        and not (
            (side_data.get("reversal_context") or {}).get(
                "futures_confirmation",
                {},
            )
        ).get("available")
        for side_data in (buy, sell)
    ):
        return True

    if any(
        (side_data.get("trend_timing_rescue") or {}).get("eligible")
        and not (side_data.get("trend_timing_rescue") or {}).get(
            "participation_available"
        )
        for side_data in (buy, sell)
    ):
        return True

    if any(
        (side_data.get("reversal_warning") or {}).get("active")
        for side_data in (buy, sell)
    ):
        return True

    best_key = (analysis.get("best_side") or "").lower()
    best = analysis.get(best_key, {}) if best_key in ("buy", "sell") else {}

    if best:
        momentum = best.get("momentum_context", {})
        return bool(
            best.get("level_ok") and
            (
                best.get("hard_ok") or
                (
                    best.get("reversal_confidence", 0) >=
                    get_config_float("FUTURES_CONTEXT_MIN_CONFIDENCE", 60)
                    and (
                        best.get("reversal_ok") or
                        momentum.get("ok")
                    )
                )
            ) and
            (
                best.get("trend_ok") or
                best.get("reversal_context", {}).get("counter_trend", {}).get("count", 0) >= 2
            ) and
            (
                best.get("confirm_ok") or
                momentum.get("ok")
            ) and
            (
                best.get("entry_ok") or
                momentum.get("ok")
            )
        )

    return bool(
        (buy.get("level_ok") and buy.get("hard_ok")) or
        (sell.get("level_ok") and sell.get("hard_ok"))
    )


def futures_context_priority(analysis):
    signal = analysis.get("signal")
    priorities = []

    for side_data in (analysis.get("buy", {}), analysis.get("sell", {})):
        if not side_data:
            continue

        priority = _safe_float(
            side_data.get("confidence"),
            _safe_float(
                side_data.get("trend_confidence"),
                analysis.get("best_confidence", 0),
            ),
        )
        priority += (
            _safe_float(side_data.get("quality_score")) *
            get_config_float("SIGNAL_RANKING_QUALITY_WEIGHT", 1.5)
        )
        priority += (
            _safe_float(side_data.get("smc_score")) *
            get_config_float("SIGNAL_RANKING_SMC_WEIGHT", 1.0)
        )
        priority += (
            _safe_float(side_data.get("regime_score")) *
            get_config_float("SIGNAL_RANKING_REGIME_WEIGHT", 1.0)
        )

        if (side_data.get("continuation_pullback") or {}).get("eligible"):
            priority += get_config_float(
                "FUTURES_CONTEXT_PRIORITY_PULLBACK_BONUS",
                8,
            )

        if (side_data.get("trend_timing_rescue") or {}).get("eligible"):
            priority += get_config_float(
                "FUTURES_CONTEXT_PRIORITY_RESCUE_BONUS",
                5,
            )

        reversal_futures = (
            (side_data.get("reversal_context") or {}).get(
                "futures_confirmation",
                {},
            )
        )

        if reversal_futures.get("eligible"):
            priority += get_config_float(
                "FUTURES_CONTEXT_PRIORITY_REVERSAL_BONUS",
                6,
            )

        if signal == side_data.get("side"):
            priority += get_config_float(
                "FUTURES_CONTEXT_PRIORITY_SIGNAL_BONUS",
                3,
            )

        priorities.append(priority)

    return round(max(priorities, default=0), 2)


def check_signal(trend_df, confirm_df, entry_df, btc_trend, btc_corr, rs):
    return analyze_signal(
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        log_details=True
    )["signal"]
