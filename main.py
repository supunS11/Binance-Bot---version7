import logging
import os
import threading
import time
from datetime import datetime, timedelta

import config

from binance.enums import SIDE_BUY, SIDE_SELL

from exchange import (
    get_klines,
    get_balance,
    get_margin_balance,
    place_market_order,
    close_position_market,
    place_tp_sl,
    get_open_position_details,
    get_open_position_detail_rows,
    get_open_position_counts,
    get_supported_symbols,
    get_futures_participation,
    get_mark_price,
    get_open_take_profit_info,
    get_open_stop_loss_info,
    place_stop_loss_only,
    place_close_position_protection,
    set_margin_type,
    setup_leverage,
    get_entry_price,
    validate_min_notional,
    cancel_open_protection_orders,
    cancel_algo_order,
    get_price_precision,
    get_private_rest_backoff_remaining
)

from indicators import apply_indicators
from strategy import (
    analyze_signal,
    analyze_signal_cached,
    evaluate_route_early_invalidation,
    evaluate_route_profit_protection,
    futures_context_priority,
    log_signal_analysis,
    should_fetch_futures_context,
    validate_live_entry_guard,
    validate_adverse_zone_level,
    validate_structure_take_profit,
    validate_entry_profit_room,
    validate_dca_structure_level,
    validate_dca_continuation_guard
)
from risk_management import calculate_position_size
from signal_journal import append_signal_journal
from llm_service import (
    apply_llm_filter,
    begin_llm_scan_budget,
    prefetch_llm_candidate_reviews,
)
from news_service import apply_news_filter
from telegram_service import (
    send_order_opened_message,
    send_dca_filled_message,
    send_tp_failure_message,
    send_telegram_message
)
from trade_state import (
    apply_multi_tp_protection_state,
    clear_dca_reservation,
    create_position_state,
    get_position_state,
    has_active_dca_reservation,
    load_trade_state,
    prune_closed_positions,
    record_dca_fill,
    reserve_dca_level,
    update_position_runtime_fields,
    update_position_tp_status,
    upsert_position_state
)
from logger import log_info, log_warning, log_error
from multi_tp import (
    RUNNER_ACTIVE,
    RUNNER_PENDING,
    TP1_PENDING,
    calculate_runner_stop,
    extract_order_id,
    roi_to_price,
    tp1_fill_confirmed,
    tp1_trigger_reached,
)


trade_times = {}
_dca_locks = {}
_dca_locks_guard = threading.Lock()
shutdown_event = threading.Event()
target_margin_stop_lock = threading.Lock()


def wait_for_next_scan(reason="SCAN_COMPLETE"):
    wait_seconds = max(float(config.SCAN_SLEEP_SECONDS), 0)
    heartbeat_seconds = max(
        float(getattr(config, "SCAN_WAIT_HEARTBEAT_SECONDS", 60)),
        1,
    )
    deadline = time.monotonic() + wait_seconds
    next_scan_at = datetime.now() + timedelta(seconds=wait_seconds)
    next_scan_label = next_scan_at.isoformat(timespec="seconds")
    log_info(
        f"Waiting next scan | REASON={reason} | "
        f"WAIT_SECONDS={round(wait_seconds, 1)} | "
        f"NEXT_SCAN_AT={next_scan_label}"
    )

    while not shutdown_event.is_set():
        remaining = deadline - time.monotonic()

        if remaining <= 0:
            log_info("Next scan wait complete | starting scan now")
            return True

        if shutdown_event.wait(min(remaining, heartbeat_seconds)):
            return False

        remaining = max(deadline - time.monotonic(), 0)

        if remaining > 0:
            log_info(
                f"Next scan heartbeat | "
                f"REMAINING_SECONDS={round(remaining, 1)} | "
                f"NEXT_SCAN_AT={next_scan_label}"
            )

    return False


def get_dca_lock(symbol):
    with _dca_locks_guard:
        if symbol not in _dca_locks:
            _dca_locks[symbol] = threading.Lock()

        return _dca_locks[symbol]


def get_scan_symbols():
    symbols = list(dict.fromkeys(config.SYMBOLS))

    if config.MAX_SCAN_SYMBOLS > 0:
        symbols = symbols[:config.MAX_SCAN_SYMBOLS]

    supported_symbols = get_supported_symbols()

    if not supported_symbols:
        return symbols

    scan_symbols = [
        symbol
        for symbol in symbols
        if symbol in supported_symbols
    ]

    skipped = len(symbols) - len(scan_symbols)

    if skipped > 0:
        log_warning(f"Skipped {skipped} unsupported symbols from scan list")

    return scan_symbols


def log_closed_trades(open_positions):
    for symbol in list(trade_times):
        if symbol in open_positions:
            continue

        exit_time = datetime.now()
        entry_time = trade_times[symbol]["entry_time"]
        duration = exit_time - entry_time

        log_info(
            f"*** {symbol} TRADE CLOSED *** | "
            f"ENTRY: {entry_time} | "
            f"EXIT: {exit_time} | "
            f"DURATION: {duration}"
        )

        del trade_times[symbol]


def prune_and_cleanup_closed_positions(trade_state, open_positions):
    tracked_symbols = set(trade_state.get("positions", {}))
    closed_symbols = sorted(tracked_symbols - set(open_positions or {}))
    cleanup_failed = set()

    for symbol in closed_symbols:
        if not cancel_open_protection_orders(symbol):
            cleanup_failed.add(symbol)
            log_warning(
                f"{symbol} closed-position protection cleanup incomplete"
            )

    effective_open_positions = dict(open_positions or {})

    for symbol in cleanup_failed:
        effective_open_positions[symbol] = 0

    return prune_closed_positions(trade_state, effective_open_positions)


def get_cached_btc_context():
    btc_df = get_klines("BTCUSDT", config.TREND_TIMEFRAME)

    if btc_df is None or len(btc_df) < 220:
        return None, "NEUTRAL"

    btc_df = apply_indicators(btc_df)

    if btc_df is None:
        return None, "NEUTRAL"

    btc = btc_df.iloc[-2]

    if btc["ema50"] > btc["ema200"]:
        return btc_df, "BULLISH"

    if btc["ema50"] < btc["ema200"]:
        return btc_df, "BEARISH"

    return btc_df, "NEUTRAL"


def calculate_btc_context(symbol, trend_df, btc_df):
    if symbol == "BTCUSDT":
        return 1.0, 0

    if trend_df is None or btc_df is None:
        return 0, 0

    try:
        coin_close = trend_df["close"].iloc[:-1].tail(100).reset_index(drop=True)
        btc_close = btc_df["close"].iloc[:-1].tail(100).reset_index(drop=True)
        length = min(len(coin_close), len(btc_close))

        if length < 20:
            btc_corr = 0
        else:
            coin_ret = coin_close.tail(length).pct_change().dropna()
            btc_ret = btc_close.tail(length).pct_change().dropna()
            btc_corr = coin_ret.corr(btc_ret)

            if btc_corr != btc_corr:
                btc_corr = 0

        if length < 10:
            rs = 0
        else:
            coin_tail = coin_close.tail(length)
            btc_tail = btc_close.tail(length)
            coin_r = (
                (coin_tail.iloc[-1] - coin_tail.iloc[-10]) /
                coin_tail.iloc[-10]
            ) * 100
            btc_r = (
                (btc_tail.iloc[-1] - btc_tail.iloc[-10]) /
                btc_tail.iloc[-10]
            ) * 100
            rs = coin_r - btc_r

        return round(float(btc_corr), 2), round(float(rs), 2)

    except Exception as e:
        log_error(f"{symbol} BTC context error: {e}")
        return 0, 0


def get_open_position_amounts(position_details):
    return {
        symbol: item["amount"]
        for symbol, item in (position_details or {}).items()
    }


def get_signal_frames(symbol, btc_trend_df):
    trend_indicators_ready = (
        symbol == "BTCUSDT" and btc_trend_df is not None
    )

    if trend_indicators_ready:
        trend_df = btc_trend_df
    else:
        trend_df = get_klines(symbol, config.TREND_TIMEFRAME)

    confirm_df = get_klines(symbol, config.CONFIRMATION_TIMEFRAME)
    entry_df = get_klines(symbol, config.ENTRY_TIMEFRAME)

    if trend_df is None or confirm_df is None or entry_df is None:
        return None, None, None

    if len(trend_df) < 220 or len(confirm_df) < 220 or len(entry_df) < 220:
        return None, None, None

    if not trend_indicators_ready:
        trend_df = apply_indicators(trend_df)

    confirm_df = apply_indicators(confirm_df)
    entry_df = apply_indicators(entry_df)

    if trend_df is None or confirm_df is None or entry_df is None:
        return None, None, None

    return trend_df, confirm_df, entry_df


def get_adverse_reversal_frame(symbol, trend_df):
    safety_timeframe = str(
        getattr(config, "ADVERSE_REVERSAL_TIMEFRAME", "1d")
    ).strip()
    trend_timeframe = str(config.TREND_TIMEFRAME).strip()

    if not safety_timeframe or safety_timeframe == trend_timeframe:
        return trend_df

    safety_df = get_klines(symbol, safety_timeframe)

    if safety_df is None or len(safety_df) < 220:
        return None

    return apply_indicators(safety_df)


def check_live_entry_guard(
    symbol,
    side,
    current_price,
    mark_price=None,
    require_both_override=None
):
    if not config.LIVE_ENTRY_CONFIRMATION_ENABLED:
        return True, current_price, {"reason": "LIVE_ENTRY_GUARD_DISABLED"}

    fast_guard_raw = get_klines(
        symbol,
        config.LIVE_ENTRY_FAST_TIMEFRAME,
        config.LIVE_ENTRY_KLINE_LIMIT
    )
    slow_guard_raw = get_klines(
        symbol,
        config.LIVE_ENTRY_SLOW_TIMEFRAME,
        config.LIVE_ENTRY_KLINE_LIMIT
    )

    def prepare_guard_frame(raw_df):
        if raw_df is None:
            return None

        enriched_df = apply_indicators(raw_df)
        min_rows = max(int(config.LIVE_ENTRY_STRUCTURE_LOOKBACK) + 3, 1)

        if enriched_df is None or len(enriched_df) < min_rows:
            return None

        return enriched_df

    fast_guard_df = prepare_guard_frame(fast_guard_raw)
    slow_guard_df = prepare_guard_frame(slow_guard_raw)

    if mark_price is None:
        mark_price = get_mark_price(symbol)

    if mark_price is not None:
        current_price = mark_price

    guard_ok, guard_info = validate_live_entry_guard(
        side,
        fast_guard_df,
        slow_guard_df,
        mark_price,
        require_both_override=require_both_override
    )

    return guard_ok, current_price, guard_info


def log_live_guard_block(symbol, guard_info):
    reason = guard_info.get("reason")
    fast = guard_info.get("fast", {})
    slow = guard_info.get("slow", {})
    log_warning(
        f"{symbol} LIVE ENTRY BLOCKED | {reason} | "
        f"FAST={fast.get('label')} "
        f"SB={fast.get('structure_break')} "
        f"REV={fast.get('opposite_reversal')} "
        f"EMA_WRONG={fast.get('ema_wrong_side')} "
        f"CHASE={fast.get('ema_chase_atr')}ATR "
        f"SUPPORT={fast.get('support_score')} | "
        f"SLOW={slow.get('label')} "
        f"SB={slow.get('structure_break')} "
        f"REV={slow.get('opposite_reversal')} "
        f"EMA_WRONG={slow.get('ema_wrong_side')} "
        f"CHASE={slow.get('ema_chase_atr')}ATR "
        f"SUPPORT={slow.get('support_score')}"
    )


def log_profit_room_ok(symbol, side, room_info, prefix=""):
    level = room_info.get("raw_level")

    if level is None:
        log_info(f"{symbol} {prefix}PROFIT ROOM OK | {room_info.get('reason')}")
        return

    label = "RESISTANCE" if side == "BUY" else "SUPPORT"
    log_info(
        f"{symbol} {prefix}{label} ROOM OK | "
        f"LEVEL={level} | "
        f"TARGET={room_info.get('target_price')} | "
        f"ROOM_ROI={room_info.get('target_roi')}% | "
        f"SRC={room_info.get('source')}"
    )


def log_active_dca_config():
    if not config.DCA_ENABLED:
        log_info("DCA disabled")
        return

    max_levels = min(
        config.DCA_MAX_ORDERS,
        len(config.DCA_TRIGGER_ROIS),
        len(config.DCA_MARGIN_PCTS)
    )
    levels = [
        f"L{index + 1}:ROI={config.DCA_TRIGGER_ROIS[index]}%,"
        f"MARGIN={config.DCA_MARGIN_PCTS[index]}%"
        for index in range(max_levels)
    ]

    log_info(
        "DCA ROI ladder active | "
        f"INITIAL_MARGIN={config.DCA_INITIAL_MARGIN_PCT}% | "
        f"LEVELS={' | '.join(levels) if levels else 'NONE'} | "
        f"REPRICE_TP={config.DCA_REPRICE_TP_AFTER_FILL} | "
        f"TP_MODE={config.DCA_TP_MODE} | "
        f"TP_ROI={config.DCA_TP_ROI}% | "
        f"WEBSOCKET={config.DCA_WEBSOCKET_ENABLED}"
    )


def log_dca_structure_level(symbol, side, level_info):
    label = "SUPPORT" if side == "BUY" else "RESISTANCE"
    log_info(
        f"{symbol} DCA {label} LEVEL OK | "
        f"KIND={level_info.get('kind')} | "
        f"LEVEL={level_info.get('level')} | "
        f"ZONE={level_info.get('zone_low')}..{level_info.get('zone_high')} | "
        f"DIST_ROI={level_info.get('distance_roi')}% | "
        f"SCORE={level_info.get('score')} | "
        f"SRC={level_info.get('source')} | "
        f"REACTION={level_info.get('reaction')}"
    )


def get_initial_trade_margin():
    if not config.DCA_ENABLED:
        return config.MARGIN_PER_TRADE

    pct = max(float(config.DCA_INITIAL_MARGIN_PCT), 0)
    return round(config.MARGIN_PER_TRADE * pct / 100, 8)


def get_dca_order_margin(dca_count):
    if not config.DCA_ENABLED:
        return 0

    if dca_count >= config.DCA_MAX_ORDERS:
        return 0

    if dca_count >= len(config.DCA_MARGIN_PCTS):
        return 0

    pct = max(float(config.DCA_MARGIN_PCTS[dca_count]), 0)
    return round(config.MARGIN_PER_TRADE * pct / 100, 8)


def get_remaining_dca_margin(dca_count):
    if not config.DCA_ENABLED:
        return 0

    max_orders = min(config.DCA_MAX_ORDERS, len(config.DCA_MARGIN_PCTS))

    if dca_count >= max_orders:
        return 0

    total_pct = sum(
        max(float(pct), 0)
        for pct in config.DCA_MARGIN_PCTS[dca_count:max_orders]
    )
    return round(config.MARGIN_PER_TRADE * total_pct / 100, 8)


def get_remaining_dca_order_count(dca_count):
    max_orders = min(config.DCA_MAX_ORDERS, len(config.DCA_MARGIN_PCTS))
    return max(max_orders - dca_count, 0)


def get_dca_trigger_roi(dca_count):
    if dca_count >= config.DCA_MAX_ORDERS:
        return None

    if dca_count >= len(config.DCA_TRIGGER_ROIS):
        return None

    return float(config.DCA_TRIGGER_ROIS[dca_count])


def get_position_adverse_roi(side, avg_entry, current_price):
    if avg_entry <= 0 or current_price <= 0:
        return 0

    if side == "BUY":
        return round(((avg_entry - current_price) / avg_entry) * config.LEVERAGE * 100, 2)

    return round(((current_price - avg_entry) / avg_entry) * config.LEVERAGE * 100, 2)


def get_dca_trigger_entry(position_state, avg_entry):
    for key in ("initial_entry", "reference_price", "avg_entry"):
        try:
            value = float(position_state.get(key) or 0)
        except Exception:
            value = 0

        if value > 0:
            return value

    return avg_entry


def get_dca_price_gap_roi(side, anchor_price, current_price):
    if anchor_price <= 0 or current_price <= 0:
        return 0

    if side == "BUY":
        return round(((anchor_price - current_price) / anchor_price) * config.LEVERAGE * 100, 2)

    return round(((current_price - anchor_price) / anchor_price) * config.LEVERAGE * 100, 2)


def seconds_since(timestamp):
    if not timestamp:
        return None

    try:
        return (datetime.now() - datetime.fromisoformat(timestamp)).total_seconds()
    except Exception:
        return None


def adopt_existing_position_state(state, symbol, position_detail):
    if not config.DCA_MANAGE_EXISTING_POSITIONS:
        return None

    entry_price = float(position_detail.get("entry_price", 0) or 0)

    if entry_price <= 0:
        entry_price = float(position_detail.get("mark_price", 0) or 0)

    if entry_price <= 0:
        log_warning(f"{symbol} existing position not adopted | missing entry price")
        return None

    item = create_position_state(
        symbol,
        position_detail["side"],
        entry_price,
        abs(float(position_detail.get("amount", 0))),
        config.MARGIN_PER_TRADE,
        0,
        entry_price,
        {"source": "ADOPTED_EXISTING_POSITION"}
    )
    item["adopted_existing"] = True
    upsert_position_state(state, symbol, item)
    log_warning(f"{symbol} existing position adopted into DCA state")
    return item


def get_updated_position_after_fill(symbol, old_avg, old_quantity, fill_price, fill_quantity):
    details = get_open_position_details(symbol)
    position_detail = (details or {}).get(symbol)

    if position_detail:
        return (
            float(position_detail.get("entry_price", 0) or fill_price),
            abs(float(position_detail.get("amount", 0))),
            position_detail
        )

    total_quantity = old_quantity + fill_quantity

    if total_quantity <= 0:
        return fill_price, fill_quantity, None

    avg_entry = (
        (old_avg * old_quantity) +
        (fill_price * fill_quantity)
    ) / total_quantity

    return avg_entry, total_quantity, None


def place_tp_sl_with_recovery(
    symbol,
    side,
    entry_price,
    quantity,
    confirm_df,
    structure_tp=None,
    roi_override=None,
    roi_mode_label=None,
    signal_type=None,
    context_label="ENTRY",
    enable_multi_tp=False,
    position_side=None,
    return_details=True
):
    attempts = max(int(config.TP_ORDER_RETRY_ATTEMPTS), 1)
    last_result = {}

    for attempt in range(1, attempts + 1):
        result = place_tp_sl(
            symbol,
            side,
            entry_price,
            quantity,
            confirm_df,
            structure_tp=structure_tp,
            roi_override=roi_override,
            roi_mode_label=roi_mode_label,
            signal_type=signal_type,
            enable_multi_tp=enable_multi_tp,
            position_side=position_side,
            return_details=True
        )
        last_result = result or {}

        if last_result.get("ok"):
            if attempt > 1:
                log_info(
                    f"{symbol} TP recovery succeeded | "
                    f"CONTEXT={context_label} | ATTEMPT={attempt}"
                )

            return last_result if return_details else True

        log_warning(
            f"{symbol} TP placement failed | "
            f"CONTEXT={context_label} | ATTEMPT={attempt}/{attempts} | "
            f"MODE={last_result.get('tp_mode')}"
        )

        if last_result.get("protection_cleanup_failed"):
            log_error(
                f"{symbol} TP recovery stopped | CONTEXT={context_label} | "
                "previous order cleanup was not confirmed"
            )
            break

        if attempt < attempts and config.TP_ORDER_RETRY_DELAY_SECONDS > 0:
            time.sleep(config.TP_ORDER_RETRY_DELAY_SECONDS)

    if (
        config.TP_FAILURE_FALLBACK_ROI_ENABLED
        and roi_override is None
        and not last_result.get("protection_cleanup_failed")
    ):
        fallback_roi = config.STRUCTURE_TP_FALLBACK_ROI
        log_warning(
            f"{symbol} TP recovery fallback ROI | "
            f"CONTEXT={context_label} | ROI={fallback_roi}%"
        )
        fallback_result = place_tp_sl(
            symbol,
            side,
            entry_price,
            quantity,
            confirm_df,
            structure_tp=None,
            roi_override=fallback_roi,
            roi_mode_label=f"TP_RECOVERY_ROI_{fallback_roi}%",
            signal_type=signal_type,
            enable_multi_tp=enable_multi_tp,
            position_side=position_side,
            return_details=True
        )
        last_result = fallback_result or last_result

        if last_result.get("ok"):
            return last_result if return_details else True

    send_tp_failure_message(
        symbol,
        side,
        context_label,
        entry_price,
        quantity,
        last_result
    )
    return last_result if return_details else False


def _manage_dca_position_legacy(symbol, state, position_detail, btc_trend_df, btc_trend):
    if shutdown_event.is_set():
        log_warning(f"{symbol} DCA skipped | bot shutdown requested")
        return

    if not config.DCA_ENABLED:
        log_warning(f"{symbol} already has open position")
        return

    position_state = get_position_state(state, symbol)

    if not position_state:
        position_state = adopt_existing_position_state(
            state,
            symbol,
            position_detail
        )

    if not position_state or not position_state.get("managed_by_bot"):
        log_warning(f"{symbol} open position is not bot-managed; DCA skipped")
        return

    side = position_state.get("side") or position_detail.get("side")

    if side not in ("BUY", "SELL"):
        log_warning(f"{symbol} DCA skipped | invalid side in state")
        return

    live_side = position_detail.get("side")

    if live_side and live_side != side:
        log_warning(
            f"{symbol} DCA skipped | state side {side} != live side {live_side}"
        )
        return

    dca_count = int(position_state.get("dca_count", 0) or 0)
    dca_margin = get_dca_order_margin(dca_count)
    trigger_roi = get_dca_trigger_roi(dca_count)

    if dca_margin <= 0 or trigger_roi is None:
        log_info(f"{symbol} DCA complete or not configured")
        return

    last_order_at = (
        position_state.get("last_dca_at") or
        position_state.get("opened_at")
    )
    elapsed = seconds_since(last_order_at)

    avg_entry = float(
        position_detail.get("entry_price") or
        position_state.get("avg_entry") or
        0
    )
    old_quantity = abs(float(position_detail.get("amount", 0)))
    mark_price = get_mark_price(symbol)

    if avg_entry <= 0 or old_quantity <= 0 or mark_price is None:
        log_warning(f"{symbol} DCA skipped | position price unavailable")
        return

    trigger_entry = get_dca_trigger_entry(position_state, avg_entry)
    spacing_anchor_price = float(
        position_state.get("last_dca_price") or
        position_state.get("initial_entry") or
        avg_entry
    )
    position_adverse_roi = get_position_adverse_roi(side, avg_entry, mark_price)
    adverse_roi = get_position_adverse_roi(side, trigger_entry, mark_price)
    force_remaining_dca = (
        config.DCA_FORCE_REMAINING_ENABLED
        and adverse_roi >= config.DCA_FORCE_REMAINING_ROI
    )

    if force_remaining_dca:
        dca_margin = get_remaining_dca_margin(dca_count)

        if dca_margin <= 0:
            log_info(f"{symbol} forced DCA skipped | no remaining DCA margin")
            return

        log_warning(
            f"{symbol} FORCE REMAINING DCA | "
            f"ADVERSE_ROI={adverse_roi}% >= "
            f"FORCE={config.DCA_FORCE_REMAINING_ROI}% | "
            f"MARGIN={dca_margin}"
        )

    if (
        not force_remaining_dca
        and elapsed is not None
        and elapsed < config.DCA_MIN_SECONDS_BETWEEN_ORDERS
    ):
        remaining = int(config.DCA_MIN_SECONDS_BETWEEN_ORDERS - elapsed)
        log_info(f"{symbol} DCA waiting cooldown | {remaining}s remaining")
        return

    if not force_remaining_dca and adverse_roi < trigger_roi:
        log_info(
            f"{symbol} DCA not triggered | "
            f"LADDER_ROI={adverse_roi}% < TRIGGER={trigger_roi}% | "
            f"POSITION_ROI={position_adverse_roi}%"
        )
        return

    if not force_remaining_dca and adverse_roi > config.DCA_MAX_ADVERSE_ROI:
        log_warning(
            f"{symbol} DCA skipped | "
            f"ADVERSE_ROI={adverse_roi}% > MAX={config.DCA_MAX_ADVERSE_ROI}%"
        )
        return

    anchor_price = float(
        position_state.get("last_dca_price") or
        position_state.get("initial_entry") or
        avg_entry
    )
    gap_roi = get_dca_price_gap_roi(side, anchor_price, mark_price)

    if not force_remaining_dca and gap_roi < config.DCA_MIN_PRICE_GAP_ROI:
        log_info(
            f"{symbol} DCA waiting wider price gap | "
            f"GAP={gap_roi}% < MIN={config.DCA_MIN_PRICE_GAP_ROI}%"
        )
        return

    trend_df, confirm_df, entry_df = get_signal_frames(symbol, btc_trend_df)

    if not force_remaining_dca and (
        trend_df is None or confirm_df is None or entry_df is None
    ):
        log_warning(f"{symbol} DCA skipped | signal data unavailable")
        return

    if trend_df is not None and confirm_df is not None and entry_df is not None:
        btc_corr, rs = calculate_btc_context(symbol, trend_df, btc_trend_df)
        analysis = analyze_signal(
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            log_details=False
        )
    else:
        btc_corr = ""
        rs = ""
        analysis = {
            "signal": "FORCE_DCA",
            "best_side": side,
            "best_confidence": "",
            "buy": {},
            "sell": {},
        }

    side_analysis = analysis.get(side.lower(), {})
    opposite_key = "sell" if side == "BUY" else "buy"
    opposite = analysis.get(opposite_key, {})

    if not force_remaining_dca and config.DCA_REQUIRE_TREND_CONFIRMATION and not (
        side_analysis.get("trend_ok") and side_analysis.get("confirm_ok")
    ):
        log_warning(
            f"{symbol} DCA skipped | higher timeframe no longer confirms {side}"
        )
        append_signal_journal(
            symbol,
            analysis,
            None,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="DCA_SKIPPED",
            skip_reason="DCA_TREND_CONFIRMATION_FAILED"
        )
        return

    opposite_trend_ok = opposite.get("trend_following_ok", opposite.get("hard_ok"))

    if (
        not force_remaining_dca
        and
        opposite_trend_ok
        and opposite.get("confidence", 0) >= (
            side_analysis.get("confidence", 0) + config.LONG_TERM_MIN_SIGNAL_EDGE
        )
    ):
        log_warning(f"{symbol} DCA skipped | opposite signal is stronger")
        append_signal_journal(
            symbol,
            analysis,
            None,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="DCA_SKIPPED",
            skip_reason="DCA_OPPOSITE_SIGNAL_STRONGER"
        )
        return

    current_price = mark_price

    if not force_remaining_dca:
        guard_ok, current_price, guard_info = check_live_entry_guard(
            symbol,
            side,
            mark_price,
            mark_price=mark_price
        )

    if not force_remaining_dca and not guard_ok:
        log_live_guard_block(symbol, guard_info)
        append_signal_journal(
            symbol,
            analysis,
            None,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="DCA_SKIPPED",
            skip_reason=guard_info.get("reason")
        )
        return

    if force_remaining_dca:
        room_ok = True
        room_info = {"reason": "FORCE_DCA_PROFIT_ROOM_BYPASSED"}
    else:
        room_ok, room_info = validate_entry_profit_room(
            side,
            current_price,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE,
            min_roi_override=config.DCA_MIN_TP_ROOM_ROI
        )

    if not room_ok:
        log_warning(f"{symbol} DCA skipped | {room_info.get('reason')}")
        append_signal_journal(
            symbol,
            analysis,
            None,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="DCA_SKIPPED",
            skip_reason=room_info.get("reason")
        )
        return

    if not force_remaining_dca:
        log_profit_room_ok(symbol, side, room_info, prefix="DCA ")

    if force_remaining_dca:
        level_ok = True
        level_info = {
            "reason": "FORCE_REMAINING_DCA",
            "level": current_price,
            "source": "force_remaining_dca",
            "score": 0,
            "adverse_roi": adverse_roi,
            "position_adverse_roi": position_adverse_roi,
            "trigger_entry": trigger_entry,
        }
    else:
        level_ok, level_info = validate_dca_structure_level(
            side,
            current_price,
            trend_df,
            confirm_df,
            entry_df,
            leverage=config.LEVERAGE
        )

    if not level_ok:
        log_warning(f"{symbol} DCA skipped | {level_info.get('reason')}")
        append_signal_journal(
            symbol,
            analysis,
            None,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="DCA_SKIPPED",
            skip_reason=level_info.get("reason")
        )
        return

    if force_remaining_dca:
        log_warning(f"{symbol} DCA structure checks bypassed by force mode")
    else:
        log_dca_structure_level(symbol, side, level_info)

    balance = get_balance()
    quantity = calculate_position_size(
        balance,
        current_price,
        level_info["level"],
        symbol,
        dca_margin
    )

    if quantity <= 0:
        log_warning(f"{symbol} DCA skipped | invalid quantity")
        return

    notional_ok, notional = validate_min_notional(
        symbol,
        quantity,
        current_price
    )

    if not notional_ok:
        log_warning(f"{symbol} DCA skipped | notional too low: {notional}")
        return

    if not set_margin_type(symbol, allow_open_order_block=True):
        log_warning(f"{symbol} DCA aborted | margin setup failed")
        return

    if not setup_leverage(symbol):
        return

    if shutdown_event.is_set():
        log_warning(f"{symbol} DCA order skipped | bot shutdown requested")
        return

    order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
    order = place_market_order(symbol, order_side, quantity)

    if not order:
        return

    fill_price = get_entry_price(symbol, order)

    if fill_price <= 0:
        fill_price = current_price
        log_warning(f"{symbol} DCA fill price unavailable | using current price")

    avg_entry, total_quantity, updated_position = get_updated_position_after_fill(
        symbol,
        avg_entry,
        old_quantity,
        fill_price,
        quantity
    )

    record_dca_fill(
        state,
        symbol,
        avg_entry,
        total_quantity,
        dca_margin,
        fill_price,
        level_info,
        dca_count_increment=(
            get_remaining_dca_order_count(dca_count)
            if force_remaining_dca
            else 1
        )
    )

    structure_tp = None

    if (
        not config.STATIC_TP_ENABLED
        and trend_df is not None
        and confirm_df is not None
    ):
        tp_ok, structure_tp = validate_structure_take_profit(
            side,
            avg_entry,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE
        )

        if tp_ok:
            log_info(
                f"{symbol} DCA STRUCTURE TP | "
                f"TARGET={structure_tp['target_price']} | "
                f"ROI={structure_tp['target_roi']}% | "
                f"SRC={structure_tp['source']}"
            )
        else:
            log_warning(
                f"{symbol} DCA {structure_tp['reason']} | "
                f"USING FALLBACK ROI TP"
            )
    elif not config.STATIC_TP_ENABLED and force_remaining_dca:
        log_warning(
            f"{symbol} FORCE DCA using fallback ROI TP | "
            f"signal frames unavailable"
        )

    if config.DCA_REPRICE_TP_AFTER_FILL:
        if cancel_open_protection_orders(symbol):
            protection_ok = place_tp_sl_with_recovery(
                symbol,
                order_side,
                avg_entry,
                total_quantity,
                confirm_df,
                structure_tp=structure_tp,
                signal_type=(
                    position_state.get("confirmation_type") or
                    position_state.get("signal_type")
                ),
                context_label="LEGACY_DCA",
                return_details=False
            )

            if not protection_ok:
                log_warning(f"{symbol} DCA TP ORDER NOT CREATED")

    append_signal_journal(
        symbol,
        analysis,
        None,
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        action="DCA_FILLED"
    )

    log_info(
        f"*** {symbol} DCA FILLED ***\n"
        f"SIDE: {side}\n"
        f"FILL: {fill_price}\n"
        f"AVG_ENTRY: {avg_entry}\n"
        f"QTY_TOTAL: {total_quantity}\n"
        f"DCA_COUNT: "
        f"{dca_count + (get_remaining_dca_order_count(dca_count) if force_remaining_dca else 1)}"
        f"/{config.DCA_MAX_ORDERS}\n"
        f"ADVERSE_ROI_AT_TRIGGER: {adverse_roi}%\n"
    )

    if updated_position:
        position_detail.update(updated_position)


def manage_dca_position(
    symbol,
    state,
    position_detail,
    btc_trend_df,
    btc_trend,
    current_price_override=None,
    price_source="scan"
):
    if shutdown_event.is_set():
        log_warning(f"{symbol} DCA skipped | bot shutdown requested")
        return

    if not config.DCA_ENABLED:
        log_warning(f"{symbol} already has open position")
        return

    position_state = get_position_state(state, symbol)

    if not position_state:
        position_state = adopt_existing_position_state(
            state,
            symbol,
            position_detail
        )

    if not position_state or not position_state.get("managed_by_bot"):
        log_warning(f"{symbol} open position is not bot-managed; DCA skipped")
        return

    if (
        getattr(config, "TP1_RUNNER_DISABLE_DCA", True) and
        position_state.get("multi_tp_stage") in (
            RUNNER_PENDING,
            RUNNER_ACTIVE,
        )
    ):
        log_info(
            f"{symbol} DCA skipped | TP1 runner protection is active"
        )
        return

    side = position_state.get("side") or position_detail.get("side")

    if side not in ("BUY", "SELL"):
        log_warning(f"{symbol} DCA skipped | invalid side in state")
        return

    live_side = position_detail.get("side")

    if live_side and live_side != side:
        log_warning(
            f"{symbol} DCA skipped | state side {side} != live side {live_side}"
        )
        return

    dca_count = int(position_state.get("dca_count", 0) or 0)
    dca_margin = get_dca_order_margin(dca_count)
    trigger_roi = get_dca_trigger_roi(dca_count)

    if dca_margin <= 0 or trigger_roi is None:
        log_info(f"{symbol} DCA complete or not configured")
        return

    avg_entry = float(
        position_detail.get("entry_price") or
        position_state.get("avg_entry") or
        0
    )
    old_quantity = abs(float(position_detail.get("amount", 0)))
    current_price = (
        float(current_price_override)
        if current_price_override is not None
        else get_mark_price(symbol)
    )

    if avg_entry <= 0 or old_quantity <= 0 or current_price is None:
        log_warning(f"{symbol} DCA skipped | position price unavailable")
        return

    trigger_entry = get_dca_trigger_entry(position_state, avg_entry)
    spacing_anchor_price = float(
        position_state.get("last_dca_price") or
        position_state.get("initial_entry") or
        trigger_entry or
        avg_entry
    )
    position_adverse_roi = get_position_adverse_roi(
        side,
        avg_entry,
        current_price
    )
    adverse_roi = get_position_adverse_roi(side, trigger_entry, current_price)

    if adverse_roi < trigger_roi:
        log_info(
            f"{symbol} DCA not triggered | "
            f"LEVEL={dca_count + 1} | "
            f"LADDER_ROI={adverse_roi}% < TRIGGER={trigger_roi}% | "
            f"POSITION_ROI={position_adverse_roi}%"
        )
        return

    if (
        config.DCA_MAX_ADVERSE_ROI > 0 and
        adverse_roi > config.DCA_MAX_ADVERSE_ROI
    ):
        log_warning(
            f"{symbol} DCA skipped | maximum risk boundary exceeded | "
            f"ROI={adverse_roi}% > MAX={config.DCA_MAX_ADVERSE_ROI}%"
        )
        return

    last_order_at = (
        position_state.get("last_dca_at") or
        position_state.get("opened_at")
    )
    elapsed = seconds_since(last_order_at)

    if (
        config.DCA_MIN_SECONDS_BETWEEN_ORDERS > 0
        and elapsed is not None
        and elapsed < config.DCA_MIN_SECONDS_BETWEEN_ORDERS
    ):
        remaining = int(config.DCA_MIN_SECONDS_BETWEEN_ORDERS - elapsed)
        log_info(f"{symbol} DCA waiting cooldown | {remaining}s remaining")
        return

    level_info = {
        "reason": "ROI_LADDER_DCA",
        "level": current_price,
        "source": "roi_ladder",
        "price_source": price_source,
        "dca_level": dca_count + 1,
        "trigger_roi": trigger_roi,
        "adverse_roi": adverse_roi,
        "position_adverse_roi": position_adverse_roi,
        "trigger_entry": trigger_entry,
        "margin": dca_margin,
    }

    log_warning(
        f"{symbol} ROI LADDER DCA TRIGGERED | "
        f"LEVEL={dca_count + 1}/{config.DCA_MAX_ORDERS} | "
        f"LADDER_ROI={adverse_roi}% >= TRIGGER={trigger_roi}% | "
        f"POSITION_ROI={position_adverse_roi}% | "
        f"MARGIN={dca_margin} | SOURCE={price_source}"
    )

    trend_df = None
    confirm_df = None
    entry_df = None

    if (
        getattr(config, "DCA_STRICT_GUARD_ENABLED", True) or
        getattr(config, "DCA_TRIGGER_MODE", "static_roi") ==
        "adaptive_hybrid"
    ):
        trend_df, confirm_df, entry_df = get_signal_frames(symbol, btc_trend_df)
        guard_ok, guard_info = validate_dca_continuation_guard(
            side,
            current_price,
            avg_entry,
            trend_df,
            confirm_df,
            entry_df,
            leverage=config.LEVERAGE,
            confirmation_type=position_state.get("confirmation_type"),
            dca_level=dca_count + 1,
            adverse_roi=adverse_roi,
            position_adverse_roi=position_adverse_roi,
            trigger_roi=trigger_roi,
            spacing_anchor_price=spacing_anchor_price,
        )

        if not guard_ok:
            log_warning(
                f"{symbol} DCA skipped | {guard_info.get('reason')} | "
                f"PRESSURE={guard_info.get('pressure_score')} | "
                f"RECOVERY={guard_info.get('recovery_score')} | "
                f"TYPE={guard_info.get('trade_type')}"
            )
            return

        log_info(
            f"{symbol} DCA strict guard OK | "
            f"PRESSURE={guard_info.get('pressure_score')} | "
            f"RECOVERY={guard_info.get('recovery_score')} | "
            f"TYPE={guard_info.get('trade_type')} | "
            f"STRUCTURE={guard_info.get('structure', {}).get('reason')} | "
            f"ADAPTIVE={guard_info.get('adaptive', {}).get('reason')} | "
            f"GAP_ATR={guard_info.get('adaptive', {}).get('gap_atr')}"
        )
        level_info["dca_guard"] = guard_info

    balance = get_balance()
    quantity = calculate_position_size(
        balance,
        current_price,
        current_price,
        symbol,
        dca_margin
    )

    if quantity <= 0:
        log_warning(f"{symbol} DCA skipped | invalid quantity")
        return

    notional_ok, notional = validate_min_notional(
        symbol,
        quantity,
        current_price
    )

    if not notional_ok:
        log_warning(f"{symbol} DCA skipped | notional too low: {notional}")
        return

    if not set_margin_type(symbol, allow_open_order_block=True):
        log_warning(f"{symbol} DCA aborted | margin setup failed")
        return

    if not setup_leverage(symbol):
        log_warning(f"{symbol} DCA aborted | leverage setup failed")
        return

    if shutdown_event.is_set():
        log_warning(f"{symbol} DCA order skipped | bot shutdown requested")
        return

    reserved, reserve_reason = reserve_dca_level(
        state,
        symbol,
        dca_count,
        level_info
    )
    dca_level = dca_count + 1

    if not reserved:
        log_warning(
            f"{symbol} DCA skipped | LEVEL={dca_level} | {reserve_reason}"
        )
        return

    order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
    log_info(
        f"{symbol} DCA placing market order | "
        f"SIDE={order_side} | QTY={quantity} | MARGIN={dca_margin}"
    )
    order = place_market_order(symbol, order_side, quantity)

    if not order:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} DCA aborted | market order failed")
        return

    fill_price = get_entry_price(symbol, order)

    if fill_price <= 0:
        fill_price = current_price
        log_warning(f"{symbol} DCA fill price unavailable | using current price")

    avg_entry, total_quantity, updated_position = get_updated_position_after_fill(
        symbol,
        avg_entry,
        old_quantity,
        fill_price,
        quantity
    )

    if not record_dca_fill(
        state,
        symbol,
        avg_entry,
        total_quantity,
        dca_margin,
        fill_price,
        level_info
    ):
        log_error(
            f"{symbol} DCA fill state update failed | "
            f"LEVEL={dca_level} | reservation kept temporarily"
        )

    if trend_df is None or confirm_df is None or entry_df is None:
        trend_df, confirm_df, entry_df = get_signal_frames(symbol, btc_trend_df)

    btc_corr = ""
    rs = ""
    analysis = {
        "signal": f"DCA_LEVEL_{dca_count + 1}",
        "best_side": side,
        "best_confidence": "",
        "buy": {},
        "sell": {},
    }

    if trend_df is not None and confirm_df is not None and entry_df is not None:
        btc_corr, rs = calculate_btc_context(symbol, trend_df, btc_trend_df)

    structure_tp = None
    dca_tp_roi = None

    if config.DCA_TP_MODE in ("roi", "fixed_roi", "fallback_roi"):
        dca_tp_roi = config.DCA_TP_ROI
        log_info(
            f"{symbol} DCA ROI TP | "
            f"ROI={dca_tp_roi}% | AVG_ENTRY={avg_entry}"
        )
    elif not config.STATIC_TP_ENABLED and trend_df is not None and confirm_df is not None:
        tp_ok, structure_tp = validate_structure_take_profit(
            side,
            avg_entry,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE
        )

        if tp_ok:
            log_info(
                f"{symbol} DCA STRUCTURE TP | "
                f"TARGET={structure_tp['target_price']} | "
                f"ROI={structure_tp['target_roi']}% | "
                f"SRC={structure_tp['source']}"
            )
        else:
            log_warning(
                f"{symbol} DCA {structure_tp['reason']} | "
                f"USING FALLBACK ROI TP"
            )
    elif not config.STATIC_TP_ENABLED:
        log_warning(
            f"{symbol} DCA using fallback ROI TP | signal frames unavailable"
        )

    old_tp_info = get_open_take_profit_info(symbol)
    new_tp_info = {}

    if config.DCA_REPRICE_TP_AFTER_FILL:
        if cancel_open_protection_orders(symbol):
            protection_result = place_tp_sl_with_recovery(
                symbol,
                order_side,
                avg_entry,
                total_quantity,
                confirm_df,
                structure_tp=structure_tp,
                roi_override=dca_tp_roi,
                roi_mode_label=(
                    f"DCA_ROI_{dca_tp_roi}%"
                    if dca_tp_roi is not None
                    else None
                ),
                signal_type=(
                    position_state.get("confirmation_type") or
                    position_state.get("signal_type")
                ),
                context_label=f"DCA_LEVEL_{dca_count + 1}",
                enable_multi_tp=(
                    bool(getattr(config, "MULTI_TP_ENABLED", False)) and
                    position_state.get("multi_tp_stage") == TP1_PENDING
                ),
                position_side=position_detail.get("position_side"),
                return_details=True
            )
            protection_ok = bool(protection_result.get("ok"))
            new_tp_info = protection_result

            if not protection_ok:
                log_warning(f"{symbol} DCA TP ORDER NOT CREATED")
        else:
            log_warning(
                f"{symbol} DCA TP reprice skipped | "
                f"existing protection cancel failed"
            )

    if new_tp_info:
        update_position_tp_status(
            state,
            symbol,
            new_tp_info,
            context=f"DCA_LEVEL_{dca_count + 1}"
        )

    append_signal_journal(
        symbol,
        analysis,
        None,
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        action="DCA_FILLED"
    )

    log_info(
        f"*** {symbol} DCA FILLED ***\n"
        f"SIDE: {side}\n"
        f"LEVEL: {dca_count + 1}/{config.DCA_MAX_ORDERS}\n"
        f"FILL: {fill_price}\n"
        f"AVG_ENTRY: {avg_entry}\n"
        f"QTY_TOTAL: {total_quantity}\n"
        f"DCA_MARGIN: {dca_margin}\n"
        f"DCA_COUNT: {dca_count + 1}/{config.DCA_MAX_ORDERS}\n"
        f"ADVERSE_ROI_AT_TRIGGER: {adverse_roi}%\n"
    )
    send_dca_filled_message(
        symbol,
        side,
        dca_count + 1,
        config.DCA_MAX_ORDERS,
        adverse_roi,
        trigger_roi,
        fill_price,
        avg_entry,
        total_quantity,
        dca_margin,
        old_tp_info,
        new_tp_info,
        price_source
    )

    if updated_position:
        position_detail.update(updated_position)


def run_dca_check(
    symbol,
    position_detail,
    btc_trend_df,
    btc_trend,
    current_price_override=None,
    price_source="scan"
):
    if shutdown_event.is_set():
        log_warning(f"{symbol} DCA check skipped | bot shutdown requested")
        return

    lock = get_dca_lock(symbol)

    if not lock.acquire(blocking=False):
        log_info(f"{symbol} DCA check skipped | already running")
        return

    try:
        state = load_trade_state()
        manage_dca_position(
            symbol,
            state,
            position_detail,
            btc_trend_df,
            btc_trend,
            current_price_override=current_price_override,
            price_source=price_source
        )
    finally:
        lock.release()


def run_scan_dca_check(
    symbol,
    position_detail,
    btc_trend_df,
    btc_trend,
    dca_monitor=None
):
    if (
        dca_monitor is not None
        and dca_monitor.should_skip_scan_dca(symbol)
    ):
        log_info(f"{symbol} DCA scan skipped | websocket monitor active")
        return

    run_dca_check(
        symbol,
        position_detail,
        btc_trend_df,
        btc_trend,
        price_source="scan"
    )


def ensure_reversal_stop_loss(
    symbol,
    position_detail,
    state,
    btc_trend_df,
):
    if not getattr(config, "REVERSAL_SL_ENABLED", False):
        return

    position_state = get_position_state(state, symbol)

    if not position_state or not position_state.get("managed_by_bot"):
        return

    signal_type = str(
        position_state.get("confirmation_type") or
        position_state.get("signal_type") or
        ""
    ).upper()

    if signal_type != "REVERSAL":
        return

    if (
        position_state.get("sl_status") == "CREATED" and
        position_state.get("sl_price") not in (None, "")
    ):
        return

    existing_sl = get_open_stop_loss_info(symbol)

    if existing_sl.get("sl_price") not in (None, ""):
        update_position_runtime_fields(
            state,
            symbol,
            {
                "sl_status": "CREATED",
                "sl_enabled": True,
                "sl_price": existing_sl.get("sl_price"),
                "sl_source": existing_sl.get("source"),
            },
        )
        return

    entry_price = float(
        position_detail.get("entry_price") or
        position_state.get("avg_entry") or
        position_state.get("initial_entry") or
        0
    )

    if entry_price <= 0:
        log_warning(f"{symbol} reversal SL reconcile skipped | missing entry")
        return

    _, confirm_df, _ = get_signal_frames(symbol, btc_trend_df)

    if confirm_df is None:
        log_warning(
            f"{symbol} reversal SL reconcile skipped | "
            "confirmation data unavailable"
        )
        return

    order_side = (
        SIDE_BUY
        if position_state.get("side") == "BUY"
        else SIDE_SELL
    )
    result = place_stop_loss_only(
        symbol,
        order_side,
        entry_price,
        confirm_df,
        signal_type="REVERSAL",
    )
    sl_created = bool(result.get("ok"))
    update_position_runtime_fields(
        state,
        symbol,
        {
            "sl_status": "CREATED" if sl_created else "FAILED",
            "sl_enabled": sl_created,
            "sl_price": result.get("sl_price"),
            "sl_source": "REVERSAL_STARTUP_RECONCILE",
        },
    )

    if sl_created:
        send_telegram_message(
            f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
            f"{symbol} reversal stop loss added\n"
            f"SL: {result.get('sl_price')}"
        )
    else:
        log_error(f"{symbol} reversal SL reconcile failed")


def dca_tick_ready(symbol, mark_price, state=None):
    state = state or load_trade_state()
    position_state = get_position_state(state, symbol)

    if not position_state or not position_state.get("managed_by_bot"):
        return False

    if (
        getattr(config, "TP1_RUNNER_DISABLE_DCA", True) and
        position_state.get("multi_tp_stage") in (
            RUNNER_PENDING,
            RUNNER_ACTIVE,
        )
    ):
        return False

    if (
        position_state.get("reversal_profit_exit_status") == "SUBMITTED" or
        position_state.get("trend_profit_exit_status") == "SUBMITTED"
    ):
        return False

    if has_active_dca_reservation(state, symbol):
        return False

    side = position_state.get("side")

    if side not in ("BUY", "SELL"):
        return False

    dca_count = int(position_state.get("dca_count", 0) or 0)
    trigger_roi = get_dca_trigger_roi(dca_count)

    if trigger_roi is None:
        return False

    avg_entry = float(position_state.get("avg_entry") or 0)
    trigger_entry = get_dca_trigger_entry(position_state, avg_entry)

    if trigger_entry <= 0 or mark_price <= 0:
        return False

    adverse_roi = get_position_adverse_roi(side, trigger_entry, mark_price)

    if (
        config.DCA_MAX_ADVERSE_ROI > 0 and
        adverse_roi > config.DCA_MAX_ADVERSE_ROI
    ):
        return False

    return adverse_roi >= trigger_roi


def parse_mark_price_message(message):
    data = message.get("data", message) if isinstance(message, dict) else {}

    if not isinstance(data, dict):
        return None, None

    symbol = data.get("s")
    mark_price = data.get("p") or data.get("markPrice")

    try:
        return symbol, float(mark_price)
    except (TypeError, ValueError):
        return symbol, None


def calculate_runner_take_profit(
    symbol,
    side,
    basis_price,
    trend_df,
    confirm_df,
    signal_type=None,
):
    precision = get_price_precision(symbol)
    structure_tp = None

    if trend_df is not None and confirm_df is not None:
        tp_ok, structure_tp = validate_structure_take_profit(
            side,
            basis_price,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE,
        )

        if tp_ok and structure_tp.get("target_price"):
            target_roi = float(structure_tp.get("target_roi") or 0)

            if str(signal_type or "").upper() == "REVERSAL":
                reversal_max = max(
                    float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
                    0,
                )

                if reversal_max > 0 and target_roi > reversal_max:
                    target_price = roi_to_price(
                        side,
                        basis_price,
                        reversal_max,
                        leverage=config.LEVERAGE,
                    )
                    return (
                        round(float(target_price), precision),
                        f"TP2_REVERSAL_STRUCTURE_CAPPED_{reversal_max}%",
                        structure_tp,
                    )

            return (
                round(float(structure_tp["target_price"]), precision),
                f"TP2_STRUCTURE_{structure_tp.get('source', 'LEVEL')}",
                structure_tp,
            )

    fallback_roi = max(float(getattr(config, "TP2_FALLBACK_ROI", 35)), 0)

    if str(signal_type or "").upper() == "REVERSAL":
        reversal_max = max(
            float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
            0,
        )

        if reversal_max > 0:
            fallback_roi = min(fallback_roi, reversal_max)

    target_price = roi_to_price(
        side,
        basis_price,
        fallback_roi,
        leverage=config.LEVERAGE,
    )
    return (
        round(float(target_price), precision),
        f"TP2_FALLBACK_ROI_{fallback_roi}%",
        structure_tp or {"reason": "NO VALID TP2 STRUCTURE LEVEL FOUND"},
    )


def _target_stop_position_summary(position_details):
    if isinstance(position_details, dict):
        positions = position_details.values()
    else:
        positions = position_details or []

    return ", ".join(
        f"{detail.get('symbol')}:{detail.get('amount', 0)}:"
        f"{detail.get('position_side', 'BOTH')}"
        for detail in positions
    )


def close_all_open_positions_for_target_stop():
    attempts = max(int(config.TARGET_MARGIN_CLOSE_RETRIES) + 1, 1)
    verify_seconds = max(float(config.TARGET_MARGIN_CLOSE_VERIFY_SECONDS), 0)
    retry_seconds = max(float(config.TARGET_MARGIN_CLOSE_RETRY_SECONDS), 0)

    for attempt in range(1, attempts + 1):
        position_details = get_open_position_detail_rows(force=True)

        if position_details is None:
            log_error("Target margin stop close aborted | position snapshot unavailable")
            return False

        if not position_details:
            log_info("Target margin stop | no open positions to close")
            return True

        log_warning(
            f"Target margin stop close attempt {attempt}/{attempts} | "
            f"POSITIONS={_target_stop_position_summary(position_details)}"
        )

        for detail in position_details:
            symbol = detail.get("symbol")
            amount = detail.get("amount", 0)
            position_side = detail.get("position_side")
            cancel_open_protection_orders(symbol)

            if close_position_market(symbol, amount, position_side=position_side):
                log_warning(f"{symbol} target margin stop close submitted")
            else:
                log_error(f"{symbol} target margin stop close failed")

        if verify_seconds > 0:
            time.sleep(verify_seconds)

        remaining_positions = get_open_position_detail_rows(force=True)

        if remaining_positions is None:
            log_error("Target margin stop close verify failed | snapshot unavailable")
            return False

        if not remaining_positions:
            log_warning("Target margin stop | all open positions closed")
            return True

        log_error(
            "Target margin stop | positions still open after close attempt | "
            f"POSITIONS={_target_stop_position_summary(remaining_positions)}"
        )

        if attempt < attempts and retry_seconds > 0:
            time.sleep(retry_seconds)

    return False


def force_target_margin_process_exit(close_success):
    if not config.TARGET_MARGIN_FORCE_EXIT_ENABLED:
        log_warning("Target margin stop force exit disabled; main loop will stop")
        return

    delay_seconds = max(float(config.TARGET_MARGIN_EXIT_DELAY_SECONDS), 0)
    status = "closed" if close_success else "close_attempts_finished"
    log_warning(
        "Target margin stop process exit scheduled | "
        f"DELAY_SECONDS={delay_seconds} | STATUS={status}"
    )

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    log_warning("Target margin stop exiting process now")
    logging.shutdown()
    os._exit(0)


def trigger_target_margin_stop(margin_balance):
    if shutdown_event.is_set():
        return

    with target_margin_stop_lock:
        if shutdown_event.is_set():
            return

        shutdown_event.set()
        log_warning(
            "TARGET MARGIN BALANCE REACHED | "
            f"MARGIN_BALANCE={margin_balance} | "
            f"TARGET={config.TARGET_MARGIN_BALANCE}"
        )
        send_telegram_message("target margin balance reached")
        close_success = close_all_open_positions_for_target_stop()
        force_target_margin_process_exit(close_success)


class TargetMarginBalanceMonitor:
    def __init__(self):
        self.enabled = bool(
            config.TARGET_MARGIN_BALANCE_STOP_ENABLED
            and config.TARGET_MARGIN_BALANCE > 0
        )
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if not self.enabled:
            log_info("Target margin balance stop disabled")
            return

        self.thread = threading.Thread(
            target=self._run,
            name="target-margin-balance-monitor",
            daemon=True
        )
        self.thread.start()
        effective_check_seconds = max(
            float(config.TARGET_MARGIN_BALANCE_CHECK_SECONDS),
            float(config.TARGET_MARGIN_BALANCE_MIN_CHECK_SECONDS),
            0.2
        )
        log_info(
            "Target margin balance monitor started | "
            f"TARGET={config.TARGET_MARGIN_BALANCE} | "
            f"CHECK_SECONDS={effective_check_seconds}"
        )

    def stop(self):
        self.stop_event.set()

    def _run(self):
        interval = max(
            float(config.TARGET_MARGIN_BALANCE_CHECK_SECONDS),
            float(config.TARGET_MARGIN_BALANCE_MIN_CHECK_SECONDS),
            0.2
        )

        while not self.stop_event.is_set() and not shutdown_event.is_set():
            try:
                backoff_remaining = get_private_rest_backoff_remaining()

                if backoff_remaining > 0:
                    log_warning(
                        "Target margin balance monitor paused | "
                        "PRIVATE_REST_BACKOFF_ACTIVE | "
                        f"WAIT_SECONDS={round(backoff_remaining, 1)}"
                    )
                    self.stop_event.wait(max(backoff_remaining, interval))
                    continue

                margin_balance = get_margin_balance()

                if margin_balance >= config.TARGET_MARGIN_BALANCE:
                    trigger_target_margin_stop(margin_balance)
                    return

            except Exception as e:
                log_error(f"Target margin balance monitor error: {e}")

            self.stop_event.wait(interval)


class DcaWebsocketMonitor:
    def __init__(self):
        persisted_state = load_trade_state()
        persisted_multi_tp = any(
            item.get("multi_tp_active") and
            item.get("multi_tp_stage") in (
                TP1_PENDING,
                RUNNER_PENDING,
                RUNNER_ACTIVE,
            )
            for item in persisted_state.get("positions", {}).values()
        )
        multi_tp_monitoring = bool(
            getattr(config, "MULTI_TP_ENABLED", False) or
            persisted_multi_tp
        )
        self.enabled = bool(
            multi_tp_monitoring or
            (
                config.DCA_WEBSOCKET_ENABLED and
                (
                    config.DCA_ENABLED or
                    getattr(
                        config,
                        "REVERSAL_PROFIT_PROTECTION_ENABLED",
                        True,
                    ) or
                    getattr(config, "TREND_PROFIT_PROTECTION_ENABLED", False) or
                    getattr(config, "EARLY_FLOW_EXIT_ENABLED", False)
                )
            )
        )
        self.twm = None
        self.socket_key = None
        self.streams = ()
        self.lock = threading.Lock()
        self.running = False
        self.resetting = False
        self.last_message_at = 0.0
        self.last_restart_at = 0.0
        self.watchdog_thread = None
        self.watchdog_stop_event = threading.Event()
        self.protection_lock = threading.Lock()
        self.reversal_peaks = {}
        self.reversal_peak_entries = {}
        self.reversal_exit_pending = set()
        self.trend_peaks = {}
        self.trend_peak_entries = {}
        self.trend_exit_pending = set()
        self.route_invalidation_check_times = {}
        self.route_exit_pending = set()
        self.multi_tp_check_times = {}
        self.synced_position_details = {}

    def start(self):
        if not self.enabled:
            log_info("DCA websocket monitor disabled")
            return

        self._start_watchdog()

        try:
            from binance import ThreadedWebsocketManager

            self.twm = ThreadedWebsocketManager(
                api_key=config.API_KEY,
                api_secret=config.SECRET_KEY
            )
            self.twm.start()
            self.running = True
            self.last_message_at = time.time()
            log_info("DCA websocket monitor started")

        except Exception as e:
            self.running = False
            log_error(f"DCA websocket monitor start error: {e}")

    def stop(self):
        self.watchdog_stop_event.set()

        with self.lock:
            self.running = False
            self._stop_socket_locked()

            if self.twm:
                try:
                    self.twm.stop()
                except Exception as e:
                    log_warning(f"DCA websocket manager stop warning: {e}")

            self.twm = None

    def _start_watchdog(self):
        if not getattr(config, "DCA_WEBSOCKET_WATCHDOG_ENABLED", True):
            log_info("DCA websocket watchdog disabled")
            return

        if self.watchdog_thread and self.watchdog_thread.is_alive():
            return

        self.watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            name="dca-websocket-watchdog",
            daemon=True
        )
        self.watchdog_thread.start()
        log_info(
            "DCA websocket watchdog started | "
            f"STALE_SECONDS={config.DCA_WEBSOCKET_STALE_SECONDS}"
        )

    def _watchdog_loop(self):
        interval = max(
            float(getattr(config, "DCA_WEBSOCKET_WATCHDOG_INTERVAL_SECONDS", 10)),
            1
        )

        while (
            not shutdown_event.is_set()
            and not self.watchdog_stop_event.wait(interval)
        ):
            self._watchdog_check()

    def _watchdog_check(self):
        if shutdown_event.is_set() or not self.enabled:
            return

        now = time.time()
        stale_seconds = max(
            float(getattr(config, "DCA_WEBSOCKET_STALE_SECONDS", 45)),
            5
        )
        cooldown_seconds = max(
            float(getattr(config, "DCA_WEBSOCKET_RESTART_COOLDOWN_SECONDS", 30)),
            5
        )

        with self.lock:
            streams = self.streams
            socket_key = self.socket_key
            running = self.running
            resetting = self.resetting
            last_message_at = self.last_message_at
            last_restart_at = self.last_restart_at

        if resetting or not streams:
            return

        reason = None

        if not running:
            reason = "manager stopped"
        elif not socket_key:
            reason = "socket missing"
        else:
            age = now - last_message_at if last_message_at else stale_seconds + 1

            if age < stale_seconds:
                return

            reason = f"stale {round(age, 1)}s >= {stale_seconds}s"

        if now - last_restart_at < cooldown_seconds:
            return

        with self.lock:
            if self.resetting:
                return

            self.last_restart_at = now

        log_warning(
            f"DCA websocket watchdog restart | REASON={reason} | "
            f"STREAMS={len(streams)}"
        )
        self.reset_connection(f"watchdog {reason}", force=True)

    def _stop_socket_locked(self):
        if not self.socket_key:
            return

        try:
            self.twm.stop_socket(self.socket_key)
        except Exception as e:
            log_warning(f"DCA websocket stop warning: {e}")

        self.socket_key = None

    def _restart_manager_locked(self):
        if shutdown_event.is_set() or not self.enabled:
            return False

        try:
            from binance import ThreadedWebsocketManager

            if self.twm:
                try:
                    self.twm.stop()
                except Exception as e:
                    log_warning(f"DCA websocket manager stop warning: {e}")

            self.twm = ThreadedWebsocketManager(
                api_key=config.API_KEY,
                api_secret=config.SECRET_KEY
            )
            self.twm.start()
            self.running = True
            log_info("DCA websocket manager restarted")
            return True

        except Exception as e:
            self.running = False
            self.twm = None
            self.socket_key = None
            log_error(f"DCA websocket manager restart error: {e}")
            return False

    def _subscribe_locked(self, streams, reason):
        self.streams = streams
        if not self.twm:
            log_error("DCA websocket subscription failed: websocket manager unavailable")
            self.socket_key = None
            return

        if not streams:
            log_info("DCA websocket monitor idle | no open positions")
            self.last_message_at = 0.0
            return

        try:
            self.socket_key = self.twm.start_futures_multiplex_socket(
                callback=self.handle_message,
                streams=list(streams)
            )
            self.last_message_at = time.time()
            log_info(
                f"DCA websocket watching {len(streams)} open position stream(s) | "
                f"REASON={reason}"
            )

        except Exception as e:
            log_error(f"DCA websocket subscribe error: {e}")
            self.socket_key = None

    def sync(self, position_details):
        if shutdown_event.is_set():
            return

        symbols = sorted((position_details or {}).keys())
        active_symbols = set(symbols)

        with self.protection_lock:
            self.reversal_peaks = {
                symbol: peak
                for symbol, peak in self.reversal_peaks.items()
                if symbol in active_symbols
            }
            self.reversal_peak_entries = {
                symbol: entry
                for symbol, entry in self.reversal_peak_entries.items()
                if symbol in active_symbols
            }
            self.reversal_exit_pending.intersection_update(active_symbols)
            self.trend_peaks = {
                symbol: peak
                for symbol, peak in self.trend_peaks.items()
                if symbol in active_symbols
            }
            self.trend_peak_entries = {
                symbol: entry
                for symbol, entry in self.trend_peak_entries.items()
                if symbol in active_symbols
            }
            self.trend_exit_pending.intersection_update(active_symbols)
            self.route_invalidation_check_times = {
                symbol: checked_at
                for symbol, checked_at in self.route_invalidation_check_times.items()
                if symbol in active_symbols
            }
            self.route_exit_pending.intersection_update(active_symbols)
            self.multi_tp_check_times = {
                symbol: checked_at
                for symbol, checked_at in self.multi_tp_check_times.items()
                if symbol in active_symbols
            }
            self.synced_position_details = {
                symbol: dict(detail)
                for symbol, detail in (position_details or {}).items()
            }

        suffix = "@markPrice@1s" if config.DCA_WEBSOCKET_FAST_MARK_PRICE else "@markPrice"
        streams = tuple(f"{symbol.lower()}{suffix}" for symbol in symbols)

        with self.lock:
            if not self.running and not self._restart_manager_locked():
                self.streams = streams
                return

            if streams == self.streams and self.socket_key:
                return

            self._stop_socket_locked()
            self._subscribe_locked(streams, "sync")

    def should_skip_scan_dca(self, symbol):
        if not self.enabled or not self.running:
            return False

        if getattr(config, "DCA_SCAN_WHEN_WEBSOCKET_ENABLED", False):
            return False

        suffix = "@markPrice@1s" if config.DCA_WEBSOCKET_FAST_MARK_PRICE else "@markPrice"
        stream = f"{symbol.lower()}{suffix}"

        with self.lock:
            return bool(self.socket_key and stream in self.streams)

    def reset_connection(self, reason, force=False):
        if shutdown_event.is_set() or not self.enabled:
            return

        if not force and not self.running:
            return

        with self.lock:
            if self.resetting:
                return

            self.resetting = True

        thread = threading.Thread(
            target=self._reset_connection,
            args=(reason,),
            daemon=True
        )
        thread.start()

    def _reset_connection(self, reason):
        time.sleep(2)

        try:
            with self.lock:
                streams = self.streams
                log_warning(
                    f"DCA websocket resetting | REASON={reason} | "
                    f"STREAMS={len(streams)}"
                )
                self._stop_socket_locked()

                if self._restart_manager_locked():
                    self.streams = ()
                    self._subscribe_locked(streams, "reset")
                else:
                    self.streams = streams

        finally:
            with self.lock:
                self.resetting = False

    def handle_message(self, message):
        if shutdown_event.is_set():
            return

        if isinstance(message, dict) and message.get("e") == "error":
            log_warning(f"DCA websocket error: {message}")
            self.reset_connection(message.get("type") or message.get("m") or "error")
            return

        symbol, mark_price = parse_mark_price_message(message)

        if not symbol or mark_price is None:
            return

        with self.lock:
            self.last_message_at = time.time()

        state = load_trade_state()

        if self._handle_multi_tp_runner(symbol, mark_price, state):
            return

        if self._handle_reversal_profit_protection(
            symbol,
            mark_price,
            state,
        ):
            return

        if self._handle_trend_profit_protection(
            symbol,
            mark_price,
            state,
        ):
            return

        if self._handle_route_early_invalidation(
            symbol,
            mark_price,
            state,
        ):
            return

        if not dca_tick_ready(symbol, mark_price, state=state):
            return

        log_warning(
            f"{symbol} DCA websocket trigger candidate | MARK={mark_price}"
        )
        details = get_open_position_details(symbol)
        position_detail = (details or {}).get(symbol)

        if not position_detail:
            log_warning(f"{symbol} DCA websocket skipped | live position not found")
            return

        run_dca_check(
            symbol,
            position_detail,
            None,
            "NEUTRAL",
            current_price_override=mark_price,
            price_source="websocket"
        )

    def _multi_tp_retry_ready(self, symbol):
        now = time.monotonic()
        retry_seconds = max(
            float(getattr(config, "TP1_RUNNER_RETRY_SECONDS", 5)),
            1,
        )

        with self.protection_lock:
            last_check = float(self.multi_tp_check_times.get(symbol, 0) or 0)

            if now - last_check < retry_seconds:
                return False

            self.multi_tp_check_times[symbol] = now
            return True

    def _synced_multi_tp_fill_confirmed(self, symbol, position_state):
        with self.protection_lock:
            detail = self.synced_position_details.get(symbol)

        if not detail:
            return False

        return tp1_fill_confirmed(
            position_state.get("tp1_base_quantity"),
            position_state.get("tp1_quantity"),
            detail.get("quantity", abs(float(detail.get("amount", 0) or 0))),
        )

    def _handle_multi_tp_runner(self, symbol, mark_price, state):
        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("multi_tp_active"):
            return False

        stage = position_state.get("multi_tp_stage")

        if stage == RUNNER_ACTIVE:
            tp1_order_id = position_state.get("tp1_order_id") or ""
            old_sl_order_id = position_state.get("initial_sl_order_id") or ""
            runner_sl_order_id = position_state.get("runner_sl_order_id") or ""

            if (
                (tp1_order_id or (old_sl_order_id and runner_sl_order_id)) and
                self._multi_tp_retry_ready(symbol)
            ):
                lock = get_dca_lock(symbol)

                if lock.acquire(blocking=False):
                    try:
                        cleanup_updates = {}

                        if (
                            tp1_order_id and
                            cancel_algo_order(symbol, tp1_order_id)
                        ):
                            cleanup_updates["tp1_order_id"] = ""

                        if (
                            old_sl_order_id and
                            runner_sl_order_id and
                            cancel_algo_order(symbol, old_sl_order_id)
                        ):
                            cleanup_updates["initial_sl_order_id"] = ""

                        if cleanup_updates:
                            update_position_runtime_fields(
                                state,
                                symbol,
                                cleanup_updates,
                            )
                    finally:
                        lock.release()

            return False

        if stage not in (TP1_PENDING, RUNNER_PENDING):
            return False

        fill_seen = self._synced_multi_tp_fill_confirmed(
            symbol,
            position_state,
        )

        if (
            stage == TP1_PENDING and
            not fill_seen and
            not tp1_trigger_reached(
                position_state.get("side"),
                mark_price,
                position_state.get("tp1_price"),
            )
        ):
            return False

        if not self._multi_tp_retry_ready(symbol):
            return True

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(f"{symbol} TP1 runner transition deferred | position busy")
            return True

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if not fresh_position_state or not fresh_position_state.get(
                "multi_tp_active"
            ):
                return True

            fresh_stage = fresh_position_state.get("multi_tp_stage")

            if fresh_stage == RUNNER_ACTIVE:
                return False

            details = get_open_position_details(symbol, force=True)
            position_detail = (details or {}).get(symbol)

            if not position_detail:
                log_warning(
                    f"{symbol} TP1 runner transition skipped | "
                    "live position not found"
                )
                return True

            live_quantity = abs(float(position_detail.get("quantity", 0) or 0))

            if fresh_stage == TP1_PENDING:
                if not tp1_fill_confirmed(
                    fresh_position_state.get("tp1_base_quantity"),
                    fresh_position_state.get("tp1_quantity"),
                    live_quantity,
                ):
                    return True

                runner_basis = float(mark_price)
                update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {
                        "multi_tp_stage": RUNNER_PENDING,
                        "tp1_filled_at": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "tp1_fill_price": fresh_position_state.get("tp1_price"),
                        "runner_basis_price": runner_basis,
                        "runner_quantity": live_quantity,
                        "runner_protection_error": "",
                    },
                )
                fresh_position_state = get_position_state(fresh_state, symbol)
                tp1_order_id = fresh_position_state.get("tp1_order_id") or ""

                if tp1_order_id and cancel_algo_order(symbol, tp1_order_id):
                    update_position_runtime_fields(
                        fresh_state,
                        symbol,
                        {"tp1_order_id": ""},
                    )
                    fresh_position_state = get_position_state(
                        fresh_state,
                        symbol,
                    )

                log_info(
                    f"{symbol} TP1 fill confirmed | "
                    f"REMAINING_QTY={live_quantity} | "
                    f"RUNNER_BASIS={runner_basis}"
                )

            return self._configure_multi_tp_runner(
                symbol,
                mark_price,
                fresh_state,
                fresh_position_state,
                position_detail,
            )

        except Exception as e:
            log_error(f"{symbol} TP1 runner transition error: {e}")
            return True

        finally:
            lock.release()

    def _configure_multi_tp_runner(
        self,
        symbol,
        mark_price,
        state,
        position_state,
        position_detail,
    ):
        side = str(position_state.get("side") or "").upper()
        order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
        original_entry = float(
            position_state.get("avg_entry") or
            position_state.get("initial_entry") or
            0
        )
        runner_basis = float(
            position_state.get("runner_basis_price") or mark_price
        )
        signal_type = (
            position_state.get("confirmation_type") or
            position_state.get("signal_type")
        )
        precision = get_price_precision(symbol)
        runner_tp_price = position_state.get("runner_tp_price")
        runner_tp_mode = position_state.get("runner_tp_mode") or ""
        runner_sl_price = position_state.get("runner_sl_price")
        runner_sl_mode = position_state.get("runner_sl_mode") or ""

        if original_entry <= 0 or runner_basis <= 0:
            update_position_runtime_fields(
                state,
                symbol,
                {"runner_protection_error": "RUNNER_BASIS_INVALID"},
            )
            return True

        if not runner_tp_price or (
            getattr(config, "TP1_RUNNER_STOP_ENABLED", True) and
            not runner_sl_price
        ):
            trend_df, confirm_df, _ = get_signal_frames(symbol, None)

            if not runner_tp_price:
                runner_tp_price, runner_tp_mode, tp_context = (
                    calculate_runner_take_profit(
                        symbol,
                        side,
                        runner_basis,
                        trend_df,
                        confirm_df,
                        signal_type=signal_type,
                    )
                )
                runner_tp_price = round(float(runner_tp_price), precision)
            else:
                tp_context = {}

            if (
                getattr(config, "TP1_RUNNER_STOP_ENABLED", True) and
                not runner_sl_price
            ):
                runner_sl_price, sl_context = calculate_runner_stop(
                    side,
                    original_entry,
                    mark_price,
                    confirm_df,
                    leverage=config.LEVERAGE,
                )

                if runner_sl_price is None:
                    reason = sl_context.get("reason", "RUNNER_STOP_UNAVAILABLE")
                    update_position_runtime_fields(
                        state,
                        symbol,
                        {"runner_protection_error": reason},
                    )
                    log_warning(f"{symbol} TP1 runner SL unavailable | {reason}")
                    return True

                runner_sl_price = round(float(runner_sl_price), precision)
                runner_sl_mode = (
                    f"RUNNER_{sl_context.get('source', 'PROFIT_LOCK')}"
                )

            update_position_runtime_fields(
                state,
                symbol,
                {
                    "runner_tp_price": runner_tp_price,
                    "runner_tp_mode": runner_tp_mode,
                    "runner_tp_context": tp_context,
                    "runner_sl_price": runner_sl_price,
                    "runner_sl_mode": runner_sl_mode,
                    "runner_protection_error": "",
                },
            )
            position_state = get_position_state(state, symbol)

        position_side = position_detail.get("position_side")
        runner_sl_order_id = position_state.get("runner_sl_order_id") or ""
        runner_tp_order_id = position_state.get("runner_tp_order_id") or ""
        tolerance = max(10 ** -precision, abs(mark_price) * 1e-9)

        if (
            getattr(config, "TP1_RUNNER_STOP_ENABLED", True) and
            not runner_sl_order_id
        ):
            sl_is_valid = (
                float(runner_sl_price) < mark_price
                if side == "BUY"
                else float(runner_sl_price) > mark_price
            )

            if not sl_is_valid:
                _, confirm_df, _ = get_signal_frames(symbol, None)
                runner_sl_price, sl_context = calculate_runner_stop(
                    side,
                    original_entry,
                    mark_price,
                    confirm_df,
                    leverage=config.LEVERAGE,
                )

                if runner_sl_price is None:
                    reason = sl_context.get(
                        "reason",
                        "RUNNER_STOP_NO_LONGER_VALID",
                    )
                    update_position_runtime_fields(
                        state,
                        symbol,
                        {"runner_protection_error": reason},
                    )
                    return True

                runner_sl_price = round(float(runner_sl_price), precision)
                runner_sl_mode = (
                    f"RUNNER_{sl_context.get('source', 'PROFIT_LOCK')}"
                )
                update_position_runtime_fields(
                    state,
                    symbol,
                    {
                        "runner_sl_price": runner_sl_price,
                        "runner_sl_mode": runner_sl_mode,
                    },
                )
                position_state = get_position_state(state, symbol)

        tp_is_ahead = (
            float(runner_tp_price) > mark_price
            if side == "BUY"
            else float(runner_tp_price) < mark_price
        )

        if not runner_tp_order_id and not tp_is_ahead:
            fallback_roi = max(
                float(getattr(config, "TP2_FALLBACK_ROI", 35)),
                0,
            )

            if str(signal_type or "").upper() == "REVERSAL":
                reversal_max = max(
                    float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
                    0,
                )

                if reversal_max > 0:
                    fallback_roi = min(fallback_roi, reversal_max)

            runner_tp_price = round(
                roi_to_price(
                    side,
                    mark_price,
                    fallback_roi,
                    leverage=config.LEVERAGE,
                ),
                precision,
            )
            runner_tp_mode = f"TP2_REBASED_FALLBACK_ROI_{fallback_roi}%"
            update_position_runtime_fields(
                state,
                symbol,
                {
                    "runner_basis_price": mark_price,
                    "runner_tp_price": runner_tp_price,
                    "runner_tp_mode": runner_tp_mode,
                },
            )
            position_state = get_position_state(state, symbol)

        if (
            getattr(config, "TP1_RUNNER_STOP_ENABLED", True) and
            not runner_sl_order_id
        ):
            existing_sl = get_open_stop_loss_info(symbol)
            existing_sl_price = float(existing_sl.get("sl_price") or 0)

            if abs(existing_sl_price - float(runner_sl_price)) <= tolerance:
                runner_sl_order_id = existing_sl.get("order_id") or ""
            else:
                sl_order = place_close_position_protection(
                    symbol,
                    order_side,
                    "STOP_MARKET",
                    runner_sl_price,
                    position_side=position_side,
                )
                runner_sl_order_id = extract_order_id(sl_order)

            if not runner_sl_order_id:
                update_position_runtime_fields(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_SL_ORDER_FAILED"},
                )
                return True

            update_position_runtime_fields(
                state,
                symbol,
                {"runner_sl_order_id": runner_sl_order_id},
            )
            position_state = get_position_state(state, symbol)

        if not runner_tp_order_id:
            existing_tp = get_open_take_profit_info(symbol)
            existing_tp_price = float(existing_tp.get("tp_price") or 0)

            if abs(existing_tp_price - float(runner_tp_price)) <= tolerance:
                runner_tp_order_id = existing_tp.get("order_id") or ""
            else:
                tp_order = place_close_position_protection(
                    symbol,
                    order_side,
                    "TAKE_PROFIT_MARKET",
                    runner_tp_price,
                    position_side=position_side,
                )
                runner_tp_order_id = extract_order_id(tp_order)

            if not runner_tp_order_id:
                update_position_runtime_fields(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_TP_ORDER_FAILED"},
                )
                return True

            update_position_runtime_fields(
                state,
                symbol,
                {"runner_tp_order_id": runner_tp_order_id},
            )
            position_state = get_position_state(state, symbol)

        old_sl_order_id = position_state.get("initial_sl_order_id") or ""

        if runner_sl_order_id and old_sl_order_id:
            if cancel_algo_order(symbol, old_sl_order_id):
                old_sl_order_id = ""

        update_position_runtime_fields(
            state,
            symbol,
            {
                "multi_tp_stage": RUNNER_ACTIVE,
                "initial_sl_order_id": old_sl_order_id,
                "tp_status": "CREATED",
                "tp_price": runner_tp_price,
                "tp_mode": runner_tp_mode,
                "tp_context": "TP2_RUNNER",
                "sl_status": (
                    "CREATED"
                    if getattr(config, "TP1_RUNNER_STOP_ENABLED", True)
                    else position_state.get("sl_status", "DISABLED")
                ),
                "sl_enabled": bool(
                    runner_sl_order_id or position_state.get("sl_enabled")
                ),
                "sl_price": (
                    runner_sl_price
                    if runner_sl_order_id
                    else position_state.get("sl_price")
                ),
                "sl_source": "TP1_RUNNER",
                "runner_protection_error": "",
            },
        )
        log_info(
            f"{symbol} TP1 runner protected | TP2={runner_tp_price} | "
            f"TP2_MODE={runner_tp_mode} | "
            f"SL={runner_sl_price if runner_sl_order_id else 'UNCHANGED'}"
        )
        send_telegram_message(
            f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
            f"{symbol} TP1 reached and partial profit booked\n"
            f"Remaining quantity: {position_detail.get('quantity')}\n"
            f"TP2: {runner_tp_price}\n"
            f"Runner stop: "
            f"{runner_sl_price if runner_sl_order_id else 'unchanged'}"
        )
        return True

    def _route_early_invalidation_context(self, position_state, mark_price):
        if not position_state or not position_state.get("managed_by_bot"):
            return None

        route = (
            "REVERSAL"
            if str(
                position_state.get("confirmation_type") or
                position_state.get("signal_type") or
                ""
            ).upper() == "REVERSAL"
            else "TREND"
        )
        route_enabled = (
            getattr(config, "EARLY_FLOW_EXIT_REVERSAL_ENABLED", True)
            if route == "REVERSAL"
            else getattr(config, "EARLY_FLOW_EXIT_TREND_ENABLED", True)
        )

        if not route_enabled:
            return None

        side = str(position_state.get("side") or "").upper()
        avg_entry = _safe_float(position_state.get("avg_entry"))

        if side not in ("BUY", "SELL") or avg_entry <= 0 or mark_price <= 0:
            return None

        last_dca_at = position_state.get("last_dca_at")
        activity_at = last_dca_at or position_state.get("opened_at")
        elapsed = seconds_since(activity_at)
        grace_minutes = float(
            getattr(
                config,
                "EARLY_FLOW_EXIT_POST_DCA_GRACE_MINUTES",
                config.EARLY_FLOW_EXIT_MINUTES,
            )
            if last_dca_at
            else config.EARLY_FLOW_EXIT_MINUTES
        )

        if elapsed is None or elapsed < max(grace_minutes, 0) * 60:
            return None

        current_roi = -get_position_adverse_roi(side, avg_entry, mark_price)
        max_roi = min(
            float(
                getattr(
                    config,
                    (
                        "EARLY_FLOW_EXIT_REVERSAL_MAX_ROI"
                        if route == "REVERSAL"
                        else "EARLY_FLOW_EXIT_TREND_MAX_ROI"
                    ),
                    config.EARLY_FLOW_EXIT_MAX_ROI,
                )
            ),
            0,
        )

        if current_roi > max_roi:
            return None

        reference_price = None

        if not position_state.get("adopted_existing"):
            reference_price = _safe_float(position_state.get("reference_price"))

        return {
            "route": route,
            "side": side,
            "avg_entry": avg_entry,
            "current_roi": current_roi,
            "max_roi": max_roi,
            "reference_price": reference_price,
        }

    def _handle_route_early_invalidation(self, symbol, mark_price, state):
        if not getattr(config, "EARLY_FLOW_EXIT_ENABLED", False):
            return False

        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("managed_by_bot"):
            return False

        if position_state.get("early_invalidation_exit_status") == "SUBMITTED":
            return True

        context = self._route_early_invalidation_context(
            position_state,
            mark_price,
        )

        if not context:
            return False

        now = time.monotonic()
        check_seconds = max(
            float(getattr(config, "EARLY_FLOW_EXIT_CHECK_SECONDS", 60)),
            1,
        )

        with self.protection_lock:
            if symbol in self.route_exit_pending:
                return True

            last_check = float(
                self.route_invalidation_check_times.get(symbol, 0) or 0
            )

            if now - last_check < check_seconds:
                return False

            self.route_invalidation_check_times[symbol] = now

        try:
            fast_raw = get_klines(
                symbol,
                config.LIVE_ENTRY_FAST_TIMEFRAME,
                config.LIVE_ENTRY_KLINE_LIMIT,
            )
            slow_raw = get_klines(
                symbol,
                config.LIVE_ENTRY_SLOW_TIMEFRAME,
                config.LIVE_ENTRY_KLINE_LIMIT,
            )
            fast_df = apply_indicators(fast_raw) if fast_raw is not None else None
            slow_df = apply_indicators(slow_raw) if slow_raw is not None else None
            info = evaluate_route_early_invalidation(
                context["side"],
                fast_df,
                slow_df,
                mark_price,
                confirmation_type=context["route"],
                reference_price=context["reference_price"],
            )

            if not info.get("should_exit"):
                if (
                    info.get("reason") == "EARLY_INVALIDATION_DATA_UNAVAILABLE" and
                    getattr(config, "EARLY_FLOW_EXIT_REQUIRE_DATA", True)
                ):
                    log_warning(
                        f"{symbol} early invalidation skipped | live data unavailable"
                    )

                return False

        except Exception as e:
            log_error(f"{symbol} early invalidation analysis error: {e}")
            return False

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(f"{symbol} early invalidation deferred | position busy")
            return True

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if (
                not fresh_position_state or
                fresh_position_state.get("early_invalidation_exit_status") == "SUBMITTED"
            ):
                return True

            fresh_context = self._route_early_invalidation_context(
                fresh_position_state,
                mark_price,
            )

            if not fresh_context:
                return True

            info = evaluate_route_early_invalidation(
                fresh_context["side"],
                fast_df,
                slow_df,
                mark_price,
                confirmation_type=fresh_context["route"],
                reference_price=fresh_context["reference_price"],
            )

            if not info.get("should_exit"):
                return True

            with self.protection_lock:
                if symbol in self.route_exit_pending:
                    return True

                self.route_exit_pending.add(symbol)

            details = get_open_position_details(symbol)
            position_detail = (details or {}).get(symbol)

            if not position_detail:
                log_warning(
                    f"{symbol} early invalidation exit skipped | "
                    "live position not found"
                )

                with self.protection_lock:
                    self.route_exit_pending.discard(symbol)

                return True

            amount = float(position_detail.get("amount", 0) or 0)
            position_side = position_detail.get("position_side")
            log_warning(
                f"{symbol} {fresh_context['route']} EARLY INVALIDATION EXIT | "
                f"ROI={fresh_context['current_roi']}% | "
                f"REASON={info.get('reason')} | "
                f"FAST_FAILURE={info.get('fast_failure')} | "
                f"SLOW_FAILURE={info.get('slow_failure')} | "
                f"REFERENCE_BROKEN={info.get('reference_broken')}"
            )
            closed = close_position_market(
                symbol,
                amount,
                position_side=position_side,
            )

            if closed:
                cancel_open_protection_orders(symbol)
                evidence = {
                    "fast_failure": bool(info.get("fast_failure")),
                    "slow_failure": bool(info.get("slow_failure")),
                    "fast_adverse": bool(info.get("fast_adverse")),
                    "slow_adverse": bool(info.get("slow_adverse")),
                    "dual_opposition": bool(info.get("dual_opposition")),
                    "reference_broken": bool(info.get("reference_broken")),
                    "fast_support_score": (info.get("fast") or {}).get(
                        "support_score"
                    ),
                    "slow_support_score": (info.get("slow") or {}).get(
                        "support_score"
                    ),
                }
                update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {
                        "early_invalidation_exit_status": "SUBMITTED",
                        "early_invalidation_exit_price": mark_price,
                        "early_invalidation_exit_roi": fresh_context["current_roi"],
                        "early_invalidation_exit_reason": info.get("reason"),
                        "early_invalidation_exit_route": fresh_context["route"],
                        "early_invalidation_exit_evidence": evidence,
                    },
                )
                send_telegram_message(
                    f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                    f"{symbol} {fresh_context['route'].lower()} early invalidation exit\n"
                    f"ROI: {fresh_context['current_roi']}%\n"
                    f"Reason: {info.get('reason')}"
                )
                return True

            log_error(f"{symbol} early invalidation exit order failed")
            update_position_runtime_fields(
                fresh_state,
                symbol,
                {"early_invalidation_exit_status": "FAILED"},
            )

            with self.protection_lock:
                self.route_exit_pending.discard(symbol)

            return True

        except Exception as e:
            log_error(f"{symbol} early invalidation exit error: {e}")

            with self.protection_lock:
                self.route_exit_pending.discard(symbol)

            return True

        finally:
            lock.release()

    def _handle_reversal_profit_protection(self, symbol, mark_price, state):
        return self._handle_route_profit_protection(
            symbol,
            mark_price,
            state,
            "REVERSAL",
        )

    def _handle_trend_profit_protection(self, symbol, mark_price, state):
        return self._handle_route_profit_protection(
            symbol,
            mark_price,
            state,
            "TREND",
        )

    def _handle_route_profit_protection(
        self,
        symbol,
        mark_price,
        state,
        route,
    ):
        route = "REVERSAL" if str(route).upper() == "REVERSAL" else "TREND"
        route_key = route.lower()
        enabled = bool(
            getattr(
                config,
                f"{route}_PROFIT_PROTECTION_ENABLED",
                route == "REVERSAL",
            )
        )

        if not enabled:
            return False

        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("managed_by_bot"):
            return False

        signal_type = str(
            position_state.get("confirmation_type") or
            position_state.get("signal_type") or
            ""
        ).upper()

        if signal_type != route:
            return False

        exit_status_field = f"{route_key}_profit_exit_status"

        if position_state.get(exit_status_field) == "SUBMITTED":
            return True

        side = position_state.get("side")
        avg_entry = float(position_state.get("avg_entry") or 0)
        peak_field = f"{route_key}_peak_roi"
        basis_field = f"{route_key}_profit_basis_entry"
        saved_peak = float(position_state.get(peak_field) or 0)
        saved_basis = float(position_state.get(basis_field) or avg_entry)
        peak_map = (
            self.reversal_peaks
            if route == "REVERSAL"
            else self.trend_peaks
        )
        basis_map = (
            self.reversal_peak_entries
            if route == "REVERSAL"
            else self.trend_peak_entries
        )
        pending = (
            self.reversal_exit_pending
            if route == "REVERSAL"
            else self.trend_exit_pending
        )
        basis_tolerance = max(abs(avg_entry) * 1e-10, 1e-10)

        if abs(saved_basis - avg_entry) > basis_tolerance:
            saved_peak = 0

        with self.protection_lock:
            memory_basis = float(basis_map.get(symbol, avg_entry) or avg_entry)
            memory_peak = float(peak_map.get(symbol, 0) or 0)

            if abs(memory_basis - avg_entry) > basis_tolerance:
                memory_peak = 0

            previous_peak = max(saved_peak, memory_peak)

        info = evaluate_route_profit_protection(
            side,
            avg_entry,
            mark_price,
            peak_roi=previous_peak,
            leverage=config.LEVERAGE,
            confirmation_type=route,
        )
        peak_roi = float(info.get("peak_roi", 0) or 0)

        with self.protection_lock:
            peak_map[symbol] = peak_roi
            basis_map[symbol] = avg_entry

        persist_step = max(
            float(
                getattr(
                    config,
                    f"{route}_PROFIT_PEAK_PERSIST_STEP_ROI",
                    2 if route == "REVERSAL" else 1,
                )
            ),
            0.1,
        )
        trigger_roi = float(info.get("trigger_roi", 0) or 0)
        should_persist = (
            peak_roi >= saved_peak + persist_step or
            (peak_roi >= trigger_roi > saved_peak) or
            abs(saved_basis - avg_entry) > basis_tolerance
        )

        if should_persist:
            update_position_runtime_fields(
                state,
                symbol,
                {
                    peak_field: round(peak_roi, 2),
                    basis_field: avg_entry,
                    f"{route_key}_profit_floor_roi": info.get("floor_roi"),
                    f"{route_key}_profit_armed": bool(info.get("armed")),
                },
            )

        if not info.get("should_exit"):
            return False

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(
                f"{symbol} {route_key} profit exit deferred | position busy"
            )
            return True

        try:
            with self.protection_lock:
                if symbol in pending:
                    return True

                pending.add(symbol)

            details = get_open_position_details(symbol)
            position_detail = (details or {}).get(symbol)

            if not position_detail:
                log_warning(
                    f"{symbol} {route_key} profit exit skipped | "
                    "live position not found"
                )

                with self.protection_lock:
                    pending.discard(symbol)

                return True

            amount = float(position_detail.get("amount", 0) or 0)
            position_side = position_detail.get("position_side")
            log_warning(
                f"{symbol} {route} PROFIT RETRACE EXIT | "
                f"CURRENT_ROI={info.get('current_roi')}% | "
                f"PEAK_ROI={info.get('peak_roi')}% | "
                f"FLOOR_ROI={info.get('floor_roi')}%"
            )
            closed = close_position_market(
                symbol,
                amount,
                position_side=position_side,
            )

            if closed:
                cancel_open_protection_orders(symbol)
                update_position_runtime_fields(
                    state,
                    symbol,
                    {
                        peak_field: round(peak_roi, 2),
                        basis_field: avg_entry,
                        exit_status_field: "SUBMITTED",
                        f"{route_key}_profit_exit_price": mark_price,
                        f"{route_key}_profit_exit_roi": info.get("current_roi"),
                        f"{route_key}_profit_exit_reason": info.get("reason"),
                    },
                )
                send_telegram_message(
                    f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                    f"{symbol} {route_key} profit protected\n"
                    f"ROI: {info.get('current_roi')}%\n"
                    f"Peak ROI: {info.get('peak_roi')}%\n"
                    f"Protection floor: {info.get('floor_roi')}%"
                )
                return True

            log_error(f"{symbol} {route_key} profit exit order failed")
            update_position_runtime_fields(
                state,
                symbol,
                {exit_status_field: "FAILED"},
            )

            with self.protection_lock:
                pending.discard(symbol)

            return True

        except Exception as e:
            log_error(f"{symbol} {route_key} profit protection error: {e}")

            with self.protection_lock:
                pending.discard(symbol)

            return True

        finally:
            lock.release()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


def calculate_signal_rank(candidate):
    signal = candidate.get("signal")
    analysis = candidate.get("analysis") or {}
    side_data = analysis.get((signal or "").lower(), {}) or {}
    news_context = candidate.get("news_context") or {}
    llm_context = candidate.get("llm_context") or {}
    rank = _safe_float(side_data.get("confidence"), analysis.get("best_confidence", 0))

    rank += _safe_float(side_data.get("quality_score")) * config.SIGNAL_RANKING_QUALITY_WEIGHT
    rank += _safe_float(side_data.get("participation_score")) * config.SIGNAL_RANKING_FLOW_WEIGHT
    rank += _safe_float(side_data.get("smc_score")) * config.SIGNAL_RANKING_SMC_WEIGHT
    rank += _safe_float(side_data.get("regime_score")) * config.SIGNAL_RANKING_REGIME_WEIGHT

    if (side_data.get("trend_timing_rescue") or {}).get("active"):
        rank -= max(
            _safe_float(
                getattr(config, "TREND_TIMING_RESCUE_RANK_PENALTY", 2.5)
            ),
            0
        )

    if (side_data.get("continuation_pullback") or {}).get("active"):
        rank -= max(
            _safe_float(
                getattr(config, "CONTINUATION_PULLBACK_RANK_PENALTY", 1.5)
            ),
            0
        )

    news_action = str(news_context.get("action") or "").upper()
    llm_action = str(llm_context.get("action") or "").upper()
    risk_label = str(llm_context.get("risk_label") or "").lower()

    if news_action == "BOOST":
        rank += 2
    elif news_action == "PENALTY":
        rank -= 2

    if llm_action == "BOOST":
        rank += 2
    elif llm_action == "PENALTY":
        rank -= 2

    if risk_label == "high":
        rank -= 4
    elif risk_label == "medium":
        rank -= 1.5

    return round(rank, 2)


def get_candidate_signal_type(candidate):
    signal = candidate.get("signal")
    analysis = candidate.get("analysis") or {}
    side_data = analysis.get((signal or "").lower(), {}) or {}
    return str(side_data.get("confirmation_type") or "").upper()


def _empty_position_counts():
    return {"total": 0, "buy": 0, "sell": 0}


def get_position_signal_type(trade_state, symbol):
    item = get_position_state(trade_state, symbol) or {}
    signal_type = str(
        item.get("signal_type") or
        item.get("confirmation_type") or
        ""
    ).upper()

    return "REVERSAL" if signal_type == "REVERSAL" else "TREND"


def get_position_pool_counts(trade_state, open_positions):
    pools = {
        "TREND": _empty_position_counts(),
        "REVERSAL": _empty_position_counts(),
    }

    for symbol, amount in (open_positions or {}).items():
        if amount == 0:
            continue

        pool = get_position_signal_type(trade_state, symbol)
        counts = pools[pool]
        counts["total"] += 1

        if amount > 0:
            counts["buy"] += 1
        else:
            counts["sell"] += 1

    return pools


def get_tp1_runner_pool_counts(trade_state, open_positions):
    pools = {
        "TREND": _empty_position_counts(),
        "REVERSAL": _empty_position_counts(),
    }

    if not getattr(config, "TP1_EXTRA_SLOTS_ENABLED", False):
        return pools

    for symbol, amount in (open_positions or {}).items():
        if amount == 0:
            continue

        item = get_position_state(trade_state, symbol) or {}

        if (
            not item.get("multi_tp_active") or
            item.get("multi_tp_stage") not in (
                RUNNER_PENDING,
                RUNNER_ACTIVE,
            )
        ):
            continue

        pool = get_position_signal_type(trade_state, symbol)
        counts = pools[pool]
        counts["total"] += 1

        if amount > 0:
            counts["buy"] += 1
        else:
            counts["sell"] += 1

    return pools


def _limit_with_tp1_capacity(base_limit, runner_count, extra_cap):
    if base_limit is None:
        return None, 0

    if not getattr(config, "TP1_EXTRA_SLOTS_ENABLED", False):
        return base_limit, 0

    earned = min(
        max(int(runner_count or 0), 0),
        max(int(extra_cap or 0), 0),
    )
    return base_limit + earned, earned


def _limit_reached(limit, count):
    return limit is not None and count >= limit


def check_entry_position_limits(
    signal,
    signal_type,
    pool_counts,
    runner_pool_counts=None,
):
    reversal_signal = signal_type == "REVERSAL"
    pool = "REVERSAL" if reversal_signal else "TREND"
    counts = pool_counts.get(pool, _empty_position_counts())

    if reversal_signal:
        total_limit = getattr(config, "REVERSAL_EXTRA_TOTAL_POSITIONS", 0)
        buy_limit = getattr(config, "REVERSAL_EXTRA_BUY_POSITIONS", 0)
        sell_limit = getattr(config, "REVERSAL_EXTRA_SELL_POSITIONS", 0)
    else:
        total_limit = config.MAX_TOTAL_POSITIONS
        buy_limit = config.MAX_BUY_POSITIONS
        sell_limit = config.MAX_SELL_POSITIONS

    runners = (runner_pool_counts or {}).get(
        pool,
        _empty_position_counts(),
    )
    total_limit, earned_total = _limit_with_tp1_capacity(
        total_limit,
        runners["total"],
        getattr(config, "TP1_EXTRA_TOTAL_POSITIONS", 0),
    )
    buy_limit, earned_buy = _limit_with_tp1_capacity(
        buy_limit,
        runners["buy"],
        getattr(config, "TP1_EXTRA_BUY_POSITIONS", 0),
    )
    sell_limit, earned_sell = _limit_with_tp1_capacity(
        sell_limit,
        runners["sell"],
        getattr(config, "TP1_EXTRA_SELL_POSITIONS", 0),
    )

    if _limit_reached(total_limit, counts["total"]):
        return False, (
            f"{pool} MAX POSITIONS REACHED | "
            f"TOTAL={counts['total']}/{total_limit} | "
            f"BUY={counts['buy']} | SELL={counts['sell']}"
        )

    if signal == "BUY" and _limit_reached(buy_limit, counts["buy"]):
        return False, (
            f"{pool} MAX BUY POSITIONS REACHED | "
            f"BUY={counts['buy']}/{buy_limit} | TOTAL={counts['total']}"
        )

    if signal == "SELL" and _limit_reached(sell_limit, counts["sell"]):
        return False, (
            f"{pool} MAX SELL POSITIONS REACHED | "
            f"SELL={counts['sell']}/{sell_limit} | TOTAL={counts['total']}"
        )

    log_info(
        f"{pool} LIMIT OK | TOTAL={counts['total']}/{total_limit if total_limit is not None else 'NA'} | "
        f"BUY={counts['buy']}/{buy_limit if buy_limit is not None else 'NA'} | "
        f"SELL={counts['sell']}/{sell_limit if sell_limit is not None else 'NA'} | "
        f"TP1_EARNED=TOTAL:{earned_total},BUY:{earned_buy},SELL:{earned_sell}"
    )

    return True, ""


def build_entry_candidate(
    symbol,
    signal,
    final_analysis,
    participation,
    trend_df,
    confirm_df,
    entry_df,
    btc_trend,
    btc_corr,
    rs,
    news_context,
    llm_context
):
    candidate = {
        "symbol": symbol,
        "signal": signal,
        "analysis": final_analysis,
        "participation": participation,
        "trend_df": trend_df,
        "confirm_df": confirm_df,
        "entry_df": entry_df,
        "btc_trend": btc_trend,
        "btc_corr": btc_corr,
        "rs": rs,
        "news_context": news_context,
        "llm_context": llm_context,
    }
    candidate["rank_score"] = calculate_signal_rank(candidate)
    return candidate


def execute_entry_candidate(
    candidate,
    trade_state,
    position_details,
    open_positions,
    btc_trend_df,
    dca_monitor
):
    symbol = candidate["symbol"]
    signal = candidate["signal"]
    final_analysis = candidate["analysis"]
    participation = candidate["participation"]
    trend_df = candidate["trend_df"]
    confirm_df = candidate["confirm_df"]
    entry_df = candidate["entry_df"]
    btc_trend = candidate["btc_trend"]
    btc_corr = candidate["btc_corr"]
    rs = candidate["rs"]
    news_context = candidate["news_context"]
    llm_context = candidate["llm_context"]
    signal_type = get_candidate_signal_type(candidate)

    try:
        if shutdown_event.is_set():
            log_warning(f"{symbol} entry skipped | bot shutdown requested")
            return position_details, open_positions, False

        latest_position_details = get_open_position_details()

        if latest_position_details is None:
            log_warning(
                f"{symbol} live position snapshot unavailable; skipping entry"
            )
            return position_details, open_positions, False

        position_details = latest_position_details
        open_positions = get_open_position_amounts(position_details)
        prune_and_cleanup_closed_positions(trade_state, open_positions)
        dca_monitor.sync(position_details)

        if symbol in open_positions:
            run_scan_dca_check(
                symbol,
                position_details[symbol],
                btc_trend_df,
                btc_trend,
                dca_monitor=dca_monitor
            )
            return position_details, open_positions, False

        counts = get_open_position_counts(open_positions)
        log_info(
            f"{symbol} LIVE POSITION COUNT | "
            f"TOTAL={counts['total']} | BUY={counts['buy']} | SELL={counts['sell']}"
        )

        pool_counts = get_position_pool_counts(trade_state, open_positions)
        runner_pool_counts = get_tp1_runner_pool_counts(
            trade_state,
            open_positions,
        )
        limit_ok, limit_reason = check_entry_position_limits(
            signal,
            signal_type,
            pool_counts,
            runner_pool_counts,
        )

        if not limit_ok:
            log_warning(limit_reason)
            return position_details, open_positions, False

        side_analysis = final_analysis.get(signal.lower(), {})
        reversal_futures = (
            (side_analysis.get("reversal_context") or {}).get(
                "futures_confirmation",
                {},
            )
        )

        if (
            signal_type == "REVERSAL" and
            getattr(config, "REVERSAL_REQUIRE_FUTURES_CONFIRMATION", True) and
            not reversal_futures.get("active")
        ):
            reason = "; ".join(
                reversal_futures.get("reasons", [])
            ) or "REVERSAL_FUTURES_CONFIRMATION_MISSING"
            log_warning(f"{symbol} REVERSAL ENTRY BLOCKED | {reason}")
            append_signal_journal(
                symbol,
                final_analysis,
                participation,
                trend_df,
                confirm_df,
                entry_df,
                btc_trend,
                btc_corr,
                rs,
                action="SKIPPED_REVERSAL_FUTURES",
                skip_reason=reason,
                news_context=news_context,
                llm_context=llm_context
            )
            return position_details, open_positions, False

        timing_rescue = side_analysis.get("trend_timing_rescue") or {}
        continuation_pullback = (
            side_analysis.get("continuation_pullback") or {}
        )
        timing_rescue_active = bool(timing_rescue.get("active"))
        continuation_pullback_active = bool(
            continuation_pullback.get("active")
        )
        require_both_live = (
            (
                timing_rescue_active and
                bool(
                    getattr(
                        config,
                        "TREND_TIMING_RESCUE_REQUIRE_BOTH_LIVE_TIMEFRAMES",
                        True
                    )
                )
            ) or
            (
                continuation_pullback_active and
                bool(
                    getattr(
                        config,
                        "CONTINUATION_PULLBACK_REQUIRE_BOTH_LIVE_TIMEFRAMES",
                        True
                    )
                )
            ) or
            (
                signal_type == "REVERSAL" and
                bool(
                    getattr(
                        config,
                        "REVERSAL_REQUIRE_BOTH_LIVE_TIMEFRAMES",
                        True
                    )
                )
            )
        )
        current_price = entry_df["close"].iloc[-2]

        if timing_rescue_active:
            log_info(
                f"{symbol} TREND TIMING RESCUE EXECUTION | "
                f"MISSED={timing_rescue.get('missed_module')} | "
                f"REQUIRE_BOTH_LIVE={require_both_live}"
            )

        if continuation_pullback_active:
            log_info(
                f"{symbol} CONTINUATION PULLBACK EXECUTION | "
                f"EMA20_DISTANCE_ATR="
                f"{continuation_pullback.get('ema20_distance_atr')} | "
                f"REQUIRE_BOTH_LIVE={require_both_live}"
            )

        if config.LIVE_ENTRY_CONFIRMATION_ENABLED:
            guard_ok, current_price, guard_info = check_live_entry_guard(
                symbol,
                signal,
                current_price,
                require_both_override=True if require_both_live else None
            )

            if not guard_ok:
                log_live_guard_block(symbol, guard_info)
                append_signal_journal(
                    symbol,
                    final_analysis,
                    participation,
                    trend_df,
                    confirm_df,
                    entry_df,
                    btc_trend,
                    btc_corr,
                    rs,
                    action="SKIPPED_LIVE_GUARD",
                    skip_reason=guard_info.get("reason"),
                    news_context=news_context,
                    llm_context=llm_context
                )
                return position_details, open_positions, False

            log_info(
                f"{symbol} LIVE ENTRY GUARD OK | "
                f"MARK={current_price} | {guard_info.get('reason')}"
            )

        min_room_override = None

        if side_analysis.get("confirmation_type") == "REVERSAL":
            min_room_override = config.REVERSAL_MIN_TP_ROOM_ROI

        room_ok, room_info = validate_entry_profit_room(
            signal,
            current_price,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE,
            min_roi_override=min_room_override
        )

        if not room_ok:
            log_warning(f"{symbol} SKIP | {room_info.get('reason')}")
            append_signal_journal(
                symbol,
                final_analysis,
                participation,
                trend_df,
                confirm_df,
                entry_df,
                btc_trend,
                btc_corr,
                rs,
                action="SKIPPED_PROFIT_ROOM",
                skip_reason=room_info.get("reason"),
                news_context=news_context,
                llm_context=llm_context
            )
            return position_details, open_positions, False

        log_profit_room_ok(symbol, signal, room_info)

        adverse_reversal_df = get_adverse_reversal_frame(symbol, trend_df)

        if adverse_reversal_df is None:
            reason = (
                f"ADVERSE REVERSAL {config.ADVERSE_REVERSAL_TIMEFRAME} "
                "FRAME UNAVAILABLE"
            )
            log_warning(f"{symbol} SKIP | {reason}")
            return position_details, open_positions, False

        level_ok, level_info = validate_adverse_zone_level(
            signal,
            current_price,
            adverse_reversal_df,
            confirm_df,
            leverage=config.LEVERAGE
        )

        if not level_ok:
            log_warning(f"{symbol} SKIP | {level_info.get('reason')}")
            return position_details, open_positions, False

        reference_price = level_info["level"]
        adverse_roi = level_info["adverse_roi"]
        max_adverse_roi = level_info.get("max_adverse_roi")
        safety_timeframe = level_info.get("safety_timeframe", config.TREND_TIMEFRAME)
        level_label = "SUPPORT" if signal == "BUY" else "RESISTANCE"

        log_info(
            f"{symbol} {level_label} SAFETY LEVEL | "
            f"PRICE={reference_price} | ROI={adverse_roi}% | "
            f"MAX_ROI={max_adverse_roi}% | TF={safety_timeframe} | "
            f"SCORE={level_info['score']} | SRC={level_info['source']}"
        )

        news_ok, final_analysis, news_context = apply_news_filter(
            symbol,
            signal,
            final_analysis
        )
        signal = final_analysis["signal"]

        if not news_ok:
            log_warning(
                f"{symbol} SKIP | {news_context.get('reason')} | "
                f"NEWS={news_context.get('label')} "
                f"SCORE={news_context.get('score')}"
            )
            append_signal_journal(
                symbol,
                final_analysis,
                participation,
                trend_df,
                confirm_df,
                entry_df,
                btc_trend,
                btc_corr,
                rs,
                action="SKIPPED_NEWS_FILTER",
                skip_reason=news_context.get("reason"),
                news_context=news_context
            )
            return position_details, open_positions, False

        llm_ok, final_analysis, llm_context = apply_llm_filter(
            symbol,
            signal,
            final_analysis,
            participation=participation,
            btc_trend=btc_trend,
            btc_corr=btc_corr,
            rs=rs,
            news_context=news_context,
            prefetched_review=candidate.get("llm_prefetched_review"),
            prefetched_source=candidate.get("llm_prefetched_source", "")
        )
        signal = final_analysis["signal"]

        if not llm_ok:
            log_warning(
                f"{symbol} SKIP | {llm_context.get('reason')} | "
                f"LLM={llm_context.get('action')} "
                f"RISK={llm_context.get('risk_label')}"
            )
            append_signal_journal(
                symbol,
                final_analysis,
                participation,
                trend_df,
                confirm_df,
                entry_df,
                btc_trend,
                btc_corr,
                rs,
                action="SKIPPED_LLM_FILTER",
                skip_reason=llm_context.get("reason"),
                news_context=news_context,
                llm_context=llm_context
            )
            return position_details, open_positions, False

        log_info(
            f"{symbol} FINAL CONTEXT OK | "
            f"NEWS={news_context.get('action')} "
            f"{news_context.get('label')} "
            f"SCORE={news_context.get('score')} | "
            f"LLM={llm_context.get('action')} "
            f"{llm_context.get('risk_label')} "
            f"ADJ={llm_context.get('confidence_adjustment')}"
        )

        balance = get_balance()
        initial_margin = get_initial_trade_margin()
        quantity = calculate_position_size(
            balance,
            current_price,
            reference_price,
            symbol,
            initial_margin
        )
        notional = quantity * current_price
        log_info(f"{symbol} QTY={quantity} | NOTIONAL={notional:.2f}")

        if quantity <= 0:
            log_warning(f"{symbol} SKIPPED | INVALID QTY")
            return position_details, open_positions, False

        notional_ok, notional = validate_min_notional(
            symbol,
            quantity,
            current_price
        )

        if not notional_ok:
            log_warning(f"{symbol} SKIP | NOTIONAL TOO LOW: {notional}")
            return position_details, open_positions, False

        if not set_margin_type(symbol):
            return position_details, open_positions, False

        if not setup_leverage(symbol):
            return position_details, open_positions, False

        if shutdown_event.is_set():
            log_warning(f"{symbol} entry order skipped | bot shutdown requested")
            return position_details, open_positions, False

        side = SIDE_BUY if signal == "BUY" else SIDE_SELL
        order = place_market_order(symbol, side, quantity)

        if not order:
            return position_details, open_positions, False

        entry_price = get_entry_price(symbol, order)

        if entry_price <= 0:
            entry_price = current_price
            log_warning(
                f"{symbol} ENTRY PRICE UNAVAILABLE | USING CURRENT PRICE FOR TP"
            )

        structure_tp = None

        if not config.STATIC_TP_ENABLED:
            tp_ok, structure_tp = validate_structure_take_profit(
                signal,
                entry_price,
                trend_df,
                confirm_df,
                leverage=config.LEVERAGE
            )

            if tp_ok:
                log_info(
                    f"{symbol} STRUCTURE TP | "
                    f"TARGET={structure_tp['target_price']} | "
                    f"RAW_LEVEL={structure_tp['raw_level']} | "
                    f"ROI={structure_tp['target_roi']}% | "
                    f"SRC={structure_tp['source']}"
                )
            else:
                log_warning(
                    f"{symbol} {structure_tp['reason']} | USING FALLBACK ROI TP"
                )

        protection_result = place_tp_sl_with_recovery(
            symbol,
            side,
            entry_price,
            quantity,
            confirm_df,
            structure_tp=structure_tp,
            signal_type=signal_type,
            context_label="ENTRY",
            enable_multi_tp=bool(
                getattr(config, "MULTI_TP_ENABLED", False)
            ),
            return_details=True
        )
        protection_ok = bool(protection_result.get("ok"))

        if not protection_ok:
            log_warning(f"{symbol} TP ORDER NOT CREATED")

        trade_times[symbol] = {
            "entry_time": datetime.now(),
            "side": signal
        }
        position_state = create_position_state(
            symbol,
            signal,
            entry_price,
            quantity,
            config.MARGIN_PER_TRADE,
            initial_margin,
            reference_price,
            level_info
        )
        position_state["signal_type"] = signal_type or "UNKNOWN"
        position_state["confirmation_type"] = signal_type or "UNKNOWN"
        position_state["tp_status"] = "CREATED" if protection_ok else "FAILED"
        position_state["tp_price"] = protection_result.get("tp_price")
        position_state["tp_mode"] = protection_result.get("tp_mode")
        position_state["tp_context"] = "ENTRY"
        position_state["sl_status"] = (
            "CREATED"
            if protection_result.get("sl_created")
            else "DISABLED"
        )
        position_state["sl_enabled"] = bool(protection_result.get("sl_created"))
        position_state["sl_price"] = protection_result.get("sl_price")
        position_state["sl_source"] = "ENTRY"
        apply_multi_tp_protection_state(position_state, protection_result)
        position_state["tp_updated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        upsert_position_state(trade_state, symbol, position_state)
        append_signal_journal(
            symbol,
            final_analysis,
            participation,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="TRADE_OPENED",
            news_context=news_context,
            llm_context=llm_context
        )

        log_info(
            f"*** {symbol} TRADE OPENED ***\n"
            f"ENTRY: {entry_price}\n"
            f"{level_label}: {reference_price}\n"
            f"ADVERSE ROI TO LEVEL: {adverse_roi}%\n"
            f"SAFETY ROI LIMIT: {max_adverse_roi}%\n"
            f"SAFETY TIMEFRAME: {safety_timeframe}\n"
            f"SL: {'ENABLED' if protection_result.get('sl_enabled') else 'DISABLED'} "
            f"({signal_type or 'UNKNOWN'})\n"
            f"BALANCE: {balance}\n"
        )
        send_order_opened_message(
            symbol,
            signal,
            entry_price,
            quantity,
            initial_margin,
            protection_result,
            final_analysis,
            news_context,
            llm_context
        )

        open_positions[symbol] = quantity if signal == "BUY" else -quantity
        order_counts = get_open_position_counts(open_positions)
        latest_position_details = get_open_position_details()

        if latest_position_details is not None:
            position_details = latest_position_details
            open_positions = get_open_position_amounts(position_details)
            dca_monitor.sync(latest_position_details)

        log_info(
            f"{symbol} OPENED | TOTAL={order_counts['total']} | "
            f"BUY={order_counts['buy']} | SELL={order_counts['sell']}"
        )

        if config.POST_TRADE_SLEEP_SECONDS > 0:
            time.sleep(config.POST_TRADE_SLEEP_SECONDS)

        return position_details, open_positions, True

    except Exception as e:
        log_error(f"{symbol} ENTRY EXECUTION ERROR: {e}")
        return position_details, open_positions, False


def process_ranked_entry_candidates(
    candidates,
    trade_state,
    position_details,
    open_positions,
    btc_trend_df,
    dca_monitor
):
    if not candidates:
        return position_details, open_positions

    ranked = sorted(
        candidates,
        key=lambda item: item.get("rank_score", 0),
        reverse=True
    )

    if config.SIGNAL_RANKING_MAX_CANDIDATES > 0:
        ranked = ranked[:config.SIGNAL_RANKING_MAX_CANDIDATES]

    log_info(f"SIGNAL RANKING | CANDIDATES={len(ranked)}")
    prefetch_llm_candidate_reviews(ranked)

    for index, candidate in enumerate(ranked, start=1):
        if shutdown_event.is_set():
            log_warning("Signal ranking stopped | bot shutdown requested")
            break

        log_info(
            f"RANK {index}/{len(ranked)} | "
            f"{candidate['symbol']} {candidate['signal']} | "
            f"SCORE={candidate.get('rank_score')}"
        )
        position_details, open_positions, _ = execute_entry_candidate(
            candidate,
            trade_state,
            position_details,
            open_positions,
            btc_trend_df,
            dca_monitor
        )

    return position_details, open_positions


def finalize_scanned_symbol(
    scan_item,
    signal_candidates,
    trade_state,
    position_details,
    open_positions,
    btc_trend_df,
    dca_monitor
):
    symbol = scan_item["symbol"]
    final_analysis = scan_item["analysis"]
    participation = scan_item.get("participation")
    trend_df = scan_item["trend_df"]
    confirm_df = scan_item["confirm_df"]
    entry_df = scan_item["entry_df"]
    btc_trend = scan_item["btc_trend"]
    btc_corr = scan_item["btc_corr"]
    rs = scan_item["rs"]

    log_signal_analysis(final_analysis)
    signal = final_analysis["signal"]

    if not signal:
        exhaustion_blocked = bool(
            final_analysis.get("trend_exhaustion_blocked")
        )
        no_signal_reason = (
            "TREND_EXHAUSTION_GUARD"
            if exhaustion_blocked
            else "NO_FINAL_SIGNAL"
        )
        append_signal_journal(
            symbol,
            final_analysis,
            participation,
            trend_df,
            confirm_df,
            entry_df,
            btc_trend,
            btc_corr,
            rs,
            action="NO_SIGNAL",
            skip_reason=no_signal_reason
        )
        log_warning(
            f"{symbol} NO SIGNAL | REASON={no_signal_reason} | "
            f"BTC={btc_trend} | CORR={btc_corr} | RS={rs}"
        )
        return position_details, open_positions

    candidate = build_entry_candidate(
        symbol,
        signal,
        final_analysis,
        participation,
        trend_df,
        confirm_df,
        entry_df,
        btc_trend,
        btc_corr,
        rs,
        {},
        {}
    )

    if config.SIGNAL_RANKING_ENABLED:
        signal_candidates.append(candidate)
        log_info(
            f"{symbol} TECHNICAL SIGNAL QUEUED | "
            f"RANK_SCORE={candidate['rank_score']}"
        )
        return position_details, open_positions

    position_details, open_positions, _ = execute_entry_candidate(
        candidate,
        trade_state,
        position_details,
        open_positions,
        btc_trend_df,
        dca_monitor
    )
    return position_details, open_positions


def run_bot():

    log_info("BOT STARTED")
    scan_symbols = get_scan_symbols()
    log_info(
        f"Scanning {len(scan_symbols)} symbols | "
        f"KLINE_LIMIT={config.KLINE_LIMIT} | "
        f"THROTTLE={config.REQUEST_THROTTLE_SECONDS}s"
    )
    log_active_dca_config()
    log_info(
        "TP1 EXTRA SLOT CONFIG | "
        f"ENABLED={getattr(config, 'TP1_EXTRA_SLOTS_ENABLED', False)} | "
        f"TOTAL_CAP={getattr(config, 'TP1_EXTRA_TOTAL_POSITIONS', 0)} | "
        f"BUY_CAP={getattr(config, 'TP1_EXTRA_BUY_POSITIONS', 0)} | "
        f"SELL_CAP={getattr(config, 'TP1_EXTRA_SELL_POSITIONS', 0)}"
    )
    dca_monitor = DcaWebsocketMonitor()
    dca_monitor.start()
    target_margin_monitor = TargetMarginBalanceMonitor()
    target_margin_monitor.start()

    try:
        while not shutdown_event.is_set():
            try:
                position_details = get_open_position_details()

                if position_details is None:
                    log_warning("Position snapshot unavailable; skipping this scan")
                    wait_for_next_scan("POSITION_SNAPSHOT_UNAVAILABLE")
                    continue

                open_positions = get_open_position_amounts(position_details)
                trade_state = load_trade_state()
                prune_and_cleanup_closed_positions(trade_state, open_positions)
                log_closed_trades(open_positions)
                dca_monitor.sync(position_details)

                btc_trend_df, btc_trend = get_cached_btc_context()
                log_info(f"BTC TREND: {btc_trend}")

                for open_symbol, position_detail in position_details.items():
                    try:
                        ensure_reversal_stop_loss(
                            open_symbol,
                            position_detail,
                            trade_state,
                            btc_trend_df,
                        )
                    except Exception as e:
                        log_error(
                            f"{open_symbol} reversal SL reconcile error: {e}"
                        )

                futures_context_queue = []
                signal_candidates = []
                begin_llm_scan_budget()

                for symbol in scan_symbols:
                    if shutdown_event.is_set():
                        log_warning("Scan stopped | bot shutdown requested")
                        break

                    try:
                        log_info(f"Checking {symbol}")

                        if symbol in open_positions:
                            run_scan_dca_check(
                                symbol,
                                position_details[symbol],
                                btc_trend_df,
                                btc_trend,
                                dca_monitor=dca_monitor
                            )
                            continue

                        trend_df, confirm_df, entry_df = get_signal_frames(
                            symbol,
                            btc_trend_df
                        )

                        if trend_df is None or confirm_df is None or entry_df is None:
                            continue

                        btc_corr, rs = calculate_btc_context(
                            symbol,
                            trend_df,
                            btc_trend_df
                        )
                        log_info(f"{symbol} BTC CORR: {btc_corr}")
                        log_info(f"{symbol} RS: {rs}%")

                        base_analysis = analyze_signal_cached(
                            trend_df,
                            confirm_df,
                            entry_df,
                            btc_trend,
                            btc_corr,
                            rs,
                            log_details=False,
                            cache_namespace=symbol,
                        )
                        scan_item = {
                            "symbol": symbol,
                            "analysis": base_analysis,
                            "participation": None,
                            "trend_df": trend_df,
                            "confirm_df": confirm_df,
                            "entry_df": entry_df,
                            "btc_trend": btc_trend,
                            "btc_corr": btc_corr,
                            "rs": rs,
                        }

                        if should_fetch_futures_context(base_analysis):
                            scan_item["futures_priority"] = (
                                futures_context_priority(base_analysis)
                            )
                            futures_context_queue.append(scan_item)
                            log_info(
                                f"{symbol} FUTURES CONTEXT QUEUED | "
                                f"PRIORITY={scan_item['futures_priority']}"
                            )
                            continue

                        position_details, open_positions = finalize_scanned_symbol(
                            scan_item,
                            signal_candidates,
                            trade_state,
                            position_details,
                            open_positions,
                            btc_trend_df,
                            dca_monitor
                        )

                    except Exception as e:
                        log_error(f"{symbol} ERROR: {e}")

                if futures_context_queue and not shutdown_event.is_set():
                    futures_context_queue.sort(
                        key=lambda item: item.get("futures_priority", 0),
                        reverse=True
                    )
                    futures_limit = max(
                        int(config.FUTURES_CONTEXT_MAX_SYMBOLS_PER_SCAN),
                        0
                    )
                    selected_count = min(
                        len(futures_context_queue),
                        futures_limit
                    )
                    log_info(
                        f"FUTURES CONTEXT RANKING | "
                        f"ELIGIBLE={len(futures_context_queue)} | "
                        f"SELECTED={selected_count} | LIMIT={futures_limit}"
                    )

                    for index, scan_item in enumerate(
                        futures_context_queue,
                        start=1
                    ):
                        if shutdown_event.is_set():
                            log_warning(
                                "Futures context ranking stopped | "
                                "bot shutdown requested"
                            )
                            break

                        symbol = scan_item["symbol"]

                        try:
                            if index <= futures_limit:
                                participation = (
                                    get_futures_participation(symbol) or {}
                                )
                                scan_item["participation"] = participation
                                log_info(
                                    f"{symbol} FUTURES CONTEXT "
                                    f"RANK={index}/{len(futures_context_queue)} | "
                                    f"PRIORITY={scan_item.get('futures_priority')} | "
                                    f"OI={participation.get('oi_change_pct')}% | "
                                    f"TAKER="
                                    f"{participation.get('taker_buy_sell_ratio')} | "
                                    f"GLOBAL_LS="
                                    f"{participation.get('global_long_short_ratio')} | "
                                    f"TOP_LS="
                                    f"{participation.get('top_long_short_ratio')} | "
                                    f"FUNDING={participation.get('funding_rate')}"
                                )
                                scan_item["analysis"] = analyze_signal(
                                    scan_item["trend_df"],
                                    scan_item["confirm_df"],
                                    scan_item["entry_df"],
                                    scan_item["btc_trend"],
                                    scan_item["btc_corr"],
                                    scan_item["rs"],
                                    participation=participation,
                                    log_details=False
                                )
                            else:
                                log_warning(
                                    f"{symbol} FUTURES CONTEXT SKIPPED | "
                                    f"RANK={index} | "
                                    f"PRIORITY={scan_item.get('futures_priority')} | "
                                    f"SCAN LIMIT={futures_limit}"
                                )

                            position_details, open_positions = (
                                finalize_scanned_symbol(
                                    scan_item,
                                    signal_candidates,
                                    trade_state,
                                    position_details,
                                    open_positions,
                                    btc_trend_df,
                                    dca_monitor
                                )
                            )
                        except Exception as e:
                            log_error(
                                f"{symbol} FUTURES CONTEXT PROCESSING ERROR: {e}"
                            )

                if config.SIGNAL_RANKING_ENABLED and not shutdown_event.is_set():
                    position_details, open_positions = process_ranked_entry_candidates(
                        signal_candidates,
                        trade_state,
                        position_details,
                        open_positions,
                        btc_trend_df,
                        dca_monitor
                    )

                if shutdown_event.is_set():
                    break

                wait_for_next_scan()

            except Exception as e:
                log_error(f"MAIN LOOP ERROR: {e}")
                wait_for_next_scan("MAIN_LOOP_ERROR")

    finally:
        target_margin_monitor.stop()
        dca_monitor.stop()

    log_warning("BOT STOPPED | manual restart required")

if __name__ == "__main__":
    run_bot()
