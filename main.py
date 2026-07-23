import signal
import threading
import time
from datetime import datetime, timedelta

import config

from binance.enums import SIDE_BUY, SIDE_SELL

from exchange import (
    sync_client_time,
    is_one_way_position_mode,
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
    is_known_futures_symbol,
    get_futures_participation,
    get_mark_price,
    get_open_take_profit_info,
    get_open_stop_loss_info,
    find_matching_close_position_stop,
    place_stop_loss_only,
    place_close_position_protection,
    place_partial_take_profit_quantity,
    set_margin_type,
    setup_leverage,
    get_entry_price,
    get_execution_reconciliation,
    get_reconciled_executed_quantity,
    is_reconciled_execution_settled,
    validate_min_notional,
    cancel_open_protection_orders,
    cancel_open_take_profit_orders,
    cancel_algo_order,
    get_price_precision,
    get_private_rest_backoff_remaining,
    get_algo_order_execution,
    find_matching_open_algo_order,
    normalize_trigger_price,
    get_symbol_price_rules,
    get_futures_depth_snapshot,
    reconcile_execution_client_orders,
    get_signal_stop_loss,
)

from indicators import apply_indicators
from strategy import (
    analyze_signal,
    analyze_signal_cached,
    detect_market_structure,
    evaluate_route_early_invalidation,
    evaluate_route_profit_protection,
    evaluate_time_exit_weakness,
    futures_context_priority,
    log_signal_analysis,
    should_fetch_futures_context,
    validate_live_entry_guard,
    validate_adverse_zone_level,
    validate_structure_take_profit,
    validate_entry_profit_room,
    validate_dca_structure_level,
    validate_dca_continuation_guard,
    validate_dca_recovery_confirmation,
)
from volume_profile import record_volume_profile_telemetry
from risk_management import calculate_position_size, get_position_risk_budget
from signal_journal import append_signal_journal
from signal_calibration import calibration_probability
from signal_outcomes import register_signal_outcome, observe_signal_outcomes
from market_intelligence import (
    MarketFlowMonitor,
    build_breadth_sample,
    calculate_market_breadth,
    calculate_regime_transition,
)
from order_flow_shadow import OrderFlowShadowMonitor
from execution_telemetry import (
    flush_execution_telemetry,
    validate_execution_telemetry_path,
)
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
    TradeStateLoadError,
    apply_multi_tp_protection_state,
    clear_dca_reservation,
    clear_confirmed_absent_entry_execution,
    create_position_state,
    get_position_state,
    get_pending_execution,
    has_active_dca_reservation,
    load_trade_state,
    prune_closed_positions,
    record_dca_fill,
    reserve_dca_level,
    update_position_runtime_fields,
    update_position_tp_status,
    upsert_position_state,
    upsert_pending_execution,
    remove_pending_execution,
    remove_position_state,
    trade_state_file_exists,
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
entry_quarantined_symbols = set()


def request_shutdown(signum=None, _frame=None):
    """Request an orderly stop from SIGINT, SIGTERM, or internal controls."""
    signal_name = "INTERNAL"

    if signum is not None:
        try:
            signal_name = signal.Signals(signum).name
        except (TypeError, ValueError):
            signal_name = str(signum)

    if not shutdown_event.is_set():
        log_warning(f"BOT SHUTDOWN REQUESTED | SOURCE={signal_name}")

    shutdown_event.set()


def install_shutdown_signal_handlers():
    try:
        signal.signal(signal.SIGINT, request_shutdown)
        signal.signal(signal.SIGTERM, request_shutdown)
        return True
    except (ValueError, OSError) as exc:
        log_warning(f"Shutdown signal handler unavailable: {exc}")
        return False


def load_runtime_trade_state(open_positions):
    if (
        getattr(config, "REQUIRE_STATE_FOR_OPEN_POSITIONS", True) and
        open_positions and
        not trade_state_file_exists()
    ):
        raise TradeStateLoadError(
            "runtime state is missing while Binance has open positions; "
            "restore DCA_STATE_PATH before starting entries"
        )

    return load_trade_state()


def wait_for_next_scan(reason="SCAN_COMPLETE", wait_seconds_override=None):
    wait_seconds = max(
        float(
            config.SCAN_SLEEP_SECONDS
            if wait_seconds_override is None
            else wait_seconds_override
        ),
        0,
    )
    heartbeat_seconds = max(
        float(getattr(config, "SCAN_WAIT_HEARTBEAT_SECONDS", 60)),
        1,
    )
    safety_poll_seconds = max(
        min(
            float(
                getattr(
                    config,
                    "PENDING_EXECUTION_RECONCILE_SECONDS",
                    5,
                )
            ),
            heartbeat_seconds,
        ),
        0.5,
    )
    wait_started_at = time.monotonic()
    deadline = wait_started_at + wait_seconds
    next_heartbeat_at = wait_started_at + heartbeat_seconds
    next_safety_poll_at = wait_started_at + safety_poll_seconds
    next_scan_at = datetime.now() + timedelta(seconds=wait_seconds)
    next_scan_label = next_scan_at.isoformat(timespec="seconds")
    log_info(
        f"Waiting next scan | REASON={reason} | "
        f"WAIT_SECONDS={round(wait_seconds, 1)} | "
        f"NEXT_SCAN_AT={next_scan_label}"
    )

    while not shutdown_event.is_set():
        now = time.monotonic()
        remaining = deadline - now

        if remaining <= 0:
            log_info("Next scan wait complete | starting scan now")
            return True

        if now >= next_safety_poll_at:
            next_safety_poll_at = now + safety_poll_seconds

            try:
                if state_requires_urgent_safety_retry(load_trade_state()):
                    log_warning(
                        "Urgent durable position-safety work detected; "
                        "ending scan wait early"
                    )
                    return True
            except Exception as exc:
                log_warning(
                    "Urgent position-safety wait check unavailable: "
                    f"{exc}"
                )

        wait_slice = min(
            remaining,
            max(next_heartbeat_at - now, 0.01),
            max(next_safety_poll_at - now, 0.01),
        )

        if shutdown_event.wait(wait_slice):
            return False

        now = time.monotonic()
        remaining = max(deadline - now, 0)

        if remaining > 0 and now >= next_heartbeat_at:
            log_info(
                f"Next scan heartbeat | "
                f"REMAINING_SECONDS={round(remaining, 1)} | "
                f"NEXT_SCAN_AT={next_scan_label}"
            )
            next_heartbeat_at = now + heartbeat_seconds

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
            entry_quarantined_symbols.add(symbol)
            log_warning(
                f"{symbol} closed-position protection cleanup incomplete"
            )
        else:
            entry_quarantined_symbols.discard(symbol)

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
    require_both_override=None,
    confidence=None
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
        require_both_override=require_both_override,
        confidence=confidence
    )

    return guard_ok, current_price, guard_info


def check_dca_recovery_confirmation(symbol, side, mark_price):
    def prepare(raw_df):
        if raw_df is None:
            return None

        enriched = apply_indicators(raw_df)
        minimum = max(int(config.LIVE_ENTRY_STRUCTURE_LOOKBACK) + 3, 1)
        return enriched if enriched is not None and len(enriched) >= minimum else None

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
    return validate_dca_recovery_confirmation(
        side,
        prepare(fast_raw),
        prepare(slow_raw),
        mark_price,
    )


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


def validate_position_management_config():
    errors = []

    if getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False):
        if not getattr(config, "TREND_SL_ENABLED", False):
            errors.append("risk sizing requires TREND_SL_ENABLED=True")

        if not getattr(config, "SL_INVALID_FAILS_PROTECTION_ORDER", False):
            errors.append("risk sizing requires fail-closed invalid SL handling")

        if float(getattr(config, "POSITION_RISK_PCT", 0)) <= 0:
            errors.append("POSITION_RISK_PCT must be positive")

        if (
            getattr(config, "REVERSAL_ENTRY_ENABLED", False) and
            not getattr(config, "REVERSAL_SL_ENABLED", False)
        ):
            errors.append(
                "risk sizing requires REVERSAL_SL_ENABLED=True when reversal "
                "entries are enabled"
            )

    if config.DCA_ENABLED and not getattr(
        config,
        "DCA_FIXED_RISK_ENABLED",
        False,
    ):
        errors.append(
            "V7 DCA requires DCA_FIXED_RISK_ENABLED=True; legacy averaging "
            "is no longer a supported live mode"
        )

    if config.DCA_ENABLED and getattr(config, "DCA_FIXED_RISK_ENABLED", False):
        if not getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False):
            errors.append("fixed-risk recovery requires risk-based sizing")

        if int(config.DCA_MAX_ORDERS) != 1:
            errors.append("fixed-risk recovery requires DCA_MAX_ORDERS=1")

        if not config.DCA_MARGIN_PCTS or not config.DCA_TRIGGER_ROIS:
            errors.append("one recovery margin and trigger must be configured")
        elif (
            float(config.DCA_INITIAL_MARGIN_PCT) <= 0 or
            float(config.DCA_MARGIN_PCTS[0]) <= 0 or
            float(config.DCA_TRIGGER_ROIS[0]) <= 0
        ):
            errors.append(
                "initial margin, recovery margin, and recovery trigger must be "
                "positive"
            )

        if not getattr(config, "DCA_REPRICE_TP_AFTER_FILL", False):
            errors.append("fixed-risk recovery requires TP repricing")

        if not getattr(config, "TP1_RUNNER_DISABLE_DCA", True):
            errors.append(
                "fixed-risk recovery requires TP1_RUNNER_DISABLE_DCA=True"
            )

        if getattr(config, "DCA_MANAGE_EXISTING_POSITIONS", False):
            errors.append(
                "fixed-risk recovery cannot auto-adopt existing positions; "
                "set DCA_MANAGE_EXISTING_POSITIONS=False"
            )

        if not getattr(config, "DCA_REQUIRE_HARD_STOP", True):
            errors.append("fixed-risk recovery requires exact hard-stop verification")

        if not getattr(config, "DCA_RECOVERY_CONFIRMATION_ENABLED", True):
            errors.append(
                "fixed-risk recovery requires arm-and-rebound confirmation"
            )

        if not getattr(config, "DCA_RECOVERY_REQUIRE_DATA", True):
            errors.append(
                "fixed-risk recovery requires fail-closed confirmation data"
            )

        if not getattr(
            config,
            "DCA_RECOVERY_REQUIRE_BOTH_TIMEFRAMES",
            True,
        ):
            errors.append(
                "fixed-risk recovery requires both 5m and 15m confirmations"
            )

        initial_risk = max(float(config.DCA_INITIAL_RISK_PCT), 0)
        recovery_risk = max(float(config.DCA_RECOVERY_RISK_PCT), 0)

        if initial_risk <= 0 or recovery_risk <= 0:
            errors.append("initial and recovery risk allocations must be positive")

        if initial_risk + recovery_risk > 100:
            errors.append("initial plus recovery risk allocations cannot exceed 100%")

        margin_total = max(float(config.DCA_INITIAL_MARGIN_PCT), 0) + sum(
            max(float(value), 0)
            for value in config.DCA_MARGIN_PCTS[:1]
        )

        if margin_total > 100:
            errors.append("initial plus recovery margin cannot exceed 100%")

        trigger_roi = float(config.DCA_TRIGGER_ROIS[0])
        stop_cap = float(getattr(config, "TREND_MAX_SL_ROI", 0))
        max_adverse_roi = max(float(config.DCA_MAX_ADVERSE_ROI), 0)
        rebound_roi = max(
            float(getattr(config, "DCA_RECOVERY_MIN_REBOUND_ROI", 0)),
            0,
        )
        stop_buffer = max(
            float(getattr(config, "DCA_MIN_HARD_STOP_BUFFER_ROI", 0)),
            0,
        )

        if stop_cap > 0 and trigger_roi + stop_buffer >= stop_cap:
            errors.append(
                "trend stop cap must be beyond recovery trigger plus buffer"
            )

        if rebound_roi <= 0:
            errors.append("fixed-risk recovery rebound must be positive")

        if max_adverse_roi and max_adverse_roi < trigger_roi + rebound_roi:
            errors.append(
                "DCA_MAX_ADVERSE_ROI must cover trigger plus recovery rebound"
            )

        if (
            stop_cap > 0 and
            max_adverse_roi > 0 and
            max_adverse_roi + stop_buffer > stop_cap
        ):
            errors.append(
                "trend stop cap must cover maximum adverse ROI plus buffer"
            )

        reversal_stop_cap = float(getattr(config, "REVERSAL_MAX_SL_ROI", 0))

        if (
            getattr(config, "REVERSAL_ENTRY_ENABLED", False) and
            reversal_stop_cap > 0 and
            trigger_roi + stop_buffer >= reversal_stop_cap
        ):
            errors.append(
                "reversal stop cap must be beyond recovery trigger plus buffer"
            )

        if (
            getattr(config, "REVERSAL_ENTRY_ENABLED", False) and
            reversal_stop_cap > 0 and
            max_adverse_roi > 0 and
            max_adverse_roi + stop_buffer > reversal_stop_cap
        ):
            errors.append(
                "reversal stop cap must cover maximum adverse ROI plus buffer"
            )

    if getattr(config, "TIME_EXIT_ENABLED", False):
        if float(getattr(config, "TIME_EXIT_MINUTES", 0)) <= 0:
            errors.append("TIME_EXIT_MINUTES must be positive")

        if (
            getattr(config, "TIME_EXIT_REQUIRE_WEAKNESS", True) and
            not getattr(config, "TIME_EXIT_REQUIRE_DATA", True)
        ):
            errors.append(
                "weakness-confirmed time exit requires TIME_EXIT_REQUIRE_DATA=True"
            )

    for error in errors:
        log_error(f"POSITION MANAGEMENT CONFIG ERROR | {error}")

    return not errors


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


def get_stop_buffer_roi(side, current_price, stop_price):
    if current_price <= 0 or stop_price <= 0:
        return 0

    if side == "BUY":
        distance = current_price - stop_price
    else:
        distance = stop_price - current_price

    if distance <= 0:
        return 0

    return round(
        (distance / current_price) * max(float(config.LEVERAGE), 1) * 100,
        2,
    )


def get_recovery_rebound_roi(side, extreme_price, current_price):
    if extreme_price <= 0 or current_price <= 0:
        return 0

    if side == "BUY":
        move = current_price - extreme_price
    else:
        move = extreme_price - current_price

    return round(
        max(move, 0) / extreme_price * max(float(config.LEVERAGE), 1) * 100,
        2,
    )


def runner_owns_position(position_state):
    if not position_state:
        return False

    stage = position_state.get("multi_tp_stage")
    return stage in (RUNNER_PENDING, RUNNER_ACTIVE)


def tp1_transition_blocks_recovery(position_state):
    """Block new exposure from TP1 touch until runner ownership is resolved."""
    if not position_state:
        return True

    stage = position_state.get("multi_tp_stage")
    return bool(
        runner_owns_position(position_state) or
        (
            stage == TP1_PENDING and
            position_state.get("tp1_trigger_seen_at")
        )
    )


def coordinated_position_management_enabled(position_state):
    if not position_state:
        return False

    if int(position_state.get("campaign_risk_version", 0) or 0) >= 2:
        return True

    return bool(
        getattr(config, "POSITION_MANAGEMENT_LEGACY_ENABLED", False)
    )


def position_exit_blocks_dca(position_state):
    if not position_state:
        return True

    if tp1_transition_blocks_recovery(position_state):
        return True

    management_status = str(
        position_state.get("position_management_status") or "ACTIVE"
    ).upper()

    if management_status != "ACTIVE":
        return True

    if str(position_state.get("tp_reprice_status") or "").upper() in (
        "PENDING",
        "FAILED",
    ):
        return True

    return any(
        position_state.get(field) in (
            "PENDING",
            "SUBMITTED",
            "UNCERTAIN",
            "FAILED",
        )
        for field in (
            "early_invalidation_exit_status",
            "time_exit_status",
            "reversal_profit_exit_status",
            "trend_profit_exit_status",
        )
    )


_POSITION_EXIT_OWNER_FIELDS = {
    "REVERSAL_PROFIT": "reversal_profit_exit_status",
    "TREND_PROFIT": "trend_profit_exit_status",
    "EARLY_INVALIDATION": "early_invalidation_exit_status",
    "TIME": "time_exit_status",
}
_POSITION_EXIT_BLOCKING_STATUSES = {
    "PENDING",
    "SUBMITTED",
    "UNCERTAIN",
    "FAILED",
}


def committed_position_exit_owner(position_state):
    """Return the single durable exit owner for a still-managed position."""
    if not position_state:
        return ""

    persisted_owner = str(
        position_state.get("position_exit_owner") or ""
    ).upper()
    persisted_field = _POSITION_EXIT_OWNER_FIELDS.get(persisted_owner)

    if persisted_field and str(
        position_state.get(persisted_field) or ""
    ).upper() in _POSITION_EXIT_BLOCKING_STATUSES:
        return persisted_owner

    for owner, field in _POSITION_EXIT_OWNER_FIELDS.items():
        if str(position_state.get(field) or "").upper() in (
            _POSITION_EXIT_BLOCKING_STATUSES
        ):
            return owner

    return ""


def get_entry_hard_stop(symbol, side, entry_price, confirm_df, signal_type):
    try:
        precision = get_price_precision(symbol)
        stop_price = get_signal_stop_loss(
            side,
            entry_price,
            confirm_df,
            signal_type,
            precision,
        )
        stop_price = normalize_trigger_price(
            symbol,
            side,
            "STOP_MARKET",
            stop_price,
        )

        if stop_price is None:
            return None

        stop_price = float(stop_price)
        valid = (
            stop_price < entry_price
            if side == SIDE_BUY
            else stop_price > entry_price
        )
        return stop_price if valid else None
    except Exception as exc:
        log_error(f"{symbol} hard-stop planning error: {exc}")
        return None


def get_campaign_risk_at_stop(avg_entry, quantity, stop_price):
    if avg_entry <= 0 or quantity <= 0 or stop_price <= 0:
        return 0

    return abs(avg_entry - stop_price) * abs(quantity)


def get_conservative_risk_equity(wallet_balance=None):
    wallet_balance = (
        float(wallet_balance)
        if wallet_balance is not None
        else float(get_balance() or 0)
    )

    try:
        margin_balance = float(get_margin_balance() or 0)
    except Exception as exc:
        log_warning(f"Risk equity margin-balance lookup failed: {exc}")
        margin_balance = 0

    positive_values = [
        value for value in (wallet_balance, margin_balance) if value > 0
    ]
    return min(positive_values) if positive_values else 0


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


def durable_exit_retry_ready(
    position_state,
    last_attempt_field,
    pending_at_field,
    retry_seconds,
):
    """Return False while a durable exit attempt is still inside its cooldown."""
    last_attempt_at = (
        position_state.get(last_attempt_field) or
        position_state.get(pending_at_field)
    )

    if not last_attempt_at:
        return True

    attempt_age = seconds_since(last_attempt_at)
    return bool(
        attempt_age is not None and
        attempt_age >= max(float(retry_seconds or 0), 1)
    )


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

    if not upsert_position_state(state, symbol, item):
        log_error(f"{symbol} existing position adoption state write failed")
        return None

    log_warning(f"{symbol} existing position adopted into DCA state")
    return item


def get_updated_position_after_fill(symbol, old_avg, old_quantity, fill_price, fill_quantity):
    details = get_open_position_details(symbol, force=True)
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


def refresh_dca_position_before_order(symbol, side, expected_amount):
    details = get_open_position_details(symbol, force=True)

    if details is None:
        return None, "POSITION_SNAPSHOT_UNAVAILABLE"

    detail = details.get(symbol)

    if not detail:
        return None, "POSITION_CLOSED_DURING_DCA_CHECK"

    live_amount = float(detail.get("amount", 0) or 0)
    expected_amount = float(expected_amount or 0)
    live_side = "BUY" if live_amount > 0 else "SELL" if live_amount < 0 else ""

    if live_side != side:
        return None, f"POSITION_SIDE_CHANGED_{live_side or 'FLAT'}"

    tolerance = max(abs(expected_amount) * 1e-9, 1e-12)

    if abs(live_amount - expected_amount) > tolerance:
        return None, (
            f"POSITION_QUANTITY_CHANGED_{expected_amount}_TO_{live_amount}"
        )

    return detail, "OK"


def verify_post_dca_position(symbol, side, pre_amount, executed_quantity):
    details = get_open_position_details(symbol, force=True)

    if details is None:
        return None, "POST_DCA_POSITION_SNAPSHOT_UNAVAILABLE"

    detail = details.get(symbol)

    if not detail:
        return None, "POST_DCA_POSITION_FLAT"

    live_amount = float(detail.get("amount", 0) or 0)
    expected_amount = float(pre_amount or 0) + (
        float(executed_quantity or 0)
        if side == "BUY"
        else -float(executed_quantity or 0)
    )
    live_side = "BUY" if live_amount > 0 else "SELL" if live_amount < 0 else ""
    tolerance = max(abs(expected_amount) * 1e-6, 1e-10)

    if live_side != side:
        return detail, f"POST_DCA_SIDE_{live_side or 'FLAT'}"

    if abs(live_amount - expected_amount) > tolerance:
        return detail, (
            f"POST_DCA_QUANTITY_EXPECTED_{expected_amount}_LIVE_{live_amount}"
        )

    return detail, "OK"


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
    sl_price_override=None,
    preserve_existing_sl=False,
    return_details=True
):
    attempts = max(int(config.TP_ORDER_RETRY_ATTEMPTS), 1)
    last_result = {}
    stop_state_uncertain = False

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
            sl_price_override=sl_price_override,
            preserve_existing_sl=preserve_existing_sl,
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

        if last_result.get("sl_created") and last_result.get("sl_price"):
            # The hard stop is already live. Reuse it on TP retries instead of
            # creating duplicate close-all stops or removing protection.
            preserve_existing_sl = True
            sl_price_override = last_result.get("sl_price")
        elif last_result.get("sl_enabled") and last_result.get("sl_price"):
            # A lost acknowledgement can mean Binance accepted the stop even
            # though the request raised locally. Query before any retry so a
            # second close-all stop is never submitted blindly.
            matched_stop = find_matching_close_position_stop(
                symbol,
                side,
                last_result.get("sl_price"),
                position_side=position_side,
            )

            if matched_stop is None:
                stop_state_uncertain = True
                log_error(
                    f"{symbol} TP recovery stopped | hard-stop lookup "
                    "unavailable after failed placement"
                )
            elif matched_stop:
                preserve_existing_sl = True
                sl_price_override = last_result.get("sl_price")

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

        if stop_state_uncertain:
            break

        if attempt < attempts and config.TP_ORDER_RETRY_DELAY_SECONDS > 0:
            time.sleep(config.TP_ORDER_RETRY_DELAY_SECONDS)

    if (
        config.TP_FAILURE_FALLBACK_ROI_ENABLED
        and roi_override is None
        and not last_result.get("protection_cleanup_failed")
        and not stop_state_uncertain
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
            sl_price_override=sl_price_override,
            preserve_existing_sl=preserve_existing_sl,
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


def fail_safe_close_unprotected_position(
    symbol,
    position_side=None,
    reference_price=None,
    context="PROTECTION_FAILURE",
):
    """Flatten a position whose protection or durable state could not be secured."""
    if not getattr(config, "PROTECTION_FAILURE_CLOSE_ENABLED", True):
        log_error(
            f"{symbol} protection fail-safe close disabled | CONTEXT={context}"
        )
        return False

    details = get_open_position_details(symbol, force=True)

    if details is None:
        log_error(
            f"{symbol} protection fail-safe aborted | "
            f"position snapshot unavailable | CONTEXT={context}"
        )
        return False

    position_detail = details.get(symbol)

    if not position_detail:
        if not cancel_open_protection_orders(symbol):
            entry_quarantined_symbols.add(symbol)
            log_error(
                f"{symbol} protection fail-safe found position flat but "
                f"protection cleanup was not verified | CONTEXT={context}"
            )
            return False

        log_warning(
            f"{symbol} protection fail-safe found position already closed | "
            f"CONTEXT={context}"
        )
        entry_quarantined_symbols.discard(symbol)
        return True

    live_amount = float(position_detail.get("amount", 0) or 0)
    live_position_side = (
        position_detail.get("position_side") or position_side
    )
    closed = close_position_market(
        symbol,
        live_amount,
        position_side=live_position_side,
        reference_price=(
            reference_price or position_detail.get("mark_price")
        ),
        context=f"FAILSAFE_{context}",
    )

    if not closed:
        log_error(
            f"{symbol} PROTECTION FAIL-SAFE CLOSE NOT CONFIRMED | "
            f"CONTEXT={context} | existing exchange protection retained"
        )
        return False

    if not cancel_open_protection_orders(symbol):
        entry_quarantined_symbols.add(symbol)
        log_error(
            f"{symbol} fail-safe close confirmed but protection cleanup was "
            "not verified | CONTEXT={context}"
        )
        return False

    log_warning(
        f"{symbol} protection fail-safe close confirmed | CONTEXT={context}"
    )
    entry_quarantined_symbols.discard(symbol)
    return True


_INTERRUPTED_DCA_SUBMISSION_PHASES = {
    "READY_TO_SUBMIT",
    "ORDER_RETURNED",
    "FAIL_CLOSE_PENDING",
}


def get_interrupted_submission(position_state):
    """Return durable ENTRY/DCA submit-boundary ownership, if unresolved."""
    position_state = position_state or {}

    for field_name, default_context in (
        ("pending_submission", "ENTRY"),
        ("pending_dca", "DCA"),
    ):
        submission = dict(position_state.get(field_name) or {})
        phase = str(submission.get("submission_phase") or "").upper()

        if phase in _INTERRUPTED_DCA_SUBMISSION_PHASES:
            return {
                "field_name": field_name,
                "context": str(
                    submission.get("context") or default_context
                ).upper(),
                "phase": phase,
                "submission": submission,
            }

    return None


def interrupted_dca_submission(position_state):
    # Kept as a compatibility name; this now covers both ENTRY and DCA
    # submit-boundary markers.
    return get_interrupted_submission(position_state) is not None


def configured_entry_symbol_scope():
    symbols = list(dict.fromkeys(getattr(config, "SYMBOLS", []) or []))
    max_symbols = int(getattr(config, "MAX_SCAN_SYMBOLS", 0) or 0)

    if max_symbols > 0:
        symbols = symbols[:max_symbols]

    return set(symbols)


def persist_entry_submission_marker(
    state,
    symbol,
    side,
    requested_quantity,
    reference_price,
    hard_stop_price,
    signal_type,
):
    now = datetime.now().isoformat(timespec="seconds")
    submission = {
        "context": "ENTRY",
        "submission_phase": "READY_TO_SUBMIT",
        "requested_quantity": float(requested_quantity or 0),
        "reference_price": float(reference_price or 0),
        "hard_stop_price": float(hard_stop_price or 0),
        "signal_type": str(signal_type or "").upper(),
        "created_at": now,
    }
    marker = {
        "symbol": symbol,
        "managed_by_bot": True,
        "side": str(side or "").upper(),
        "avg_entry": float(reference_price or 0),
        "initial_entry": float(reference_price or 0),
        "initial_quantity": 0.0,
        "opened_at": now,
        "confirmation_type": str(signal_type or "").upper(),
        "signal_type": str(signal_type or "").upper(),
        "hard_stop_price": float(hard_stop_price or 0),
        "campaign_stop_price": float(hard_stop_price or 0),
        "pending_submission": submission,
        "position_management_status": "ENTRY_READY_TO_SUBMIT",
    }
    return upsert_position_state(state, symbol, marker)


def submit_entry_order_with_marker(
    state,
    symbol,
    side,
    requested_quantity,
    reference_price,
    hard_stop_price,
    signal_type,
):
    """Persist ownership before crossing the external order boundary."""
    if not persist_entry_submission_marker(
        state,
        symbol,
        side,
        requested_quantity,
        reference_price,
        hard_stop_price,
        signal_type,
    ):
        entry_quarantined_symbols.add(symbol)
        log_error(
            f"{symbol} entry blocked | pre-submit ownership marker "
            "could not be persisted"
        )
        shutdown_event.set()
        return None

    order = place_market_order(
        symbol,
        side,
        requested_quantity,
        pre_position_amount=0,
        pre_average_price=0,
        reference_price=reference_price,
        context="ENTRY",
    )

    if not order:
        entry_quarantined_symbols.add(symbol)
        log_error(
            f"{symbol} entry returned without a settled order record | "
            "durable submit marker retained for position reconciliation"
        )

    return order


def retain_entry_close_retry(
    state,
    symbol,
    order,
    side,
    requested_quantity,
    reference_price,
    signal_type,
    hard_stop_price,
    context,
):
    """Retain ambiguous/unsafe entry ownership without stopping its retry loop."""
    entry_quarantined_symbols.add(symbol)
    persisted = persist_pending_execution(
        state,
        symbol,
        order,
        side,
        requested_quantity,
        0,
        reference_price,
        context=context,
        signal_type=signal_type,
        hard_stop_price=hard_stop_price,
    )

    if not persisted:
        log_error(
            f"{symbol} {context} pending update was not persisted; "
            "the original pre-submit ownership marker remains for retry"
        )

    return persisted


def state_requires_urgent_safety_retry(state):
    if (state or {}).get("pending_executions"):
        return True

    return any(
        interrupted_dca_submission(item) or
        str(item.get("position_management_status") or "").upper() ==
        "UNTRACKED_FAIL_CLOSE_PENDING"
        for item in (state or {}).get("positions", {}).values()
    )


def persist_dca_fail_close_pending(state, symbol, reason):
    position_state = get_position_state(state, symbol) or {}
    pending_dca = dict(position_state.get("pending_dca") or {})
    pending_dca.update({
        "interrupted_original_phase": str(
            pending_dca.get("interrupted_original_phase") or
            pending_dca.get("submission_phase") or
            "ORDER_RETURNED"
        ).upper(),
        "submission_phase": "FAIL_CLOSE_PENDING",
        "fail_close_reason": str(reason or "DCA_FAIL_CLOSE_PENDING"),
    })
    return update_position_runtime_fields(
        state,
        symbol,
        {
            "pending_dca": pending_dca,
            "dca_recovery_disabled": True,
            "dca_recovery_disabled_reason": str(reason or "DCA_FAIL_CLOSE_PENDING"),
            "position_management_status": "INTERRUPTED_DCA_FAIL_CLOSE_PENDING",
        },
    )


def fail_close_post_dca_safety_violation(
    state,
    symbol,
    reason,
    position_detail,
    reference_price,
):
    """Keep a post-fill DCA violation quarantined until flat is verified."""
    entry_quarantined_symbols.add(symbol)
    marker_saved = persist_dca_fail_close_pending(state, symbol, reason)
    closed = False

    if position_detail:
        closed = fail_safe_close_unprotected_position(
            symbol,
            position_side=position_detail.get("position_side"),
            reference_price=reference_price,
            context=reason,
        )

    if closed:
        if not remove_position_state(state, symbol):
            shutdown_event.set()
        return True

    if not marker_saved:
        log_error(
            f"{symbol} {reason} marker and first close attempt both failed; "
            "retaining quarantine for retry"
        )

    return False


def reconcile_untracked_open_positions(position_details, state):
    """Fail-close live exposure that has no durable V7 ownership record."""
    attempted = False
    unresolved = set()
    entry_scope = configured_entry_symbol_scope()

    for symbol, detail in (position_details or {}).items():
        # V7 cannot have originated a new entry outside its configured scan
        # universe. Never infer ownership of a manual/legacy account position.
        if symbol not in entry_scope:
            continue

        position_state = get_position_state(state, symbol)
        is_untracked_marker = bool(
            position_state and
            str(position_state.get("position_management_status") or "").upper()
            == "UNTRACKED_FAIL_CLOSE_PENDING"
        )

        if (
            get_pending_execution(state, symbol) or
            (position_state and not is_untracked_marker)
        ):
            continue

        attempted = True
        marker_saved = True

        if not position_state:
            marker = {
                "symbol": symbol,
                "managed_by_bot": False,
                "side": str(detail.get("side") or "").upper(),
                "avg_entry": float(detail.get("entry_price", 0) or 0),
                "initial_entry": float(detail.get("entry_price", 0) or 0),
                "initial_quantity": abs(float(detail.get("amount", 0) or 0)),
                "opened_at": datetime.now().isoformat(timespec="seconds"),
                "position_management_status": "UNTRACKED_FAIL_CLOSE_PENDING",
                "untracked_detected_at": datetime.now().isoformat(
                    timespec="seconds"
                ),
                "untracked_reason": "OPEN_POSITION_WITHOUT_DURABLE_STATE",
            }

            marker_saved = upsert_position_state(state, symbol, marker)

            if not marker_saved:
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} untracked-position safety marker could not "
                    "be persisted; emergency close will still be attempted"
                )

        entry_quarantined_symbols.add(symbol)

        if not getattr(config, "UNTRACKED_POSITION_FAIL_CLOSE_ENABLED", True):
            log_error(
                f"{symbol} untracked live position requires manual close; "
                "automatic fail-close is disabled"
            )
            shutdown_event.set()
            unresolved.add(symbol)
            continue

        closed = fail_safe_close_unprotected_position(
            symbol,
            position_side=detail.get("position_side"),
            reference_price=detail.get("mark_price") or detail.get("entry_price"),
            context="UNTRACKED_OPEN_POSITION",
        )

        if closed:
            if not remove_position_state(state, symbol):
                entry_quarantined_symbols.add(symbol)
                shutdown_event.set()
                unresolved.add(symbol)
            elif not marker_saved:
                # The emergency close succeeded, but a state write failure is
                # still a process-level integrity fault.
                shutdown_event.set()
        else:
            unresolved.add(symbol)

            if not marker_saved:
                log_error(
                    f"{symbol} untracked-position close and marker write "
                    "both failed; retaining quarantine and retrying"
                )

    return attempted, unresolved


def reconcile_interrupted_dca_submissions(position_details, state):
    """Flatten ENTRY/DCA exposure interrupted across an order submit boundary."""
    attempted = False
    unresolved = set()

    candidates = [
        symbol
        for symbol, item in list((state or {}).get("positions", {}).items())
        if interrupted_dca_submission(item) and
        not get_pending_execution(state, symbol)
    ]

    for symbol in candidates:
        attempted = True
        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            unresolved.add(symbol)
            continue

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if (
                not fresh_position_state or
                get_pending_execution(fresh_state, symbol) or
                not interrupted_dca_submission(fresh_position_state)
            ):
                state["positions"] = fresh_state.get("positions", {})
                state["pending_executions"] = fresh_state.get(
                    "pending_executions",
                    {},
                )
                continue

            live_details = get_open_position_details(symbol, force=True)
            detail = (live_details or {}).get(symbol)

            if live_details is None:
                unresolved.add(symbol)
                continue

            if not detail:
                # The position is already flat; verified cleanup and normal
                # pruning can release the interrupted reservation.
                if not cancel_open_protection_orders(symbol):
                    unresolved.add(symbol)
                    continue

                if not remove_position_state(fresh_state, symbol):
                    shutdown_event.set()
                    unresolved.add(symbol)
                else:
                    entry_quarantined_symbols.discard(symbol)

                state["positions"] = fresh_state.get("positions", {})
                state["pending_executions"] = fresh_state.get(
                    "pending_executions",
                    {},
                )
                continue

            interrupted = get_interrupted_submission(fresh_position_state)

            if not interrupted:
                state["positions"] = fresh_state.get("positions", {})
                state["pending_executions"] = fresh_state.get(
                    "pending_executions",
                    {},
                )
                continue

            submission_field = interrupted["field_name"]
            submission = interrupted["submission"]
            submission_context = interrupted["context"]
            original_phase = str(
                submission.get("interrupted_original_phase") or
                submission.get("submission_phase") or
                "UNKNOWN"
            ).upper()
            submission.update({
                "interrupted_original_phase": original_phase,
                "submission_phase": "FAIL_CLOSE_PENDING",
                "fail_close_reason": (
                    f"INTERRUPTED_{submission_context}_SUBMISSION"
                ),
            })
            updates = {
                submission_field: submission,
                "position_management_status": (
                    f"INTERRUPTED_{submission_context}_FAIL_CLOSE_PENDING"
                ),
            }

            if submission_field == "pending_dca":
                updates.update({
                    "dca_recovery_disabled": True,
                    "dca_recovery_disabled_reason": (
                        "INTERRUPTED_DCA_SUBMISSION"
                    ),
                })

            marker_saved = update_position_runtime_fields(
                fresh_state,
                symbol,
                updates,
            )

            if not marker_saved:
                log_error(
                    f"{symbol} interrupted-submission safety marker could not be "
                    "persisted; emergency close will still be attempted"
                )

            entry_quarantined_symbols.add(symbol)
            closed = fail_safe_close_unprotected_position(
                symbol,
                position_side=detail.get("position_side"),
                reference_price=(
                    detail.get("mark_price") or detail.get("entry_price")
                ),
                context=(
                    f"INTERRUPTED_{submission_context}_{original_phase}"
                ),
            )

            if closed:
                if not remove_position_state(fresh_state, symbol):
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()
                    unresolved.add(symbol)
                elif not marker_saved:
                    shutdown_event.set()
            else:
                unresolved.add(symbol)

                if not marker_saved:
                    log_error(
                        f"{symbol} interrupted-submission close and marker write "
                        "both failed; retaining quarantine and retrying"
                    )

            state["positions"] = fresh_state.get("positions", {})
            state["pending_executions"] = fresh_state.get(
                "pending_executions",
                {},
            )
        finally:
            lock.release()

    return attempted, unresolved


def _normalized_pending_client_order_ids(value):
    if isinstance(value, str):
        values = value.split(",")
    elif isinstance(value, (list, tuple, set)):
        values = value
    else:
        values = []

    normalized = []

    for item in values:
        client_order_id = str(item or "").strip()

        if client_order_id and client_order_id not in normalized:
            normalized.append(client_order_id)

    return tuple(normalized)


def _pending_execution_float(value):
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _pending_execution_age_seconds(pending):
    created_at_value = pending.get("created_at")

    if not created_at_value:
        return None

    try:
        created_at = datetime.fromisoformat(str(created_at_value))
        now = (
            datetime.now(created_at.tzinfo)
            if created_at.tzinfo is not None
            else datetime.now()
        )
        return max((now - created_at).total_seconds(), 0.0)
    except (TypeError, ValueError):
        return None


def _reset_pending_absence_confirmations(pending, reason):
    try:
        previous_count = int(
            pending.get("consecutive_absence_confirmations", 0) or 0
        )
    except (TypeError, ValueError):
        previous_count = 0

    first_confirmation = pending.get("absence_first_confirmed_at")
    changed = bool(previous_count or first_confirmation)
    pending["consecutive_absence_confirmations"] = 0
    pending["absence_first_confirmed_at"] = None

    if changed:
        pending["last_absence_reset_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        pending["last_absence_reset_reason"] = str(reason or "UNCERTAIN")

    return changed


def _record_pending_execution_observation(pending, result):
    initial = pending.get("reconciliation") or {}
    result = result or {}
    order_seen = bool(
        pending.get("order_seen") or
        pending.get("order_ids") or
        initial.get("order_ids") or
        result.get("any_order_seen") or
        result.get("orders") or
        result.get("order_ids")
    )
    max_executed_quantity = max(
        _pending_execution_float(
            pending.get("max_executed_quantity_seen")
        ),
        _pending_execution_float(
            initial.get("max_executed_quantity_seen")
        ),
        _pending_execution_float(initial.get("executed_quantity")),
        _pending_execution_float(
            result.get("max_executed_quantity_seen")
        ),
        _pending_execution_float(result.get("executed_quantity")),
    )
    pending["order_seen"] = order_seen
    pending["max_executed_quantity_seen"] = max_executed_quantity
    return order_seen, max_executed_quantity


def _authoritative_position_snapshot_is_flat(position_details, symbol):
    if not isinstance(position_details, dict):
        return False

    normalized_symbol = str(symbol or "").strip().upper()
    open_symbols = {
        str(open_symbol or "").strip().upper()
        for open_symbol in position_details
    }
    return normalized_symbol not in open_symbols


def persist_pending_execution(
    state,
    symbol,
    order,
    side,
    requested_quantity,
    pre_position_amount,
    reference_price,
    context,
    position_side=None,
    signal_type=None,
    dca_level=None,
    hard_stop_price=None,
    pre_average_price=None,
):
    reconciliation = get_execution_reconciliation(order)
    initial_executed_quantity = max(
        _pending_execution_float(reconciliation.get("executed_quantity")),
        _pending_execution_float(
            reconciliation.get("max_executed_quantity_seen")
        ),
    )
    pending = {
        "symbol": symbol,
        "side": str(side or "").upper(),
        "context": str(context or "ENTRY").upper(),
        "requested_quantity": float(requested_quantity or 0),
        "pre_position_amount": float(pre_position_amount or 0),
        "pre_average_price": float(pre_average_price or 0),
        "reference_price": float(reference_price or 0),
        "position_side": str(position_side or "BOTH").upper(),
        "signal_type": str(signal_type or "").upper(),
        "dca_level": dca_level,
        "hard_stop_price": float(hard_stop_price or 0),
        "client_order_ids": reconciliation.get("client_order_ids") or "",
        "order_ids": reconciliation.get("order_ids") or "",
        "execution_mode": reconciliation.get("execution_mode") or "",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "last_reconciled_at": None,
        "emergency_protection_secured": False,
        "order_seen": bool(
            reconciliation.get("order_ids") or
            reconciliation.get("orders") or
            reconciliation.get("any_order_seen") or
            initial_executed_quantity > 0
        ),
        "max_executed_quantity_seen": initial_executed_quantity,
        "consecutive_absence_confirmations": 0,
        "absence_first_confirmed_at": None,
        "reconciliation": reconciliation,
    }

    if upsert_pending_execution(state, symbol, pending):
        log_error(
            f"{symbol} persisted unsettled execution | "
            f"CONTEXT={pending['context']} | "
            f"CLIENT_IDS={pending['client_order_ids']}"
        )
        return True

    log_error(
        f"{symbol} CRITICAL: unsettled execution could not be persisted | "
        f"CONTEXT={pending['context']}"
    )
    return False


def _pending_execution_live_detail(symbol, pending):
    rows = get_open_position_detail_rows(symbol, force=True)

    if rows is None:
        return False, None

    position_side = str(pending.get("position_side") or "BOTH").upper()

    if position_side in ("LONG", "SHORT"):
        for detail in rows:
            if str(detail.get("position_side") or "").upper() == position_side:
                return True, detail

        return True, None

    if len(rows) > 1:
        log_error(
            f"{symbol} pending execution reconciliation blocked | "
            "multiple hedge legs require explicit position identity"
        )
        return False, None

    return True, rows[0] if rows else None


def _pending_execution_delta(pending, position_detail):
    side = str(pending.get("side") or "").upper()
    direction = 1 if side == "BUY" else -1
    pre_amount = float(pending.get("pre_position_amount", 0) or 0)
    live_amount = (
        float(position_detail.get("amount", 0) or 0)
        if position_detail
        else 0
    )
    return max((live_amount - pre_amount) * direction, 0)


def _secure_pending_execution_protection(state, symbol, pending, detail):
    live_quantity = abs(float(detail.get("amount", 0) or 0))
    entry_price = float(
        detail.get("entry_price") or pending.get("reference_price") or 0
    )
    side = SIDE_BUY if detail.get("side") == "BUY" else SIDE_SELL

    if live_quantity <= 0 or entry_price <= 0:
        return False

    is_dca = str(pending.get("context") or "").startswith("DCA")
    hard_stop_price = float(pending.get("hard_stop_price") or 0)

    if hard_stop_price > 0:
        exact_stop = find_matching_close_position_stop(
            symbol,
            side,
            hard_stop_price,
            position_side=detail.get("position_side"),
        )

        if exact_stop is None:
            log_warning(
                f"{symbol} unsettled execution stop lookup unavailable"
            )
            return False

        if not exact_stop:
            restored = place_stop_loss_only(
                symbol,
                side,
                entry_price,
                None,
                signal_type=pending.get("signal_type"),
                position_side=detail.get("position_side"),
                sl_price_override=hard_stop_price,
            )

            if not restored.get("ok"):
                log_error(
                    f"{symbol} unsettled execution exact hard stop restore failed"
                )
                return False

        if is_dca:
            pending["emergency_protection_secured"] = True
            pending["emergency_protection_mode"] = "EXACT_CAMPAIGN_SL"
            pending["emergency_sl_price"] = hard_stop_price
            return upsert_pending_execution(state, symbol, pending)
    elif is_dca:
        log_error(
            f"{symbol} unsettled DCA has no persisted campaign stop"
        )
        return False

    _, confirm_df, _ = get_signal_frames(symbol, None)
    protection_result = place_tp_sl_with_recovery(
        symbol,
        side,
        entry_price,
        live_quantity,
        confirm_df,
        signal_type=pending.get("signal_type"),
        context_label=f"{pending.get('context')}_UNSETTLED",
        enable_multi_tp=False,
        position_side=detail.get("position_side"),
        sl_price_override=hard_stop_price if hard_stop_price > 0 else None,
        preserve_existing_sl=hard_stop_price > 0,
        return_details=True,
    )

    if not protection_result.get("ok"):
        log_error(
            f"{symbol} unsettled execution emergency protection failed | "
            f"CONTEXT={pending.get('context')}"
        )
        return False

    pending["emergency_protection_secured"] = True
    pending["emergency_protection_mode"] = "FULL_POSITION_TP_SL"
    pending["emergency_tp_price"] = protection_result.get("tp_price")
    pending["emergency_sl_price"] = protection_result.get("sl_price")
    return upsert_pending_execution(state, symbol, pending)


def reconcile_pending_executions(state, position_details=None):
    pending_items = dict(state.get("pending_executions") or {})
    authoritative_position_details = position_details
    global_snapshot_attempted = position_details is not None
    absence_position_details = None
    absence_global_snapshot_attempted = False

    def forced_absence_position_details():
        nonlocal absence_position_details
        nonlocal absence_global_snapshot_attempted

        if not absence_global_snapshot_attempted:
            absence_position_details = get_open_position_details(force=True)
            absence_global_snapshot_attempted = True

        return absence_position_details

    for symbol, pending in pending_items.items():
        symbol_known = is_known_futures_symbol(symbol)

        if symbol_known is None:
            entry_quarantined_symbols.add(symbol)

            if _reset_pending_absence_confirmations(
                pending,
                "SYMBOL_CATALOG_UNAVAILABLE",
            ) and not upsert_pending_execution(state, symbol, pending):
                log_error(
                    f"{symbol} pending absence reset could not be persisted; "
                    "original marker remains quarantined"
                )

            log_warning(
                f"{symbol} pending execution symbol validation unavailable; "
                "marker retained without symbol-private reconciliation"
            )
            continue

        if symbol_known is False:
            if not global_snapshot_attempted:
                authoritative_position_details = get_open_position_details(
                    force=True,
                )
                global_snapshot_attempted = True

            if not isinstance(authoritative_position_details, dict):
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} is absent from the futures catalog but the "
                    "authoritative all-position snapshot is unavailable; "
                    "pending marker retained"
                )
                continue

            authoritative_open_symbols = {
                str(open_symbol or "").strip().upper()
                for open_symbol in authoritative_position_details
            }

            if str(symbol or "").strip().upper() in authoritative_open_symbols:
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} is absent from the futures catalog but appears "
                    "in the authoritative all-position snapshot; pending "
                    "marker retained for manual review"
                )
                continue

            try:
                dca_level = int(pending.get("dca_level") or 0)
            except (TypeError, ValueError):
                dca_level = 0

            pending_context = str(
                pending.get("context") or ""
            ).upper()
            invalid_entry_ids = _normalized_pending_client_order_ids(
                pending.get("client_order_ids")
            )
            invalid_flat_entry = bool(
                pending_context == "ENTRY" and
                abs(
                    _pending_execution_float(
                        pending.get("pre_position_amount")
                    )
                ) <= 1e-12 and
                invalid_entry_ids
            )

            if invalid_flat_entry:
                cleanup_result = clear_confirmed_absent_entry_execution(
                    state,
                    symbol,
                    invalid_entry_ids,
                )

                if isinstance(cleanup_result, tuple):
                    cleanup_succeeded, cleanup_reason = cleanup_result
                else:
                    cleanup_succeeded = bool(cleanup_result)
                    cleanup_reason = (
                        "CLEARED" if cleanup_succeeded else "CLEAR_FAILED"
                    )

                if not cleanup_succeeded:
                    entry_quarantined_symbols.add(symbol)
                    log_error(
                        f"{symbol} invalid-symbol ENTRY cleanup failed | "
                        f"REASON={cleanup_reason}; marker retained"
                    )
                    continue

                entry_quarantined_symbols.discard(symbol)
                log_warning(
                    f"{symbol} invalid exchange-symbol ENTRY state cleared "
                    "atomically | REASON=AUTHORITATIVE_CATALOG_ABSENCE_AND_"
                    "FLAT_ACCOUNT"
                )
                continue

            pending_dca_reservation = bool(
                (get_position_state(state, symbol) or {}).get("pending_dca")
            )

            if dca_level <= 0 and pending_dca_reservation:
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} invalid-symbol pending cleanup deferred | "
                    "DCA reservation exists but its execution level is missing"
                )
                continue

            if dca_level > 0:
                reservation_cleared = clear_dca_reservation(
                    state,
                    symbol,
                    dca_level,
                )

                if reservation_cleared is False:
                    entry_quarantined_symbols.add(symbol)
                    log_error(
                        f"{symbol} invalid-symbol DCA reservation cleanup "
                        "failed; pending marker retained"
                    )
                    continue

            pending_removed = remove_pending_execution(state, symbol)

            if pending_removed is False:
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} invalid-symbol pending marker cleanup failed; "
                    "marker retained"
                )
                continue

            entry_quarantined_symbols.discard(symbol)
            log_warning(
                f"{symbol} invalid exchange-symbol runtime state cleared | "
                "REASON=AUTHORITATIVE_CATALOG_ABSENCE_AND_FLAT_ACCOUNT"
            )
            continue

        client_ids = pending.get("client_order_ids") or ""
        result = reconcile_execution_client_orders(
            symbol,
            client_ids,
            cancel_unsettled=True,
        )
        pending["last_reconciled_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        pending["last_reconciliation"] = result
        _record_pending_execution_observation(pending, result)
        snapshot_available, detail = _pending_execution_live_detail(
            symbol,
            pending,
        )

        if not snapshot_available:
            _reset_pending_absence_confirmations(
                pending,
                "SYMBOL_POSITION_SNAPSHOT_UNAVAILABLE",
            )
            if not upsert_pending_execution(state, symbol, pending):
                entry_quarantined_symbols.add(symbol)
                log_error(
                    f"{symbol} CRITICAL: pending reconciliation state "
                    "update could not be persisted; original marker remains "
                    "for retry"
                )
            continue

        observed_delta = _pending_execution_delta(pending, detail)
        pending["observed_position_delta"] = observed_delta
        pre_amount = float(pending.get("pre_position_amount", 0) or 0)
        live_amount = (
            float(detail.get("amount", 0) or 0)
            if detail
            else 0.0
        )
        topology_tolerance = max(abs(pre_amount) * 1e-8, 1e-12)

        if observed_delta > topology_tolerance:
            pending["order_seen"] = True
            pending["max_executed_quantity_seen"] = max(
                _pending_execution_float(
                    pending.get("max_executed_quantity_seen")
                ),
                observed_delta,
            )

        amount_matches_pre_position = bool(
            detail and
            abs(live_amount - pre_amount) <= topology_tolerance
        )
        pre_average_price = float(
            pending.get("pre_average_price", 0) or 0
        )
        live_average_price = (
            float(detail.get("entry_price", 0) or 0)
            if detail
            else 0.0
        )
        average_price_matches = bool(
            pre_average_price <= 0 or
            live_average_price <= 0 or
            abs(live_average_price - pre_average_price) <= max(
                abs(pre_average_price) * 1e-8,
                1e-12,
            )
        )
        pending_side = str(pending.get("side") or "").upper()
        live_side = str((detail or {}).get("side") or "").upper()
        side_matches = bool(
            not detail or
            pending_side not in ("BUY", "SELL") or
            live_side == pending_side
        )
        unchanged_pre_position = bool(
            amount_matches_pre_position and
            average_price_matches and
            side_matches
        )
        is_dca_pending = bool(
            pending.get("dca_level") or
            str(pending.get("context") or "").upper().startswith("DCA")
        )
        reported_executed_quantity = max(
            float(result.get("executed_quantity", 0) or 0),
            0,
        )

        # Closed exposure remains owned until every originating client ID is
        # terminal. A one-cycle absence sweep is not enough after an actual
        # position change has already been observed.
        origin_resolution_proven = bool(result.get("order_terminal"))

        if (
            pending.get("unsettled_exposure_close_confirmed") and
            not detail and
            origin_resolution_proven
        ):
            forced_positions = forced_absence_position_details()

            if not _authoritative_position_snapshot_is_flat(
                forced_positions,
                symbol,
            ):
                entry_quarantined_symbols.add(symbol)
                pending["closed_exposure_origin_cleanup_error"] = (
                    "AUTHORITATIVE_FLAT_SNAPSHOT_UNAVAILABLE"
                    if not isinstance(forced_positions, dict)
                    else "AUTHORITATIVE_AND_SYMBOL_TOPOLOGY_CONFLICT"
                )
                _reset_pending_absence_confirmations(
                    pending,
                    pending["closed_exposure_origin_cleanup_error"],
                )
                upsert_pending_execution(state, symbol, pending)
                log_error(
                    f"{symbol} closed unsettled exposure origin resolved but "
                    "authoritative flat topology is unconfirmed"
                )
                continue

            context = str(pending.get("context") or "").upper()
            expected_ids = _normalized_pending_client_order_ids(
                pending.get("client_order_ids")
            )
            cleanup_succeeded = False
            cleanup_reason = "UNSUPPORTED_PENDING_CONTEXT"

            if context == "ENTRY":
                cleanup_result = clear_confirmed_absent_entry_execution(
                    state,
                    symbol,
                    expected_ids,
                )

                if isinstance(cleanup_result, tuple):
                    cleanup_succeeded, cleanup_reason = cleanup_result
                else:
                    cleanup_succeeded = bool(cleanup_result)
                    cleanup_reason = (
                        "CLEARED" if cleanup_succeeded else "CLEAR_FAILED"
                    )
            elif is_dca_pending and pending.get("dca_level"):
                reservation_result = clear_dca_reservation(
                    state,
                    symbol,
                    pending.get("dca_level"),
                )

                if reservation_result is not False:
                    remove_result = remove_pending_execution(state, symbol)
                    cleanup_succeeded = remove_result is not False
                    cleanup_reason = (
                        "CLEARED"
                        if cleanup_succeeded
                        else "PENDING_EXECUTION_REMOVE_FAILED"
                    )
                else:
                    cleanup_reason = "DCA_RESERVATION_CLEAR_FAILED"

            if cleanup_succeeded:
                entry_quarantined_symbols.discard(symbol)
                log_warning(
                    f"{symbol} closed unsettled exposure released after "
                    "origin resolution and authoritative flat confirmation"
                )
            else:
                entry_quarantined_symbols.add(symbol)
                pending["closed_exposure_origin_cleanup_error"] = str(
                    cleanup_reason
                )

                if context == "ENTRY":
                    if cleanup_reason == "STATE_WRITE_FAILED":
                        shutdown_event.set()
                elif not upsert_pending_execution(state, symbol, pending):
                    shutdown_event.set()

                log_error(
                    f"{symbol} closed unsettled exposure cleanup failed | "
                    f"REASON={cleanup_reason}"
                )

            continue

        if not result.get("order_terminal"):
            context = str(pending.get("context") or "").upper()
            pending_ids = _normalized_pending_client_order_ids(
                pending.get("client_order_ids")
            )
            result_ids = _normalized_pending_client_order_ids(
                result.get("client_order_ids")
            )
            grace_seconds = max(
                _pending_execution_float(
                    getattr(
                        config,
                        "PENDING_EXECUTION_ABSENCE_GRACE_SECONDS",
                        60,
                    )
                ),
                60.0,
            )
            try:
                required_confirmations = max(
                    int(
                        getattr(
                            config,
                            "PENDING_EXECUTION_ABSENCE_CONFIRMATIONS",
                            3,
                        )
                    ),
                    3,
                )
            except (TypeError, ValueError):
                required_confirmations = 3

            pending_age = _pending_execution_age_seconds(pending)
            no_order_or_execution_seen = bool(
                not pending.get("order_seen") and
                _pending_execution_float(
                    pending.get("max_executed_quantity_seen")
                ) <= topology_tolerance
            )
            local_absence_proof = bool(
                context == "ENTRY" and
                abs(pre_amount) <= topology_tolerance and
                pending_ids and
                result.get("client_order_ids_valid") is True and
                result_ids == pending_ids and
                result.get("all_definitively_absent") is True and
                not result.get("lookup_uncertain") and
                no_order_or_execution_seen and
                detail is None and
                pending_age is not None and
                pending_age >= grace_seconds
            )

            if local_absence_proof:
                forced_positions = forced_absence_position_details()

                if _authoritative_position_snapshot_is_flat(
                    forced_positions,
                    symbol,
                ):
                    try:
                        absence_count = int(
                            pending.get(
                                "consecutive_absence_confirmations",
                                0,
                            ) or 0
                        ) + 1
                    except (TypeError, ValueError):
                        absence_count = 1

                    confirmed_at = datetime.now().isoformat(
                        timespec="seconds"
                    )
                    pending["consecutive_absence_confirmations"] = (
                        absence_count
                    )
                    pending["absence_first_confirmed_at"] = (
                        pending.get("absence_first_confirmed_at") or
                        confirmed_at
                    )
                    pending["last_absence_confirmed_at"] = confirmed_at
                    pending["last_absence_evidence"] = dict(
                        result.get("absence_evidence") or {}
                    )
                    entry_quarantined_symbols.add(symbol)

                    if absence_count >= required_confirmations:
                        clear_result = (
                            clear_confirmed_absent_entry_execution(
                                state,
                                symbol,
                                pending_ids,
                            )
                        )

                        if isinstance(clear_result, tuple):
                            cleared, clear_reason = clear_result
                        else:
                            cleared = bool(clear_result)
                            clear_reason = (
                                "CLEARED" if cleared else "CLEAR_FAILED"
                            )

                        if cleared:
                            entry_quarantined_symbols.discard(symbol)
                            log_warning(
                                f"{symbol} confirmed never-accepted ENTRY "
                                "cleared | forced symbol/global flat, no "
                                "order or execution observed"
                            )
                            continue

                        pending["absence_cleanup_error"] = str(clear_reason)

                        if clear_reason == "STATE_WRITE_FAILED":
                            shutdown_event.set()

                        log_error(
                            f"{symbol} confirmed-absent ENTRY cleanup "
                            f"failed | REASON={clear_reason}"
                        )
                        continue

                    if not upsert_pending_execution(
                        state,
                        symbol,
                        pending,
                    ):
                        log_error(
                            f"{symbol} absence confirmation state could not "
                            "be persisted; original marker remains"
                        )

                    log_warning(
                        f"{symbol} pending ENTRY absence confirmation | "
                        f"COUNT={absence_count}/{required_confirmations}"
                    )
                    continue

                _reset_pending_absence_confirmations(
                    pending,
                    "AUTHORITATIVE_GLOBAL_FLAT_UNCONFIRMED",
                )
            else:
                reset_reason = (
                    "ORDER_LOOKUP_UNCERTAIN"
                    if result.get("lookup_uncertain")
                    else "ABSENCE_PROOF_INCOMPLETE"
                )
                _reset_pending_absence_confirmations(pending, reset_reason)

            topology_requires_protection = bool(
                detail and
                (
                    observed_delta > 0 or
                    reported_executed_quantity > topology_tolerance or
                    (is_dca_pending and not unchanged_pre_position)
                )
            )

            if topology_requires_protection:
                protected = _secure_pending_execution_protection(
                    state,
                    symbol,
                    pending,
                    detail,
                )

                if not protected:
                    entry_quarantined_symbols.add(symbol)
                    pending["emergency_protection_error"] = (
                        "EMERGENCY_PROTECTION_STATE_NOT_PERSISTED"
                        if pending.get("emergency_protection_secured")
                        else "EMERGENCY_PROTECTION_NOT_SECURED"
                    )

                    closed = fail_safe_close_unprotected_position(
                        symbol,
                        position_side=detail.get("position_side"),
                        reference_price=(
                            detail.get("mark_price") or
                            pending.get("reference_price")
                        ),
                        context=(
                            f"{pending.get('context')}_UNSETTLED_UNPROTECTED"
                        ),
                    )

                    if closed:
                        pending["order_seen"] = True
                        pending["max_executed_quantity_seen"] = max(
                            _pending_execution_float(
                                pending.get("max_executed_quantity_seen")
                            ),
                            observed_delta,
                            reported_executed_quantity,
                        )
                        pending["unsettled_exposure_close_confirmed"] = True
                        pending["unsettled_exposure_closed_at"] = (
                            datetime.now().isoformat(timespec="seconds")
                        )
                        pending.pop("emergency_protection_error", None)

                        if not upsert_pending_execution(
                            state,
                            symbol,
                            pending,
                        ):
                            log_error(
                                f"{symbol} unsettled position closed but "
                                "origin-tracking state update failed; the "
                                "durable marker remains and shutdown is "
                                "requested"
                            )
                            shutdown_event.set()
                        else:
                            log_warning(
                                f"{symbol} unsettled exposure closed; pending "
                                "origin retained until terminal or "
                                "definitive-absence proof"
                            )
                    else:
                        if not upsert_pending_execution(
                            state,
                            symbol,
                            pending,
                        ):
                            log_error(
                                f"{symbol} CRITICAL: unprotected unsettled "
                                "execution update could not be persisted; "
                                "original durable marker remains quarantined "
                                "for retry"
                            )

                    continue
            else:
                if not upsert_pending_execution(state, symbol, pending):
                    log_error(
                        f"{symbol} CRITICAL: unsettled execution state "
                        "update could not be persisted; original marker "
                        "remains for retry"
                    )

            log_error(
                f"{symbol} execution remains unsettled | "
                f"CONTEXT={pending.get('context')} | no new order allowed"
            )
            continue

        reconciled_quantity = reported_executed_quantity
        terminal_without_new_fill = bool(
            (
                not detail and
                reconciled_quantity <= topology_tolerance
            ) or
            (
                is_dca_pending and
                reconciled_quantity <= topology_tolerance and
                unchanged_pre_position
            )
        )

        if not detail and reconciled_quantity > topology_tolerance:
            # The order endpoint confirms a fill but the position endpoint is
            # still stale/unavailable. Never classify this as flat and never
            # discard its durable marker. Best-effort install the immutable
            # campaign stop directly, then wait for observable topology so the
            # ambiguous exposure can be reconciled safely.
            hard_stop_price = float(pending.get("hard_stop_price") or 0)
            order_side = str(pending.get("side") or "").upper()
            terminal_stop_secured = False

            if hard_stop_price > 0 and order_side in (SIDE_BUY, SIDE_SELL):
                exact_stop = find_matching_close_position_stop(
                    symbol,
                    order_side,
                    hard_stop_price,
                    position_side=pending.get("position_side"),
                )

                if exact_stop:
                    terminal_stop_secured = True
                elif exact_stop is not None:
                    emergency_stop = place_stop_loss_only(
                        symbol,
                        order_side,
                        float(
                            result.get("average_fill_price") or
                            pending.get("reference_price") or
                            0
                        ),
                        None,
                        signal_type=pending.get("signal_type"),
                        position_side=pending.get("position_side"),
                        sl_price_override=hard_stop_price,
                    )
                    terminal_stop_secured = bool(emergency_stop.get("ok"))

            pending["terminal_fill_topology_pending"] = True
            pending["terminal_fill_protection_secured"] = (
                terminal_stop_secured
            )
            pending["emergency_protection_secured"] = terminal_stop_secured
            pending["emergency_protection_mode"] = (
                "EXACT_CAMPAIGN_SL" if terminal_stop_secured else ""
            )
            persisted = upsert_pending_execution(state, symbol, pending)
            entry_quarantined_symbols.add(symbol)

            if not persisted:
                log_error(
                    f"{symbol} terminal-fill topology update was not "
                    "persisted; original marker remains for retry"
                )

            log_error(
                f"{symbol} terminal fill awaits live position topology | "
                f"QTY={reconciled_quantity} | "
                f"STOP_SECURED={terminal_stop_secured}"
            )
            continue

        if terminal_without_new_fill:
            if detail:
                # A terminal DCA with no new fill still has the original live
                # position. Its campaign stop/TP belong to that position and
                # must never be swept as stale protection. Re-verify (and, if
                # needed, restore) the exact immutable stop before releasing
                # the durable recovery reservation.
                protected = _secure_pending_execution_protection(
                    state,
                    symbol,
                    pending,
                    detail,
                )

                if not protected:
                    pending["cleanup_error"] = (
                        "NO_FILL_ORIGINAL_POSITION_PROTECTION_UNCONFIRMED"
                    )
                    persisted = upsert_pending_execution(state, symbol, pending)
                    entry_quarantined_symbols.add(symbol)

                    if not persisted:
                        log_error(
                            f"{symbol} original-position protection update "
                            "was not persisted; marker remains for retry"
                        )
                    log_error(
                        f"{symbol} terminal recovery had no new fill but the "
                        "original position stop could not be verified"
                    )
                    continue
            elif not cancel_open_protection_orders(symbol):
                pending["cleanup_error"] = (
                    "CLOSED_POSITION_PROTECTION_CLEANUP_UNCONFIRMED"
                )
                persisted = upsert_pending_execution(state, symbol, pending)
                entry_quarantined_symbols.add(symbol)

                if not persisted:
                    log_error(
                        f"{symbol} flat-position cleanup update was not "
                        "persisted; marker remains for retry"
                    )
                log_error(
                    f"{symbol} pending execution is flat but stale protection "
                    "cleanup is unconfirmed"
                )
                continue

            reservation_cleared = True

            if pending.get("dca_level"):
                reservation_result = clear_dca_reservation(
                    state,
                    symbol,
                    pending.get("dca_level"),
                )
                reservation_cleared = reservation_result is not False

            remove_result = (
                remove_pending_execution(state, symbol)
                if reservation_cleared
                else False
            )
            pending_removed = bool(
                reservation_cleared and remove_result is not False
            )

            if not pending_removed:
                pending["cleanup_error"] = (
                    "PENDING_EXECUTION_CLEANUP_NOT_PERSISTED"
                )
                if not upsert_pending_execution(state, symbol, pending):
                    log_error(
                        f"{symbol} cleanup retry update was not persisted; "
                        "original marker remains for retry"
                    )
                log_error(
                    f"{symbol} terminal pending execution cleanup failed; "
                    "symbol remains blocked"
                )
                continue

            log_warning(
                f"{symbol} pending execution reconciled terminal with no new fill"
            )
            if not detail:
                entry_quarantined_symbols.discard(symbol)
            continue

        closed = fail_safe_close_unprotected_position(
            symbol,
            position_side=detail.get("position_side"),
            reference_price=(
                detail.get("mark_price") or pending.get("reference_price")
            ),
            context=f"{pending.get('context')}_LATE_RECONCILIATION",
        )

        if closed:
            remove_result = remove_pending_execution(state, symbol)

            if remove_result is not False:
                log_warning(
                    f"{symbol} late/ambiguous fill was safely closed after "
                    "originating order became terminal"
                )
            else:
                pending["cleanup_error"] = (
                    "CLOSED_POSITION_PENDING_STATE_REMOVE_FAILED"
                )
                upsert_pending_execution(state, symbol, pending)
                log_error(
                    f"{symbol} late fill closed but pending state cleanup "
                    "failed; symbol remains blocked"
                )
        else:
            pending["terminal_fill_close_failed"] = True
            protected = _secure_pending_execution_protection(
                state,
                symbol,
                pending,
                detail,
            )
            pending["terminal_fill_protection_secured"] = bool(protected)
            entry_quarantined_symbols.add(symbol)
            persisted = upsert_pending_execution(state, symbol, pending)

            if not persisted:
                log_error(
                    f"{symbol} failed-close protection update was not "
                    "persisted; original marker remains for retry"
                )

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

    if state_requires_urgent_safety_retry(state):
        log_warning(
            f"{symbol} recovery add paused | another position-safety "
            "reconciliation is urgent"
        )
        return

    position_state = get_position_state(state, symbol)

    if get_pending_execution(state, symbol):
        log_warning(
            f"{symbol} DCA skipped | unsettled execution requires reconciliation"
        )
        return

    if has_active_dca_reservation(state, symbol):
        log_warning(
            f"{symbol} recovery add skipped | durable reservation requires "
            "completion or reconciliation"
        )
        return

    if not position_state:
        position_state = adopt_existing_position_state(
            state,
            symbol,
            position_detail
        )

    if not position_state or not position_state.get("managed_by_bot"):
        log_warning(f"{symbol} open position is not bot-managed; DCA skipped")
        return

    fixed_risk_enabled = bool(
        getattr(config, "DCA_FIXED_RISK_ENABLED", True)
    )

    if fixed_risk_enabled and not coordinated_position_management_enabled(
        position_state
    ):
        log_info(
            f"{symbol} recovery add skipped | legacy campaign is not migrated"
        )
        return

    if position_exit_blocks_dca(position_state):
        log_info(f"{symbol} recovery add skipped | exit/runner owns position")
        return

    confirmation_type = str(
        position_state.get("confirmation_type") or
        position_state.get("signal_type") or
        ""
    ).upper()
    recovery_mode = bool(
        getattr(config, "DCA_RECOVERY_CONFIRMATION_ENABLED", True)
    )

    if (
        recovery_mode and
        getattr(config, "DCA_RECOVERY_TREND_ONLY", True) and
        confirmation_type != "TREND"
    ):
        log_info(
            f"{symbol} recovery add skipped | route {confirmation_type or 'UNKNOWN'}"
        )
        return

    if (
        fixed_risk_enabled and
        int(config.DCA_MAX_ORDERS) != 1
    ):
        log_error(
            f"{symbol} recovery add disabled | DCA_MAX_ORDERS must equal 1"
        )
        return

    if (
        getattr(config, "TP1_RUNNER_DISABLE_DCA", True) and
        tp1_transition_blocks_recovery(position_state)
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

    if (
        config.DCA_MAX_ADVERSE_ROI > 0 and
        adverse_roi > config.DCA_MAX_ADVERSE_ROI
    ):
        disabled_saved = update_position_runtime_fields(
            state,
            symbol,
            {
                "dca_recovery_status": "CANCELLED",
                "dca_recovery_disabled": True,
                "dca_recovery_disabled_reason": "MAX_ADVERSE_ROI_EXCEEDED",
            },
        )

        if not disabled_saved:
            log_error(
                f"{symbol} maximum-adverse recovery lock was not durable; "
                "stopping the bot to prevent a later add"
            )
            shutdown_event.set()

        log_warning(
            f"{symbol} DCA skipped | maximum risk boundary exceeded | "
            f"ROI={adverse_roi}% > MAX={config.DCA_MAX_ADVERSE_ROI}%"
        )
        return

    if position_state.get("dca_recovery_disabled"):
        log_info(
            f"{symbol} recovery add disabled for campaign | "
            f"REASON={position_state.get('dca_recovery_disabled_reason')}"
        )
        return

    recovery_level = dca_count + 1
    recovery_status = str(
        position_state.get("dca_recovery_status") or ""
    ).upper()

    if not recovery_mode and adverse_roi < trigger_roi:
        log_info(
            f"{symbol} DCA not triggered | "
            f"LADDER_ROI={adverse_roi}% < TRIGGER={trigger_roi}%"
        )
        return

    recovery_armed = bool(
        not recovery_mode or
        (
            recovery_status == "ARMED" and
            int(position_state.get("dca_recovery_level", 0) or 0) ==
            recovery_level
        )
    )

    if not recovery_armed:
        if adverse_roi < trigger_roi:
            log_info(
                f"{symbol} recovery not armed | LEVEL={recovery_level} | "
                f"LADDER_ROI={adverse_roi}% < TRIGGER={trigger_roi}%"
            )
            return

        armed_at = datetime.now().isoformat(timespec="seconds")
        armed_updates = {
            "dca_recovery_status": "ARMED",
            "dca_recovery_level": recovery_level,
            "dca_recovery_armed_at": armed_at,
            "dca_recovery_arm_price": current_price,
            "dca_recovery_extreme_price": current_price,
            "dca_recovery_extreme_at": armed_at,
            "dca_recovery_peak_adverse_roi": adverse_roi,
            "dca_recovery_trigger_roi": trigger_roi,
        }

        if not update_position_runtime_fields(state, symbol, armed_updates):
            log_error(f"{symbol} recovery arm persistence failed; add blocked")
            return

        log_warning(
            f"{symbol} RECOVERY ARMED | LEVEL={recovery_level} | "
            f"ADVERSE_ROI={adverse_roi}% | waiting for rebound and 5m/15m"
        )
        return

    armed_elapsed = seconds_since(position_state.get("dca_recovery_armed_at"))
    arm_timeout = (
        max(
            float(getattr(config, "DCA_RECOVERY_ARM_TIMEOUT_MINUTES", 240)),
            0,
        ) * 60
        if recovery_mode
        else 0
    )

    if arm_timeout and armed_elapsed is not None and armed_elapsed > arm_timeout:
        expired_saved = update_position_runtime_fields(
            state,
            symbol,
            {
                "dca_recovery_status": "EXPIRED",
                "dca_recovery_disabled": True,
                "dca_recovery_disabled_reason": "RECOVERY_ARM_TIMEOUT",
            },
        )

        if not expired_saved:
            log_error(
                f"{symbol} recovery timeout was not durable; stopping the bot "
                "to prevent a later add"
            )
            shutdown_event.set()

        log_warning(f"{symbol} recovery add expired; campaign will not add")
        return

    extreme_price = float(
        position_state.get("dca_recovery_extreme_price") or current_price
    )
    new_extreme = (
        current_price < extreme_price
        if side == "BUY"
        else current_price > extreme_price
    )

    if new_extreme:
        previous_extreme = extreme_price
        extreme_price = current_price
        extreme_step = get_position_adverse_roi(
            side,
            previous_extreme,
            current_price,
        )
        persist_step = max(
            float(
                getattr(
                    config,
                    "DCA_RECOVERY_EXTREME_PERSIST_STEP_ROI",
                    1,
                )
            ),
            0,
        )

        if extreme_step >= persist_step:
            update_position_runtime_fields(
                state,
                symbol,
                {
                    "dca_recovery_extreme_price": extreme_price,
                    "dca_recovery_extreme_at": datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "dca_recovery_peak_adverse_roi": max(
                        adverse_roi,
                        float(
                            position_state.get(
                                "dca_recovery_peak_adverse_roi",
                                0,
                            ) or 0
                        ),
                    ),
                },
            )
        return

    rebound_roi = get_recovery_rebound_roi(side, extreme_price, current_price)
    min_rebound_roi = (
        max(
            float(getattr(config, "DCA_RECOVERY_MIN_REBOUND_ROI", 5)),
            0,
        )
        if recovery_mode
        else 0
    )

    if rebound_roi < min_rebound_roi:
        log_info(
            f"{symbol} recovery waiting | REBOUND={rebound_roi}% < "
            f"REQUIRED={min_rebound_roi}%"
        )
        return

    minimum_price_gap_roi = max(
        float(getattr(config, "DCA_MIN_PRICE_GAP_ROI", 0)),
        0,
    )
    recovery_price_gap_roi = get_dca_price_gap_roi(
        side,
        trigger_entry,
        current_price,
    )

    if recovery_price_gap_roi < minimum_price_gap_roi:
        log_info(
            f"{symbol} recovery add skipped | adverse price gap "
            f"{recovery_price_gap_roi}% < {minimum_price_gap_roi}%"
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
        "reason": "FIXED_RISK_RECOVERY_ADD",
        "level": current_price,
        "source": "armed_recovery",
        "price_source": price_source,
        "dca_level": dca_count + 1,
        "trigger_roi": trigger_roi,
        "adverse_roi": adverse_roi,
        "position_adverse_roi": position_adverse_roi,
        "trigger_entry": trigger_entry,
        "margin": dca_margin,
        "recovery_rebound_roi": rebound_roi,
        "recovery_extreme_price": extreme_price,
        "recovery_price_gap_roi": recovery_price_gap_roi,
    }

    log_warning(
        f"{symbol} RECOVERY ADD CANDIDATE | "
        f"LEVEL={dca_count + 1}/{config.DCA_MAX_ORDERS} | "
        f"LADDER_ROI={adverse_roi}% >= TRIGGER={trigger_roi}% | "
        f"POSITION_ROI={position_adverse_roi}% | REBOUND={rebound_roi}% | "
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

    if recovery_mode:
        recovery_ok, recovery_info = check_dca_recovery_confirmation(
            symbol,
            side,
            current_price,
        )

        if not recovery_ok:
            log_warning(
                f"{symbol} recovery add waiting | {recovery_info.get('reason')} | "
                f"SUPPORT={recovery_info.get('support_count')}/"
                f"{recovery_info.get('required_support')}"
            )
            return

        level_info["recovery_confirmation"] = recovery_info

    hard_stop_price = float(
        position_state.get("campaign_stop_price") or
        position_state.get("hard_stop_price") or
        0
    )

    if hard_stop_price <= 0:
        log_error(f"{symbol} recovery add blocked | campaign hard stop missing")
        return

    stop_buffer_roi = get_stop_buffer_roi(side, current_price, hard_stop_price)
    minimum_stop_buffer = max(
        float(getattr(config, "DCA_MIN_HARD_STOP_BUFFER_ROI", 5)),
        0,
    )

    if stop_buffer_roi < minimum_stop_buffer:
        log_warning(
            f"{symbol} recovery add blocked | hard-stop buffer "
            f"{stop_buffer_roi}% < {minimum_stop_buffer}%"
        )
        return

    exact_stop = find_matching_close_position_stop(
        symbol,
        side,
        hard_stop_price,
        position_side=position_detail.get("position_side"),
    )

    if getattr(config, "DCA_REQUIRE_HARD_STOP", True) and not exact_stop:
        log_error(f"{symbol} recovery add blocked | exact exchange stop missing")
        return

    planned_margin = max(
        float(position_state.get("planned_margin") or config.MARGIN_PER_TRADE),
        0,
    )
    used_margin = max(float(position_state.get("used_margin") or 0), 0)
    remaining_margin = max(planned_margin - used_margin, 0)
    dca_margin = min(dca_margin, remaining_margin)

    if dca_margin <= 0:
        log_info(f"{symbol} recovery add skipped | campaign margin exhausted")
        return

    campaign_risk_budget = max(
        float(position_state.get("campaign_risk_budget_usdt") or 0),
        0,
    )
    existing_risk = get_campaign_risk_at_stop(
        avg_entry,
        old_quantity,
        hard_stop_price,
    )
    remaining_risk = max(campaign_risk_budget - existing_risk, 0)
    recovery_risk_cap = campaign_risk_budget * max(
        float(getattr(config, "DCA_RECOVERY_RISK_PCT", 30)),
        0,
    ) / 100
    remaining_risk = min(remaining_risk, recovery_risk_cap)

    if getattr(config, "DCA_FIXED_RISK_ENABLED", True) and remaining_risk <= 0:
        log_info(f"{symbol} recovery add skipped | campaign risk exhausted")
        return

    balance = get_balance()
    quantity = calculate_position_size(
        balance,
        current_price,
        hard_stop_price,
        symbol,
        dca_margin,
        risk_budget_override=remaining_risk,
    )

    if quantity <= 0:
        log_warning(f"{symbol} recovery add skipped | risk/margin quantity is zero")
        return


    level_info.update({
        "hard_stop_price": hard_stop_price,
        "hard_stop_order_id": exact_stop.get("order_id") if exact_stop else "",
        "hard_stop_buffer_roi": stop_buffer_roi,
        "campaign_risk_budget_usdt": campaign_risk_budget,
        "existing_risk_usdt": round(existing_risk, 8),
        "recovery_risk_budget_usdt": round(remaining_risk, 8),
        "remaining_margin": round(remaining_margin, 8),
        "margin": dca_margin,
    })

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

    fresh_position, fresh_reason = refresh_dca_position_before_order(
        symbol,
        side,
        position_detail.get("amount", 0),
    )

    if not fresh_position:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} DCA aborted | {fresh_reason}")
        return

    latest_state = load_trade_state()
    latest_position_state = get_position_state(latest_state, symbol)

    if position_exit_blocks_dca(latest_position_state):
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} recovery add aborted | exit state changed")
        return

    exact_stop = find_matching_close_position_stop(
        symbol,
        side,
        hard_stop_price,
        position_side=fresh_position.get("position_side"),
    )

    if getattr(config, "DCA_REQUIRE_HARD_STOP", True) and not exact_stop:
        clear_dca_reservation(state, symbol, dca_level)
        log_error(f"{symbol} recovery add aborted | hard stop changed/missing")
        return

    refreshed_mark = get_mark_price(symbol)

    if refreshed_mark is None:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} recovery add aborted | fresh mark unavailable")
        return

    refreshed_rebound = get_recovery_rebound_roi(
        side,
        extreme_price,
        float(refreshed_mark),
    )

    if (
        refreshed_rebound < min_rebound_roi or
        get_dca_price_gap_roi(side, trigger_entry, float(refreshed_mark)) <
        minimum_price_gap_roi or
        get_stop_buffer_roi(side, float(refreshed_mark), hard_stop_price) <
        minimum_stop_buffer
    ):
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} recovery add aborted | recovery/stop buffer changed")
        return

    current_price = float(refreshed_mark)
    quantity = calculate_position_size(
        get_balance(),
        current_price,
        hard_stop_price,
        symbol,
        dca_margin,
        risk_budget_override=remaining_risk,
    )
    notional_ok, notional = validate_min_notional(
        symbol,
        quantity,
        current_price,
    )

    if quantity <= 0 or not notional_ok:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(
            f"{symbol} recovery add aborted | refreshed risk quantity invalid | "
            f"NOTIONAL={notional}"
        )
        return

    position_detail = fresh_position
    avg_entry = float(position_detail.get("entry_price", 0) or avg_entry)
    pre_position_amount = float(position_detail.get("amount", 0) or 0)
    old_quantity = abs(pre_position_amount)
    order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
    requested_quantity = quantity
    reservation_state = load_trade_state()
    reservation_item = get_position_state(reservation_state, symbol) or {}
    pending_dca = dict(reservation_item.get("pending_dca") or {})
    pending_dca.update({
        "submission_phase": "READY_TO_SUBMIT",
        "pre_position_amount": pre_position_amount,
        "pre_average_price": avg_entry,
        "requested_quantity": requested_quantity,
        "order_side": order_side,
    })

    if not update_position_runtime_fields(
        reservation_state,
        symbol,
        {"pending_dca": pending_dca},
    ):
        clear_dca_reservation(state, symbol, dca_level)
        log_error(
            f"{symbol} recovery add aborted | submission intent not durable"
        )
        return

    log_info(
        f"{symbol} DCA placing market order | "
        f"SIDE={order_side} | QTY={requested_quantity} | MARGIN={dca_margin}"
    )
    order = place_market_order(
        symbol,
        order_side,
        requested_quantity,
        pre_position_amount=pre_position_amount,
        pre_average_price=avg_entry,
        reference_price=current_price,
        context=f"DCA_LEVEL_{dca_level}",
    )

    if not order:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} DCA aborted | market order failed")
        return

    reconciliation = get_execution_reconciliation(order)
    pending_dca.update({
        "submission_phase": "ORDER_RETURNED",
        "execution_reconciliation": reconciliation,
    })
    order_returned_saved = update_position_runtime_fields(
        state,
        symbol,
        {"pending_dca": pending_dca},
    )

    if not order_returned_saved:
        persisted = persist_pending_execution(
            state,
            symbol,
            order,
            order_side,
            requested_quantity,
            pre_position_amount,
            current_price,
            context=f"DCA_LEVEL_{dca_level}",
            position_side=position_detail.get("position_side"),
            signal_type=(
                position_state.get("confirmation_type") or
                position_state.get("signal_type")
            ),
            dca_level=dca_level,
            hard_stop_price=hard_stop_price,
            pre_average_price=avg_entry,
        )

        log_error(
            f"{symbol} recovery order returned but its durable result state "
            "could not be updated; reconciliation is required"
        )
        entry_quarantined_symbols.add(symbol)

        if not persisted:
            closed = fail_safe_close_unprotected_position(
                symbol,
                position_side=position_detail.get("position_side"),
                reference_price=current_price,
                context="DCA_ORDER_RESULT_STATE_FAILURE",
            )

            if closed and not remove_position_state(state, symbol):
                shutdown_event.set()
        else:
            reconcile_pending_executions(state)

        return

    if not is_reconciled_execution_settled(order):
        pending_dca = dict(
            (get_position_state(state, symbol) or {}).get("pending_dca") or {}
        )
        pending_dca.update({
            "execution_unsettled": True,
            "execution_context": f"DCA_LEVEL_{dca_level}",
            "execution_reconciliation": reconciliation,
        })
        update_position_runtime_fields(
            state,
            symbol,
            {"pending_dca": pending_dca},
        )
        persisted = persist_pending_execution(
            state,
            symbol,
            order,
            order_side,
            requested_quantity,
            pre_position_amount,
            current_price,
            context=f"DCA_LEVEL_{dca_level}",
            position_side=position_detail.get("position_side"),
            signal_type=(
                position_state.get("confirmation_type") or
                position_state.get("signal_type")
            ),
            dca_level=dca_level,
            hard_stop_price=hard_stop_price,
            pre_average_price=avg_entry,
        )

        if not persisted:
            entry_quarantined_symbols.add(symbol)
            log_error(
                f"{symbol} unsettled DCA update was not persisted; "
                "durable submission reservation remains for retry"
            )
        else:
            # Do not wait for the next five-minute scan. A stop/DCA race can
            # leave an ambiguous reopened leg that needs its exact campaign
            # stop verified immediately.
            reconcile_pending_executions(state)

        log_error(
            f"{symbol} DCA execution is unsettled | LEVEL={dca_level} | "
            "reservation retained; no duplicate fallback will be submitted"
        )
        return

    quantity = get_reconciled_executed_quantity(order)

    if quantity <= 0:
        clear_dca_reservation(state, symbol, dca_level)
        log_warning(f"{symbol} DCA aborted | confirmed zero fill")
        return

    level_info["requested_quantity"] = requested_quantity
    level_info["executed_quantity"] = quantity
    level_info["execution_mode"] = reconciliation.get("execution_mode")
    level_info["execution_fallback_used"] = bool(
        reconciliation.get("fallback_used")
    )

    fill_price = get_entry_price(symbol, order)

    if fill_price <= 0:
        fill_price = current_price
        log_warning(f"{symbol} DCA fill price unavailable | using current price")

    filled_dca_margin = (
        quantity * fill_price / max(float(config.LEVERAGE), 1)
    )
    updated_position, topology_reason = verify_post_dca_position(
        symbol,
        side,
        pre_position_amount,
        quantity,
    )

    if topology_reason != "OK":
        log_error(
            f"{symbol} recovery add topology mismatch | {topology_reason}"
        )
        fail_close_post_dca_safety_violation(
            state,
            symbol,
            "DCA_POST_FILL_TOPOLOGY",
            updated_position,
            current_price,
        )
        return

    avg_entry = float(updated_position.get("entry_price", 0) or fill_price)
    total_quantity = abs(float(updated_position.get("amount", 0) or 0))
    post_fill_stop = find_matching_close_position_stop(
        symbol,
        side,
        hard_stop_price,
        position_side=updated_position.get("position_side"),
    )

    if getattr(config, "DCA_REQUIRE_HARD_STOP", True) and not post_fill_stop:
        log_error(f"{symbol} hard stop missing after recovery fill")
        fail_close_post_dca_safety_violation(
            state,
            symbol,
            "DCA_POST_FILL_STOP_MISSING",
            updated_position,
            current_price,
        )
        return

    actual_campaign_risk = get_campaign_risk_at_stop(
        avg_entry,
        total_quantity,
        hard_stop_price,
    )
    risk_tolerance = 1 + max(
        float(
            getattr(config, "POSITION_RISK_OVERRUN_TOLERANCE_PCT", 2)
        ),
        0,
    ) / 100

    if (
        campaign_risk_budget > 0 and
        actual_campaign_risk > campaign_risk_budget * risk_tolerance
    ):
        log_error(
            f"{symbol} recovery fill exceeded campaign risk | "
            f"ACTUAL={round(actual_campaign_risk, 4)} > "
            f"BUDGET={round(campaign_risk_budget, 4)}"
        )
        fail_close_post_dca_safety_violation(
            state,
            symbol,
            "DCA_RISK_OVERRUN",
            updated_position,
            current_price,
        )
        return

    level_info["actual_campaign_risk_usdt"] = round(actual_campaign_risk, 8)
    level_info["actual_dca_margin"] = round(filled_dca_margin, 8)

    if not record_dca_fill(
        state,
        symbol,
        avg_entry,
        total_quantity,
        filled_dca_margin,
        fill_price,
        level_info
    ):
        log_error(
            f"{symbol} DCA fill state update failed | "
            f"LEVEL={dca_level} | closing position as a fail-safe"
        )
        fail_safe_close_unprotected_position(
            symbol,
            position_side=position_detail.get("position_side"),
            reference_price=current_price,
            context=f"DCA_LEVEL_{dca_level}_STATE_PERSISTENCE",
        )
        shutdown_event.set()
        return

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
        if cancel_open_take_profit_orders(symbol):
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
                position_side=updated_position.get("position_side"),
                sl_price_override=hard_stop_price,
                preserve_existing_sl=True,
                return_details=True
            )
            protection_ok = bool(protection_result.get("ok"))
            new_tp_info = protection_result

            if not protection_ok:
                log_warning(f"{symbol} DCA TP ORDER NOT CREATED")

                closed = fail_safe_close_unprotected_position(
                    symbol,
                    position_side=updated_position.get("position_side"),
                    reference_price=current_price,
                    context="DCA_TP_REPLACEMENT_FAILURE",
                )
                send_telegram_message(
                    f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                    f"{symbol} recovery TP replacement failed\n"
                    f"Fail-safe close confirmed: {bool(closed)}"
                )

                if not closed:
                    shutdown_event.set()

                return
        else:
            log_error(
                f"{symbol} recovery TP cancel failed | hard stop retained; "
                "flattening to avoid mixed TP state"
            )
            closed = fail_safe_close_unprotected_position(
                symbol,
                position_side=updated_position.get("position_side"),
                reference_price=current_price,
                context="DCA_TP_CANCEL_FAILURE",
            )

            if not closed:
                shutdown_event.set()

            return

    if new_tp_info:
        if not update_position_tp_status(
            state,
            symbol,
            new_tp_info,
            context=f"DCA_LEVEL_{dca_count + 1}"
        ):
            log_error(
                f"{symbol} recovery protection state persistence failed"
            )
            fail_safe_close_unprotected_position(
                symbol,
                position_side=updated_position.get("position_side"),
                reference_price=current_price,
                context="DCA_PROTECTION_STATE_FAILURE",
            )
            shutdown_event.set()
            return

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


def ensure_route_stop_loss(
    symbol,
    position_detail,
    state,
    btc_trend_df,
):
    if not getattr(config, "HARD_STOP_RECONCILE_ENABLED", True):
        return

    lock = get_dca_lock(symbol)

    if not lock.acquire(blocking=False):
        log_info(f"{symbol} hard-stop reconcile deferred | position busy")
        return

    try:
        fresh_state = load_trade_state()
        result = _ensure_route_stop_loss_locked(
            symbol,
            position_detail,
            fresh_state,
            btc_trend_df,
        )
        state["positions"] = fresh_state.get("positions", {})
        state["pending_executions"] = fresh_state.get(
            "pending_executions",
            {},
        )
        return result
    finally:
        lock.release()


def _ensure_route_stop_loss_locked(
    symbol,
    position_detail,
    state,
    btc_trend_df,
):
    """Reconcile the immutable stop while owning the position-management lock."""

    position_state = get_position_state(state, symbol)

    if not position_state or not position_state.get("managed_by_bot"):
        return

    if position_state.get("multi_tp_stage") in (
        RUNNER_PENDING,
        RUNNER_ACTIVE,
    ):
        return

    signal_type = str(
        position_state.get("confirmation_type") or
        position_state.get("signal_type") or
        ""
    ).upper()

    signal_type = "REVERSAL" if signal_type == "REVERSAL" else "TREND"
    route_enabled = bool(
        getattr(
            config,
            f"{signal_type}_SL_ENABLED",
            getattr(config, "SL_ENABLED", False),
        )
    )

    if not route_enabled:
        return

    entry_price = float(
        position_detail.get("entry_price") or
        position_state.get("avg_entry") or
        position_state.get("initial_entry") or
        0
    )

    if entry_price <= 0:
        log_warning(f"{symbol} hard-stop reconcile skipped | missing entry")
        return

    state_side = str(position_state.get("side") or "").upper()
    live_side = str(position_detail.get("side") or "").upper()

    if state_side not in ("BUY", "SELL") or live_side not in ("BUY", "SELL"):
        log_error(
            f"{symbol} hard-stop reconcile blocked | invalid state/live side"
        )
        return

    if state_side != live_side:
        log_error(
            f"{symbol} hard-stop reconcile blocked | state side {state_side} "
            f"!= live side {live_side}"
        )
        return

    order_side = SIDE_BUY if state_side == "BUY" else SIDE_SELL
    stop_price = float(
        position_state.get("campaign_stop_price") or
        position_state.get("hard_stop_price") or
        0
    )
    confirm_df = None

    if stop_price <= 0:
        if not getattr(
            config,
            "HARD_STOP_RECONCILE_LEGACY_POSITIONS",
            False,
        ):
            update_position_runtime_fields(
                state,
                symbol,
                {
                    "dca_recovery_disabled": True,
                    "dca_recovery_disabled_reason": "LEGACY_RISK_PLAN_MISSING",
                    "sl_status": "LEGACY_UNMIGRATED",
                },
            )
            log_warning(
                f"{symbol} legacy position not auto-migrated to a new hard stop"
            )
            return

        _, confirm_df, _ = get_signal_frames(symbol, btc_trend_df)

        if confirm_df is None:
            log_warning(
                f"{symbol} legacy hard-stop reconcile skipped | data unavailable"
            )
            return

        stop_price = get_entry_hard_stop(
            symbol,
            order_side,
            entry_price,
            confirm_df,
            signal_type,
        ) or 0

        if stop_price <= 0:
            log_error(f"{symbol} legacy hard-stop planning failed")
            return

    exact_stop = find_matching_close_position_stop(
        symbol,
        order_side,
        stop_price,
        position_side=position_detail.get("position_side"),
    )

    if exact_stop is None:
        if not position_state.get("dca_recovery_disabled"):
            update_position_runtime_fields(
                state,
                symbol,
                {
                    "dca_recovery_disabled": True,
                    "dca_recovery_disabled_reason": (
                        "HARD_STOP_QUERY_UNAVAILABLE"
                    ),
                },
            )
        log_warning(
            f"{symbol} hard-stop reconcile deferred | exchange order query "
            "unavailable"
        )
        return

    if exact_stop:
        existing_disable_reason = str(
            position_state.get("dca_recovery_disabled_reason") or ""
        )
        preserve_existing_disable = bool(
            position_state.get("dca_recovery_disabled") and
            (
                existing_disable_reason not in {
                    "HARD_STOP_QUERY_UNAVAILABLE",
                    "HARD_STOP_RECONCILE_FAILED",
                } or
                str(
                    position_state.get("position_management_status") or
                    "ACTIVE"
                ).upper() != "ACTIVE"
            )
        )
        update_position_runtime_fields(
            state,
            symbol,
            {
                "sl_status": "CREATED",
                "sl_enabled": True,
                "sl_price": exact_stop.get("sl_price"),
                "sl_source": "STARTUP_EXCHANGE_VERIFIED",
                "hard_stop_price": exact_stop.get("sl_price"),
                "hard_stop_order_id": exact_stop.get("order_id"),
                "campaign_stop_price": stop_price,
                "dca_recovery_disabled": preserve_existing_disable,
                "dca_recovery_disabled_reason": (
                    existing_disable_reason if preserve_existing_disable else ""
                ),
            },
        )
        return

    if confirm_df is None:
        _, confirm_df, _ = get_signal_frames(symbol, btc_trend_df)

    result = place_stop_loss_only(
        symbol,
        order_side,
        entry_price,
        confirm_df,
        signal_type=signal_type,
        position_side=position_detail.get("position_side"),
        sl_price_override=stop_price,
    )
    sl_created = bool(result.get("ok"))
    sl_order_id = extract_order_id(result.get("sl_order"))
    prior_disabled = bool(position_state.get("dca_recovery_disabled"))
    prior_disable_reason = str(
        position_state.get("dca_recovery_disabled_reason") or ""
    )
    preserve_terminal_disable = bool(
        prior_disabled and
        (
            prior_disable_reason not in {
                "HARD_STOP_QUERY_UNAVAILABLE",
                "HARD_STOP_RECONCILE_FAILED",
            } or
            str(
                position_state.get("position_management_status") or "ACTIVE"
            ).upper() != "ACTIVE"
        )
    )
    update_position_runtime_fields(
        state,
        symbol,
        {
            "sl_status": "CREATED" if sl_created else "FAILED",
            "sl_enabled": sl_created,
            "sl_price": result.get("sl_price"),
            "sl_source": f"{signal_type}_STARTUP_RECONCILE",
            "hard_stop_price": result.get("sl_price") if sl_created else stop_price,
            "hard_stop_order_id": sl_order_id,
            "campaign_stop_price": stop_price,
            "dca_recovery_disabled": (
                True if not sl_created else preserve_terminal_disable
            ),
            "dca_recovery_disabled_reason": (
                (
                    prior_disable_reason
                    if preserve_terminal_disable
                    else ""
                )
                if sl_created
                else "HARD_STOP_RECONCILE_FAILED"
            ),
        },
    )

    if sl_created:
        send_telegram_message(
            f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
            f"{symbol} {signal_type.lower()} hard stop restored\n"
            f"SL: {result.get('sl_price')}"
        )
    else:
        log_error(f"{symbol} {signal_type.lower()} hard-stop reconcile failed")

        if getattr(config, "HARD_STOP_STARTUP_FAIL_CLOSE", False):
            closed = fail_safe_close_unprotected_position(
                symbol,
                position_side=position_detail.get("position_side"),
                reference_price=position_detail.get("mark_price"),
                context="STARTUP_HARD_STOP_FAILURE",
            )

            if not closed:
                shutdown_event.set()


def ensure_reversal_stop_loss(symbol, position_detail, state, btc_trend_df):
    """Backward-compatible wrapper retained for existing integrations/tests."""
    return ensure_route_stop_loss(symbol, position_detail, state, btc_trend_df)


def repair_pending_dca_tp_reprice(
    symbol,
    position_detail,
    state,
    btc_trend_df,
):
    caller_position_state = get_position_state(state, symbol) or {}
    caller_reprice_pending = str(
        caller_position_state.get("tp_reprice_status") or ""
    ).upper() in ("PENDING", "FAILED")
    lock = get_dca_lock(symbol)

    if not lock.acquire(blocking=False):
        log_info(f"{symbol} TP reprice repair deferred | position busy")
        return True

    try:
        fresh_state = load_trade_state()
        fresh_position_state = get_position_state(fresh_state, symbol)

        def sync_fresh_state():
            state["positions"] = fresh_state.get("positions", {})
            state["pending_executions"] = fresh_state.get(
                "pending_executions",
                {},
            )

        if not fresh_position_state:
            sync_fresh_state()
            return caller_reprice_pending

        fresh_reprice_status = str(
            fresh_position_state.get("tp_reprice_status") or ""
        ).upper()

        if fresh_reprice_status not in ("PENDING", "FAILED"):
            sync_fresh_state()
            return False

        if get_pending_execution(fresh_state, symbol):
            sync_fresh_state()
            return True

        if committed_position_exit_owner(fresh_position_state):
            sync_fresh_state()
            return True

        if runner_owns_position(fresh_position_state):
            result = _repair_pending_dca_tp_reprice_locked(
                symbol,
                {},
                fresh_state,
                btc_trend_df,
            )
            sync_fresh_state()
            return result

        live_details = get_open_position_details(symbol, force=True)

        if live_details is None:
            sync_fresh_state()
            log_warning(
                f"{symbol} TP reprice repair deferred | live topology unavailable"
            )
            return True

        fresh_position_detail = live_details.get(symbol)

        if not fresh_position_detail:
            sync_fresh_state()
            log_warning(
                f"{symbol} TP reprice repair deferred | live position missing"
            )
            return True

        result = _repair_pending_dca_tp_reprice_locked(
            symbol,
            fresh_position_detail,
            fresh_state,
            btc_trend_df,
        )
        sync_fresh_state()
        return result
    finally:
        lock.release()


def _repair_pending_dca_tp_reprice_locked(
    symbol,
    position_detail,
    state,
    btc_trend_df,
):
    position_state = get_position_state(state, symbol)
    reprice_status = str(
        (position_state or {}).get("tp_reprice_status") or ""
    ).upper()

    if reprice_status not in ("PENDING", "FAILED"):
        return False

    if not position_state or not position_state.get("managed_by_bot"):
        return True

    if runner_owns_position(position_state):
        update_position_runtime_fields(
            state,
            symbol,
            {"tp_reprice_status": "COMPLETE_RUNNER_OWNERSHIP"},
        )
        return True

    side = str(position_detail.get("side") or position_state.get("side") or "").upper()
    order_side = SIDE_BUY if side == "BUY" else SIDE_SELL if side == "SELL" else ""
    avg_entry = float(
        position_detail.get("entry_price") or
        position_state.get("tp_reprice_avg_entry") or
        position_state.get("avg_entry") or
        0
    )
    quantity = abs(float(position_detail.get("amount", 0) or 0))
    hard_stop_price = float(
        position_state.get("tp_reprice_hard_stop_price") or
        position_state.get("campaign_stop_price") or
        0
    )

    if not order_side or avg_entry <= 0 or quantity <= 0 or hard_stop_price <= 0:
        log_error(f"{symbol} TP reprice repair blocked | invalid persisted context")
        return True

    exact_stop = find_matching_close_position_stop(
        symbol,
        order_side,
        hard_stop_price,
        position_side=position_detail.get("position_side"),
    )

    if exact_stop is None:
        log_warning(f"{symbol} TP reprice repair deferred | stop query unavailable")
        return True

    if not exact_stop:
        log_warning(f"{symbol} TP reprice repair deferred | exact stop missing")
        return True

    trend_df, confirm_df, _ = get_signal_frames(symbol, btc_trend_df)
    structure_tp = None
    dca_tp_roi = None

    if config.DCA_TP_MODE in ("roi", "fixed_roi", "fallback_roi"):
        dca_tp_roi = config.DCA_TP_ROI
    elif (
        not config.STATIC_TP_ENABLED and
        trend_df is not None and
        confirm_df is not None
    ):
        tp_ok, structure_tp = validate_structure_take_profit(
            side,
            avg_entry,
            trend_df,
            confirm_df,
            leverage=config.LEVERAGE,
        )

        if not tp_ok:
            structure_tp = None

    if not cancel_open_take_profit_orders(symbol):
        log_error(f"{symbol} TP reprice repair deferred | TP cleanup unavailable")
        return True

    protection_result = place_tp_sl_with_recovery(
        symbol,
        order_side,
        avg_entry,
        quantity,
        confirm_df,
        structure_tp=structure_tp,
        roi_override=dca_tp_roi,
        roi_mode_label=(
            f"DCA_ROI_{dca_tp_roi}%" if dca_tp_roi is not None else None
        ),
        signal_type=(
            position_state.get("confirmation_type") or
            position_state.get("signal_type")
        ),
        context_label="DCA_RESTART_REPRICE",
        enable_multi_tp=(
            bool(getattr(config, "MULTI_TP_ENABLED", False)) and
            position_state.get("multi_tp_stage") == TP1_PENDING
        ),
        position_side=position_detail.get("position_side"),
        sl_price_override=hard_stop_price,
        preserve_existing_sl=True,
        return_details=True,
    )

    if not protection_result.get("ok"):
        update_position_runtime_fields(
            state,
            symbol,
            {"tp_reprice_status": "FAILED"},
        )
        closed = fail_safe_close_unprotected_position(
            symbol,
            position_side=position_detail.get("position_side"),
            reference_price=position_detail.get("mark_price"),
            context="DCA_RESTART_TP_REPRICE_FAILURE",
        )

        if not closed:
            shutdown_event.set()

        return True

    if not update_position_tp_status(
        state,
        symbol,
        protection_result,
        context="DCA_LEVEL_RESTART_REPAIR",
    ):
        fail_safe_close_unprotected_position(
            symbol,
            position_side=position_detail.get("position_side"),
            reference_price=position_detail.get("mark_price"),
            context="DCA_RESTART_TP_STATE_FAILURE",
        )
        shutdown_event.set()
        return True

    log_info(f"{symbol} interrupted DCA TP reprice repaired")
    return True


def dca_tick_ready(symbol, mark_price, state=None):
    state = state or load_trade_state()
    position_state = get_position_state(state, symbol)

    if not position_state or not position_state.get("managed_by_bot"):
        return False

    if (
        getattr(config, "TP1_RUNNER_DISABLE_DCA", True) and
        tp1_transition_blocks_recovery(position_state)
    ):
        return False

    if position_exit_blocks_dca(position_state):
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

    if position_state.get("dca_recovery_disabled"):
        return False

    if (
        str(position_state.get("dca_recovery_status") or "").upper() ==
        "ARMED" and
        int(position_state.get("dca_recovery_level", 0) or 0) == dca_count + 1
    ):
        return True

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

            if close_position_market(
                symbol,
                amount,
                position_side=position_side,
                reference_price=detail.get("mark_price"),
                context="TARGET_MARGIN_EXIT",
            ):
                cancel_open_protection_orders(symbol)
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

    request_shutdown()
    log_warning(
        "Target margin stop requested graceful process shutdown; "
        "monitor cleanup and telemetry flush will run"
    )


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
        thread = self.thread

        if (
            thread is not None and
            thread.is_alive() and
            thread is not threading.current_thread()
        ):
            thread.join()

        if thread is None or not thread.is_alive():
            self.thread = None

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
                    getattr(config, "EARLY_FLOW_EXIT_ENABLED", False) or
                    getattr(config, "TIME_EXIT_ENABLED", False)
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
        self.time_exit_check_times = {}
        self.time_exit_pending = set()
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
            self.time_exit_check_times = {
                symbol: checked_at
                for symbol, checked_at in self.time_exit_check_times.items()
                if symbol in active_symbols
            }
            self.time_exit_pending.intersection_update(active_symbols)
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

    def reconcile_multi_tp_positions(self, position_details, state):
        """Run the same TP1/runner recovery during scans and after restarts."""
        for symbol, detail in (position_details or {}).items():
            if get_pending_execution(state, symbol):
                continue

            position_state = get_position_state(state, symbol)

            if not position_state or not position_state.get("multi_tp_active"):
                continue

            mark_price = float(detail.get("mark_price", 0) or 0)

            if mark_price <= 0:
                continue

            try:
                self._handle_multi_tp_runner(symbol, mark_price, state)
            except Exception as exc:
                log_error(f"{symbol} scan TP runner reconciliation error: {exc}")

    def reconcile_position_management(self, position_details, state):
        """REST-scan fallback for websocket-owned position decisions."""
        blocked_symbols = set()

        for symbol, detail in (position_details or {}).items():
            if get_pending_execution(state, symbol):
                blocked_symbols.add(symbol)
                continue

            mark_price = float(detail.get("mark_price", 0) or 0)

            if mark_price <= 0:
                blocked_symbols.add(symbol)
                log_warning(
                    f"{symbol} position lifecycle deferred | mark price unavailable"
                )
                continue

            try:
                if runner_owns_position(get_position_state(state, symbol)):
                    blocked_symbols.add(symbol)
                    continue

                if self._handle_reversal_profit_protection(
                    symbol,
                    mark_price,
                    state,
                ):
                    blocked_symbols.add(symbol)
                    continue

                if self._handle_trend_profit_protection(
                    symbol,
                    mark_price,
                    state,
                ):
                    blocked_symbols.add(symbol)
                    continue

                if self._handle_route_early_invalidation(
                    symbol,
                    mark_price,
                    state,
                ):
                    blocked_symbols.add(symbol)
                    continue

                if self._handle_time_exit(symbol, mark_price, state):
                    blocked_symbols.add(symbol)
                    continue

                latest_position_state = get_position_state(
                    load_trade_state(),
                    symbol,
                )

                if position_exit_blocks_dca(latest_position_state):
                    blocked_symbols.add(symbol)
            except Exception as exc:
                blocked_symbols.add(symbol)
                log_error(f"{symbol} scan position lifecycle error: {exc}")

        return blocked_symbols

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

        if get_pending_execution(state, symbol):
            return

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

        if self._handle_time_exit(symbol, mark_price, state):
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

    def _multi_tp_retry_ready(self, symbol, retry_seconds=None):
        now = time.monotonic()
        retry_seconds = max(
            float(
                retry_seconds
                if retry_seconds is not None
                else getattr(config, "TP1_RUNNER_RETRY_SECONDS", 5)
            ),
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

    def _persist_multi_tp_updates(self, state, symbol, updates, error_label):
        if update_position_runtime_fields(state, symbol, updates):
            return True

        log_error(
            f"{symbol} TP runner state persistence failed | {error_label}"
        )
        return False

    @staticmethod
    def _algo_intent_matches(
        execution,
        order_type,
        close_side,
        position_side,
        close_position,
        quantity=None,
        trigger_price=None,
        reduce_only=None,
        working_type="MARK_PRICE",
    ):
        if not execution.get("found"):
            return False

        order = execution.get("algo_order") or {}
        actual_type = str(
            order.get("orderType") or order.get("type") or ""
        ).upper()
        actual_side = str(order.get("side") or "").upper()
        actual_position_side = str(
            order.get("positionSide") or "BOTH"
        ).upper()
        actual_close_position = str(
            order.get("closePosition", False)
        ).lower() == "true"
        actual_reduce_only = str(
            order.get("reduceOnly", False)
        ).lower() == "true"
        actual_working_type = str(order.get("workingType") or "").upper()

        if actual_type != str(order_type).upper():
            return False

        if actual_side != str(close_side).upper():
            return False

        if actual_position_side != str(position_side or "BOTH").upper():
            return False

        if actual_close_position != bool(close_position):
            return False

        if reduce_only is not None and actual_reduce_only != bool(reduce_only):
            return False

        if working_type and actual_working_type != str(working_type).upper():
            return False

        if trigger_price is not None:
            actual_trigger = float(
                order.get("triggerPrice") or order.get("stopPrice") or 0
            )
            expected_trigger = float(trigger_price or 0)
            tolerance = max(abs(expected_trigger) * 1e-12, 1e-12)

            if (
                actual_trigger <= 0 or
                abs(actual_trigger - expected_trigger) > tolerance
            ):
                return False

        if quantity is not None:
            expected_quantity = abs(float(quantity or 0))
            actual_quantity = abs(float(order.get("quantity", 0) or 0))
            tolerance = max(expected_quantity * 1e-6, 1e-12)

            if abs(actual_quantity - expected_quantity) > tolerance:
                return False

        return True

    def _resolve_exact_tp1_fill(
        self,
        symbol,
        state,
        position_state,
        position_detail,
    ):
        tp1_order_id = position_state.get("tp1_order_id") or ""
        side = str(position_state.get("side") or "").upper()
        close_side = SIDE_SELL if side == "BUY" else SIDE_BUY
        position_side = position_detail.get("position_side") or "BOTH"
        target_quantity = abs(
            float(position_state.get("tp1_quantity", 0) or 0)
        )
        current_order_quantity = abs(
            float(
                position_state.get("tp1_order_quantity") or
                target_quantity
            )
        )
        base_quantity = abs(
            float(position_state.get("tp1_base_quantity", 0) or 0)
        )
        live_quantity = abs(float(position_detail.get("quantity", 0) or 0))
        tolerance = max(target_quantity * 1e-6, 1e-12)

        if (
            side not in ("BUY", "SELL") or
            target_quantity <= 0 or
            current_order_quantity <= 0 or
            base_quantity <= 0
        ):
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "TP1_STATE_INVALID"},
                "TP1_STATE_INVALID",
            )
            return False, None

        execution = None

        if tp1_order_id:
            execution = get_algo_order_execution(symbol, tp1_order_id)

            if not execution.get("query_ok"):
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "TP1_ORDER_QUERY_UNAVAILABLE"},
                    "TP1_ORDER_QUERY_UNAVAILABLE",
                )
                return False, None

            if execution.get("found") and not self._algo_intent_matches(
                execution,
                "TAKE_PROFIT_MARKET",
                close_side,
                position_side,
                False,
                quantity=current_order_quantity,
                trigger_price=position_state.get("tp1_price"),
                reduce_only=(position_side == "BOTH"),
            ):
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "TP1_ORDER_INTENT_MISMATCH"},
                    "TP1_ORDER_INTENT_MISMATCH",
                )
                return False, None

        if execution and execution.get("found"):
            algo_status = execution.get("algo_status") or "UNKNOWN"

            if not self._persist_multi_tp_updates(
                state,
                symbol,
                {"tp1_order_status": algo_status},
                "TP1_ORDER_STATUS",
            ):
                return False, None

            if execution.get("open"):
                return False, execution

            execution_terminal = bool(
                execution.get("terminal") or execution.get("filled")
            )

            if not execution_terminal:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "TP1_ORDER_STATUS_AMBIGUOUS"},
                    "TP1_ORDER_STATUS_AMBIGUOUS",
                )
                return False, execution

            accounted_ids = list(
                position_state.get("tp1_accounted_order_ids") or []
            )
            cumulative_quantity = max(
                float(position_state.get("tp1_executed_quantity", 0) or 0),
                0,
            )
            cumulative_quote = max(
                float(position_state.get("tp1_executed_quote", 0) or 0),
                0,
            )

            if tp1_order_id not in accounted_ids:
                current_executed = max(
                    float(execution.get("executed_quantity", 0) or 0),
                    0,
                )

                # Backward-compatible test/adoption records may only expose
                # the exact-child FILLED flag. Production resolver always
                # supplies exact child executedQty.
                if execution.get("filled") and current_executed <= 0:
                    current_executed = current_order_quantity

                current_executed = min(
                    current_executed,
                    current_order_quantity,
                )
                actual_order = execution.get("actual_order") or {}
                algo_order = execution.get("algo_order") or {}
                current_fill_price = float(
                    actual_order.get("avgPrice") or
                    algo_order.get("actualPrice") or
                    0
                )
                cumulative_quantity = min(
                    cumulative_quantity + current_executed,
                    target_quantity,
                )

                if current_fill_price > 0 and current_executed > 0:
                    cumulative_quote += current_fill_price * current_executed

                accounted_ids.append(tp1_order_id)

                if not self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {
                        "tp1_accounted_order_ids": accounted_ids,
                        "tp1_executed_quantity": cumulative_quantity,
                        "tp1_executed_quote": cumulative_quote,
                    },
                    "TP1_EXECUTION_ACCOUNTING",
                ):
                    return False, execution

                position_state = get_position_state(state, symbol) or position_state
            else:
                cumulative_quantity = max(
                    float(
                        position_state.get("tp1_executed_quantity", 0) or 0
                    ),
                    0,
                )
                cumulative_quote = max(
                    float(position_state.get("tp1_executed_quote", 0) or 0),
                    0,
                )

            if (
                cumulative_quantity + tolerance >= target_quantity and
                tp1_fill_confirmed(
                    base_quantity,
                    target_quantity,
                    live_quantity,
                )
            ):
                resolved_execution = dict(execution)
                resolved_execution["tp1_average_fill_price"] = (
                    cumulative_quote / cumulative_quantity
                    if cumulative_quantity > 0 and cumulative_quote > 0
                    else 0
                )
                return True, resolved_execution

            observed_reduction = max(base_quantity - live_quantity, 0)

            if observed_reduction + tolerance < cumulative_quantity:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {
                        "runner_protection_error": (
                            "TP1_EXECUTION_POSITION_MISMATCH"
                        )
                    },
                    "TP1_EXECUTION_POSITION_MISMATCH",
                )
                return False, execution
        else:
            cumulative_quantity = max(
                float(position_state.get("tp1_executed_quantity", 0) or 0),
                0,
            )

        observed_reduction = max(base_quantity - live_quantity, 0)
        residual_quantity = min(
            max(target_quantity - cumulative_quantity, 0),
            max(target_quantity - observed_reduction, 0),
        )

        if residual_quantity <= tolerance:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {
                    "runner_protection_error": (
                        "TP1_EXACT_ATTRIBUTION_INCOMPLETE"
                    )
                },
                "TP1_EXACT_ATTRIBUTION_INCOMPLETE",
            )
            return False, execution

        matching = find_matching_open_algo_order(
            symbol,
            "TAKE_PROFIT_MARKET",
            close_side,
            position_side=position_side,
            trigger_price=position_state.get("tp1_price"),
            close_position=False,
            quantity=residual_quantity,
            reduce_only=(position_side == "BOTH"),
        )

        if matching.get("query_ok") is False:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "TP1_OPEN_ORDER_QUERY_UNAVAILABLE"},
                "TP1_OPEN_ORDER_QUERY_UNAVAILABLE",
            )
            return False, execution

        repaired_id = matching.get("order_id") or ""

        if repaired_id:
            if not self._persist_multi_tp_updates(
                state,
                symbol,
                {
                    "tp1_order_id": repaired_id,
                    "tp1_order_quantity": residual_quantity,
                    "tp1_order_status": "NEW",
                    "runner_protection_error": "",
                },
                "TP1_ORDER_ID_REPAIR",
            ):
                return False, execution

            return False, execution

        mark_price = float(position_detail.get("mark_price", 0) or 0)
        trigger_price = float(position_state.get("tp1_price", 0) or 0)
        trigger_ahead = (
            trigger_price > mark_price
            if side == "BUY"
            else trigger_price < mark_price
        )

        if not trigger_ahead:
            price_rules = get_symbol_price_rules(symbol)
            tick_size = max(
                float(price_rules.get("tick_size", 0) or 0),
                mark_price * 1e-6,
                1e-12,
            )
            raw_trigger = (
                mark_price + tick_size
                if side == "BUY"
                else mark_price - tick_size
            )
            trigger_price = normalize_trigger_price(
                symbol,
                SIDE_BUY if side == "BUY" else SIDE_SELL,
                "TAKE_PROFIT_MARKET",
                raw_trigger,
            )
            trigger_ahead = (
                trigger_price > mark_price
                if side == "BUY"
                else trigger_price < mark_price
            )

        if mark_price <= 0 or trigger_price <= 0 or not trigger_ahead:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "TP1_REPAIR_TRIGGER_INVALID"},
                "TP1_REPAIR_TRIGGER_INVALID",
            )
            return False, execution

        repair_count = int(position_state.get("tp1_repair_count", 0) or 0) + 1
        repair_updates = {
            "tp1_order_id": "",
            "tp1_order_quantity": residual_quantity,
            "tp1_order_status": "REPAIR_PENDING",
            "tp1_repair_count": repair_count,
            "runner_protection_error": "TP1_REPAIR_SUBMISSION_PENDING",
        }

        if abs(trigger_price - float(position_state.get("tp1_price", 0) or 0)) > 1e-12:
            repair_updates.update({
                "tp1_rearmed_from_price": position_state.get("tp1_price"),
                "tp1_price": trigger_price,
            })

        # Persist the exact replacement intent before submission. A restart
        # can then strictly adopt an acknowledged order without duplicating it.
        if not self._persist_multi_tp_updates(
            state,
            symbol,
            repair_updates,
            "TP1_REPAIR_INTENT",
        ):
            return False, execution

        try:
            repaired_order, placed_quantity = place_partial_take_profit_quantity(
                symbol,
                SIDE_BUY if side == "BUY" else SIDE_SELL,
                residual_quantity,
                trigger_price,
                position_side=position_side,
            )
            repaired_id = extract_order_id(repaired_order)
        except Exception as exc:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {
                    "runner_protection_error": (
                        f"TP1_REPAIR_SUBMISSION_ERROR: {exc}"
                    )
                },
                "TP1_REPAIR_SUBMISSION_ERROR",
            )
            return False, execution

        if (
            not repaired_id or
            abs(float(placed_quantity or 0) - residual_quantity) > tolerance
        ):
            if repaired_id:
                cancel_algo_order(symbol, repaired_id)

            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "TP1_REPAIR_ORDER_FAILED"},
                "TP1_REPAIR_ORDER_FAILED",
            )
            return False, execution

        if not self._persist_multi_tp_updates(
            state,
            symbol,
            {
                "tp1_order_id": repaired_id,
                "tp1_order_quantity": placed_quantity,
                "tp1_order_status": "NEW",
                "runner_protection_error": "",
            },
            "TP1_REPAIR_ORDER_ID",
        ):
            cancel_algo_order(symbol, repaired_id)

        return False, execution

    def _validate_runner_order_id(
        self,
        symbol,
        order_id,
        order_type,
        close_side,
        position_side,
        trigger_price=None,
    ):
        if not order_id:
            return "MISSING"

        execution = get_algo_order_execution(symbol, order_id)

        if not execution.get("query_ok") or execution.get("ambiguous"):
            return "UNAVAILABLE"

        if not self._algo_intent_matches(
            execution,
            order_type,
            close_side,
            position_side,
            True,
        ):
            return "INVALID"

        if trigger_price is not None:
            order = execution.get("algo_order") or {}
            actual_trigger = float(
                order.get("triggerPrice") or order.get("stopPrice") or 0
            )
            expected_trigger = float(trigger_price or 0)
            tolerance = max(abs(expected_trigger) * 1e-12, 1e-12)

            if (
                actual_trigger <= 0 or
                abs(actual_trigger - expected_trigger) > tolerance
            ):
                return "INVALID"

        if execution.get("open"):
            return "OPEN"

        if execution.get("terminal") or execution.get("filled"):
            return "TERMINAL"

        return "UNAVAILABLE"

    def _handle_multi_tp_runner(self, symbol, mark_price, state):
        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("multi_tp_active"):
            return False

        stage = position_state.get("multi_tp_stage")

        if stage == RUNNER_ACTIVE:
            lock = get_dca_lock(symbol)

            if not lock.acquire(blocking=False):
                return False

            try:
                if not self._multi_tp_retry_ready(symbol):
                    return False

                fresh_state = load_trade_state()
                fresh_position_state = get_position_state(fresh_state, symbol)
                details = get_open_position_details(symbol, force=True)
                position_detail = (details or {}).get(symbol)

                if not fresh_position_state or not position_detail:
                    return False

                side = str(fresh_position_state.get("side") or "").upper()
                close_side = SIDE_SELL if side == "BUY" else SIDE_BUY
                position_side = position_detail.get("position_side") or "BOTH"
                runner_tp_order_id = (
                    fresh_position_state.get("runner_tp_order_id") or ""
                )
                runner_sl_order_id = (
                    fresh_position_state.get("runner_sl_order_id") or ""
                )
                tp_state = self._validate_runner_order_id(
                    symbol,
                    runner_tp_order_id,
                    "TAKE_PROFIT_MARKET",
                    close_side,
                    position_side,
                    trigger_price=fresh_position_state.get("runner_tp_price"),
                )
                sl_required = bool(
                    getattr(config, "TP1_RUNNER_STOP_ENABLED", True)
                )
                sl_state = (
                    self._validate_runner_order_id(
                        symbol,
                        runner_sl_order_id,
                        "STOP_MARKET",
                        close_side,
                        position_side,
                        trigger_price=fresh_position_state.get("runner_sl_price"),
                    )
                    if sl_required
                    else "OPEN"
                )

                if "UNAVAILABLE" in (tp_state, sl_state):
                    self._persist_multi_tp_updates(
                        fresh_state,
                        symbol,
                        {"runner_protection_error": "RUNNER_ORDER_QUERY_UNAVAILABLE"},
                        "RUNNER_ORDER_QUERY_UNAVAILABLE",
                    )
                    return False

                repair_updates = {}

                if tp_state != "OPEN":
                    repair_updates["runner_tp_order_id"] = ""

                if sl_required and sl_state != "OPEN":
                    repair_updates["runner_sl_order_id"] = ""

                if repair_updates:
                    repair_updates.update({
                        "multi_tp_stage": RUNNER_PENDING,
                        "runner_protection_error": "RUNNER_ORDER_REPAIR_REQUIRED",
                    })
                    self._persist_multi_tp_updates(
                        fresh_state,
                        symbol,
                        repair_updates,
                        "RUNNER_ORDER_REPAIR_REQUIRED",
                    )
                    return False

                cleanup_updates = {}
                tp1_order_id = fresh_position_state.get("tp1_order_id") or ""

                if tp1_order_id:
                    tp1_execution = get_algo_order_execution(
                        symbol,
                        tp1_order_id,
                    )

                    if tp1_execution.get("query_ok") and (
                        not tp1_execution.get("found") or
                        tp1_execution.get("filled")
                    ):
                        cleanup_updates["tp1_order_id"] = ""

                old_sl_order_id = (
                    fresh_position_state.get("initial_sl_order_id") or ""
                )

                if old_sl_order_id and runner_sl_order_id:
                    if old_sl_order_id == runner_sl_order_id:
                        cleanup_updates["initial_sl_order_id"] = ""
                    else:
                        old_sl_state = self._validate_runner_order_id(
                            symbol,
                            old_sl_order_id,
                            "STOP_MARKET",
                            close_side,
                            position_side,
                        )

                        if old_sl_state == "OPEN" and cancel_algo_order(
                            symbol,
                            old_sl_order_id,
                        ):
                            cleanup_updates["initial_sl_order_id"] = ""
                        elif old_sl_state in ("INVALID", "TERMINAL"):
                            cleanup_updates["initial_sl_order_id"] = ""

                if cleanup_updates:
                    self._persist_multi_tp_updates(
                        fresh_state,
                        symbol,
                        cleanup_updates,
                        "RUNNER_ACTIVE_CLEANUP",
                    )
            finally:
                lock.release()

            return False

        if stage not in (TP1_PENDING, RUNNER_PENDING):
            return False

        quantity_reduction_seen = self._synced_multi_tp_fill_confirmed(
            symbol,
            position_state,
        )
        trigger_seen = bool(position_state.get("tp1_trigger_seen_at"))
        trigger_now = tp1_trigger_reached(
            position_state.get("side"),
            mark_price,
            position_state.get("tp1_price"),
        )

        transition_expected = bool(
            stage == RUNNER_PENDING or
            trigger_seen or
            trigger_now or
            quantity_reduction_seen
        )
        check_seconds = (
            getattr(config, "TP1_RUNNER_RETRY_SECONDS", 5)
            if transition_expected
            else getattr(config, "TP1_HEALTH_CHECK_SECONDS", 60)
        )

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(f"{symbol} TP1 runner transition deferred | position busy")
            return False

        try:
            # Ordinary health checks may be throttled before disk/API work. A
            # TP1 touch/fill candidate must first be durably latched under the
            # symbol lock so a stale websocket state cannot reserve recovery.
            if (
                not transition_expected and
                not self._multi_tp_retry_ready(symbol, check_seconds)
            ):
                return False

            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if not fresh_position_state or not fresh_position_state.get(
                "multi_tp_active"
            ):
                return False

            fresh_stage = fresh_position_state.get("multi_tp_stage")

            if fresh_stage == RUNNER_ACTIVE:
                return False

            fresh_trigger_seen = bool(
                fresh_position_state.get("tp1_trigger_seen_at")
            )
            fresh_trigger_now = tp1_trigger_reached(
                fresh_position_state.get("side"),
                mark_price,
                fresh_position_state.get("tp1_price"),
            )
            fresh_quantity_reduction_seen = (
                self._synced_multi_tp_fill_confirmed(
                    symbol,
                    fresh_position_state,
                )
            )

            if (
                fresh_stage == TP1_PENDING and
                (fresh_trigger_now or fresh_quantity_reduction_seen) and
                not fresh_trigger_seen
            ):
                if not self._persist_multi_tp_updates(
                    fresh_state,
                    symbol,
                    {
                        "tp1_trigger_seen_at": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                    },
                    "TP1_TRIGGER_LATCH",
                ):
                    shutdown_event.set()
                    return True

                fresh_position_state = get_position_state(fresh_state, symbol)

            state["positions"] = fresh_state.get("positions", {})
            state["pending_executions"] = fresh_state.get(
                "pending_executions",
                {},
            )

            if (
                transition_expected and
                not self._multi_tp_retry_ready(symbol, check_seconds)
            ):
                return False

            details = get_open_position_details(symbol, force=True)
            position_detail = (details or {}).get(symbol)

            if not position_detail:
                log_warning(
                    f"{symbol} TP1 runner transition skipped | "
                    "live position not found"
                )
                return False

            live_quantity = abs(float(position_detail.get("quantity", 0) or 0))

            if fresh_stage == TP1_PENDING:
                fill_confirmed, tp1_execution = self._resolve_exact_tp1_fill(
                    symbol,
                    fresh_state,
                    fresh_position_state,
                    position_detail,
                )

                if not fill_confirmed:
                    return False

                algo_order = (tp1_execution or {}).get("algo_order") or {}
                runner_basis = float(
                    (tp1_execution or {}).get("tp1_average_fill_price") or
                    algo_order.get("actualPrice") or
                    fresh_position_state.get("tp1_price") or
                    mark_price
                )
                actual_price = float(
                    algo_order.get("actualPrice") or runner_basis
                )

                if not self._persist_multi_tp_updates(
                    fresh_state,
                    symbol,
                    {
                        "multi_tp_stage": RUNNER_PENDING,
                        "tp1_filled_at": datetime.now().isoformat(
                            timespec="seconds"
                        ),
                        "tp1_fill_price": actual_price,
                        "runner_basis_price": runner_basis,
                        "runner_quantity": live_quantity,
                        "runner_protection_error": "",
                    },
                    "TP1_FILL_TRANSITION",
                ):
                    return False

                fresh_position_state = get_position_state(fresh_state, symbol)
                tp1_order_id = fresh_position_state.get("tp1_order_id") or ""

                if tp1_order_id and not cancel_algo_order(
                    symbol,
                    tp1_order_id,
                ):
                    self._persist_multi_tp_updates(
                        fresh_state,
                        symbol,
                        {"runner_protection_error": "TP1_CANCEL_FAILED"},
                        "TP1_CANCEL_FAILED",
                    )
                    return False

                if tp1_order_id:
                    if not self._persist_multi_tp_updates(
                        fresh_state,
                        symbol,
                        {"tp1_order_id": ""},
                        "TP1_ID_CLEAR",
                    ):
                        return False

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
            return False

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
        """Idempotently install and persist runner SL first, then TP2."""
        side = str(position_state.get("side") or "").upper()

        if side not in ("BUY", "SELL"):
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "RUNNER_SIDE_INVALID"},
                "RUNNER_SIDE_INVALID",
            )
            return False

        order_side = SIDE_BUY if side == "BUY" else SIDE_SELL
        close_side = SIDE_SELL if side == "BUY" else SIDE_BUY
        position_side = position_detail.get("position_side") or "BOTH"
        original_entry = float(
            position_state.get("avg_entry") or
            position_state.get("initial_entry") or
            0
        )
        runner_basis = float(
            position_state.get("runner_basis_price") or mark_price or 0
        )
        signal_type = (
            position_state.get("confirmation_type") or
            position_state.get("signal_type")
        )
        sl_required = bool(
            getattr(config, "TP1_RUNNER_STOP_ENABLED", True)
        )

        if original_entry <= 0 or runner_basis <= 0 or mark_price <= 0:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "RUNNER_BASIS_INVALID"},
                "RUNNER_BASIS_INVALID",
            )
            return False

        runner_tp_price = position_state.get("runner_tp_price")
        runner_tp_mode = position_state.get("runner_tp_mode") or ""
        runner_tp_context = position_state.get("runner_tp_context") or {}
        runner_sl_price = position_state.get("runner_sl_price")
        runner_sl_mode = position_state.get("runner_sl_mode") or ""
        trend_df = confirm_df = None

        if not runner_tp_price or (sl_required and not runner_sl_price):
            trend_df, confirm_df, _ = get_signal_frames(symbol, None)

        if not runner_tp_price:
            runner_tp_price, runner_tp_mode, runner_tp_context = (
                calculate_runner_take_profit(
                    symbol,
                    side,
                    runner_basis,
                    trend_df,
                    confirm_df,
                    signal_type=signal_type,
                )
            )

        runner_tp_price = normalize_trigger_price(
            symbol,
            order_side,
            "TAKE_PROFIT_MARKET",
            runner_tp_price,
        )

        runner_tp_order_id = position_state.get("runner_tp_order_id") or ""

        if runner_tp_order_id:
            tp_id_state = self._validate_runner_order_id(
                symbol,
                runner_tp_order_id,
                "TAKE_PROFIT_MARKET",
                close_side,
                position_side,
                trigger_price=runner_tp_price,
            )

            if tp_id_state == "UNAVAILABLE":
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_TP_QUERY_UNAVAILABLE"},
                    "RUNNER_TP_QUERY_UNAVAILABLE",
                )
                return False

            if tp_id_state != "OPEN":
                if not self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_tp_order_id": ""},
                    "RUNNER_TP_STALE_ID_CLEAR",
                ):
                    return False

                runner_tp_order_id = ""

        tp_is_ahead = (
            runner_tp_price > mark_price
            if side == "BUY"
            else runner_tp_price < mark_price
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

            runner_basis = float(mark_price)
            runner_tp_price = normalize_trigger_price(
                symbol,
                order_side,
                "TAKE_PROFIT_MARKET",
                roi_to_price(
                    side,
                    runner_basis,
                    fallback_roi,
                    leverage=config.LEVERAGE,
                ),
            )
            runner_tp_mode = f"TP2_REBASED_FALLBACK_ROI_{fallback_roi}%"
            runner_tp_context = {"reason": "TP2_REBASED_AFTER_TRIGGER"}

        if runner_tp_price <= 0:
            self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_protection_error": "RUNNER_TP_PRICE_INVALID"},
                "RUNNER_TP_PRICE_INVALID",
            )
            return False

        if sl_required and not runner_sl_price:
            if confirm_df is None:
                _, confirm_df, _ = get_signal_frames(symbol, None)

            runner_sl_price, sl_context = calculate_runner_stop(
                side,
                original_entry,
                mark_price,
                confirm_df,
                leverage=config.LEVERAGE,
            )

            if runner_sl_price is None:
                reason = sl_context.get("reason", "RUNNER_STOP_UNAVAILABLE")
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": reason},
                    reason,
                )
                return False

            runner_sl_mode = (
                f"RUNNER_{sl_context.get('source', 'PROFIT_LOCK')}"
            )

        if sl_required:
            runner_sl_price = normalize_trigger_price(
                symbol,
                order_side,
                "STOP_MARKET",
                runner_sl_price,
            )

            sl_is_valid = (
                runner_sl_price < mark_price
                if side == "BUY"
                else runner_sl_price > mark_price
            )

            if runner_sl_price <= 0 or not sl_is_valid:
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
                    self._persist_multi_tp_updates(
                        state,
                        symbol,
                        {"runner_protection_error": reason},
                        reason,
                    )
                    return False

                runner_sl_price = normalize_trigger_price(
                    symbol,
                    order_side,
                    "STOP_MARKET",
                    runner_sl_price,
                )
                runner_sl_mode = (
                    f"RUNNER_{sl_context.get('source', 'PROFIT_LOCK')}"
                )

        target_updates = {
            "runner_basis_price": runner_basis,
            "runner_quantity": abs(
                float(position_detail.get("quantity", 0) or 0)
            ),
            "runner_tp_price": runner_tp_price,
            "runner_tp_mode": runner_tp_mode,
            "runner_tp_context": runner_tp_context,
            "runner_sl_price": runner_sl_price if sl_required else None,
            "runner_sl_mode": runner_sl_mode if sl_required else "",
            "runner_protection_error": "",
        }

        if not self._persist_multi_tp_updates(
            state,
            symbol,
            target_updates,
            "RUNNER_TARGETS",
        ):
            return False

        position_state = get_position_state(state, symbol) or position_state
        runner_sl_order_id = position_state.get("runner_sl_order_id") or ""

        if sl_required and runner_sl_order_id:
            sl_id_state = self._validate_runner_order_id(
                symbol,
                runner_sl_order_id,
                "STOP_MARKET",
                close_side,
                position_side,
                trigger_price=runner_sl_price,
            )

            if sl_id_state == "UNAVAILABLE":
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_SL_QUERY_UNAVAILABLE"},
                    "RUNNER_SL_QUERY_UNAVAILABLE",
                )
                return False

            if sl_id_state != "OPEN":
                if not self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_sl_order_id": ""},
                    "RUNNER_SL_STALE_ID_CLEAR",
                ):
                    return False

                runner_sl_order_id = ""

        if sl_required and not runner_sl_order_id:
            matching_sl = find_matching_open_algo_order(
                symbol,
                "STOP_MARKET",
                close_side,
                position_side=position_side,
                trigger_price=runner_sl_price,
                close_position=True,
            )

            if matching_sl.get("query_ok") is False:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {
                        "runner_protection_error":
                        "RUNNER_SL_OPEN_ORDER_QUERY_UNAVAILABLE"
                    },
                    "RUNNER_SL_OPEN_ORDER_QUERY_UNAVAILABLE",
                )
                return False

            runner_sl_order_id = matching_sl.get("order_id") or ""
            placed_runner_sl = False

            if not runner_sl_order_id:
                sl_order = place_close_position_protection(
                    symbol,
                    order_side,
                    "STOP_MARKET",
                    runner_sl_price,
                    position_side=position_side,
                )
                runner_sl_order_id = extract_order_id(sl_order)
                placed_runner_sl = bool(runner_sl_order_id)

            if not runner_sl_order_id:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_SL_ORDER_FAILED"},
                    "RUNNER_SL_ORDER_FAILED",
                )
                return False

            if not self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_sl_order_id": runner_sl_order_id},
                "RUNNER_SL_ID",
            ):
                if placed_runner_sl:
                    cancel_algo_order(symbol, runner_sl_order_id)
                return False

        position_state = get_position_state(state, symbol) or position_state
        runner_tp_order_id = position_state.get("runner_tp_order_id") or ""

        if not runner_tp_order_id:
            matching_tp = find_matching_open_algo_order(
                symbol,
                "TAKE_PROFIT_MARKET",
                close_side,
                position_side=position_side,
                trigger_price=runner_tp_price,
                close_position=True,
            )

            if matching_tp.get("query_ok") is False:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {
                        "runner_protection_error":
                        "RUNNER_TP_OPEN_ORDER_QUERY_UNAVAILABLE"
                    },
                    "RUNNER_TP_OPEN_ORDER_QUERY_UNAVAILABLE",
                )
                return False

            runner_tp_order_id = matching_tp.get("order_id") or ""
            placed_runner_tp = False

            if not runner_tp_order_id:
                tp_order = place_close_position_protection(
                    symbol,
                    order_side,
                    "TAKE_PROFIT_MARKET",
                    runner_tp_price,
                    position_side=position_side,
                )
                runner_tp_order_id = extract_order_id(tp_order)
                placed_runner_tp = bool(runner_tp_order_id)

            if not runner_tp_order_id:
                self._persist_multi_tp_updates(
                    state,
                    symbol,
                    {"runner_protection_error": "RUNNER_TP_ORDER_FAILED"},
                    "RUNNER_TP_ORDER_FAILED",
                )
                return False

            if not self._persist_multi_tp_updates(
                state,
                symbol,
                {"runner_tp_order_id": runner_tp_order_id},
                "RUNNER_TP_ID",
            ):
                if placed_runner_tp:
                    cancel_algo_order(symbol, runner_tp_order_id)
                return False

        position_state = get_position_state(state, symbol) or position_state
        old_sl_order_id = position_state.get("initial_sl_order_id") or ""
        old_sl_remaining = old_sl_order_id

        if old_sl_order_id and runner_sl_order_id:
            if old_sl_order_id == runner_sl_order_id:
                old_sl_remaining = ""
            else:
                old_sl_state = self._validate_runner_order_id(
                    symbol,
                    old_sl_order_id,
                    "STOP_MARKET",
                    close_side,
                    position_side,
                )

                if old_sl_state == "OPEN":
                    if cancel_algo_order(symbol, old_sl_order_id):
                        old_sl_remaining = ""
                elif old_sl_state in ("INVALID", "TERMINAL"):
                    old_sl_remaining = ""

        active_updates = {
            "multi_tp_stage": RUNNER_ACTIVE,
            "initial_sl_order_id": old_sl_remaining,
            "tp_status": "CREATED",
            "tp_price": runner_tp_price,
            "tp_mode": runner_tp_mode,
            "tp_context": "TP2_RUNNER",
            "sl_status": (
                "CREATED"
                if sl_required
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
        }

        if not self._persist_multi_tp_updates(
            state,
            symbol,
            active_updates,
            "RUNNER_ACTIVATION",
        ):
            return False

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

        if not coordinated_position_management_enabled(position_state):
            return None

        if runner_owns_position(position_state):
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

    @staticmethod
    def _committed_early_invalidation_context(position_state, mark_price):
        """Rebuild an already-committed exit without re-opening its thesis."""
        if not position_state or runner_owns_position(position_state):
            return None

        route = str(
            position_state.get("early_invalidation_exit_route") or
            position_state.get("confirmation_type") or
            position_state.get("signal_type") or
            "TREND"
        ).upper()
        route = "REVERSAL" if route == "REVERSAL" else "TREND"
        side = str(position_state.get("side") or "").upper()
        avg_entry = _safe_float(position_state.get("avg_entry"))

        if side not in ("BUY", "SELL") or avg_entry <= 0 or mark_price <= 0:
            return None

        return {
            "route": route,
            "side": side,
            "avg_entry": avg_entry,
            "current_roi": -get_position_adverse_roi(side, avg_entry, mark_price),
            "max_roi": position_state.get("early_invalidation_exit_max_roi"),
            "reference_price": position_state.get("reference_price"),
        }

    def _handle_route_early_invalidation(self, symbol, mark_price, state):
        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("managed_by_bot"):
            return False

        exit_owner = committed_position_exit_owner(position_state)

        if exit_owner and exit_owner != "EARLY_INVALIDATION":
            return False

        exit_status = str(
            position_state.get("early_invalidation_exit_status") or ""
        ).upper()
        committed_exit = exit_status in ("PENDING", "UNCERTAIN", "FAILED")

        if exit_status == "SUBMITTED":
            return True

        if committed_exit and runner_owns_position(position_state):
            if not update_position_runtime_fields(
                state,
                symbol,
                {
                    "early_invalidation_exit_status":
                    "CANCELLED_RUNNER_OWNERSHIP",
                    "position_exit_owner": "",
                },
            ):
                log_error(
                    f"{symbol} early invalidation runner handoff was not persisted"
                )
                shutdown_event.set()
            return True

        check_seconds = max(
            float(getattr(config, "EARLY_FLOW_EXIT_CHECK_SECONDS", 60)),
            1,
        )

        if committed_exit and not durable_exit_retry_ready(
            position_state,
            "early_invalidation_exit_last_attempt_at",
            "early_invalidation_exit_pending_at",
            check_seconds,
        ):
            return True

        if (
            not committed_exit and
            not getattr(config, "EARLY_FLOW_EXIT_ENABLED", False)
        ):
            return False

        context = (
            self._committed_early_invalidation_context(position_state, mark_price)
            if committed_exit
            else self._route_early_invalidation_context(position_state, mark_price)
        )

        if not context:
            return committed_exit

        now = time.monotonic()
        with self.protection_lock:
            if symbol in self.route_exit_pending:
                return True

            last_check = float(
                self.route_invalidation_check_times.get(symbol, 0) or 0
            )

            if now - last_check < check_seconds:
                # Once the position is inside the configured adverse zone, a
                # throttled invalidation check owns this tick. Recovery must
                # not race ahead of evidence that was intentionally deferred.
                return True

            self.route_invalidation_check_times[symbol] = now

        fast_df = None
        slow_df = None

        if committed_exit:
            saved_evidence = (
                position_state.get("early_invalidation_exit_evidence") or {}
            )
            info = {
                "should_exit": True,
                "reason": (
                    position_state.get("early_invalidation_exit_reason") or
                    "EARLY_INVALIDATION_COMMITTED_RETRY"
                ),
                **saved_evidence,
            }
        else:
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
                fast_df = (
                    apply_indicators(fast_raw)
                    if fast_raw is not None
                    else None
                )
                slow_df = (
                    apply_indicators(slow_raw)
                    if slow_raw is not None
                    else None
                )
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
                        info.get("reason") ==
                        "EARLY_INVALIDATION_DATA_UNAVAILABLE" and
                        getattr(config, "EARLY_FLOW_EXIT_REQUIRE_DATA", True)
                    ):
                        log_warning(
                            f"{symbol} early invalidation skipped | "
                            "live data unavailable"
                        )
                        # Required invalidation evidence owns this adverse-zone
                        # tick; recovery cannot race ahead of missing data.
                        return True

                    return False

            except Exception as e:
                log_error(f"{symbol} early invalidation analysis error: {e}")
                return bool(
                    getattr(config, "EARLY_FLOW_EXIT_REQUIRE_DATA", True)
                )

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(f"{symbol} early invalidation deferred | position busy")
            return True

        fresh_state = None
        exit_intent_persisted = False

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if not fresh_position_state:
                return True

            fresh_status = str(
                fresh_position_state.get("early_invalidation_exit_status") or ""
            ).upper()

            if fresh_status == "SUBMITTED":
                return True

            fresh_exit_owner = committed_position_exit_owner(
                fresh_position_state
            )

            if fresh_exit_owner and fresh_exit_owner != "EARLY_INVALIDATION":
                return False

            fresh_committed = fresh_status in (
                "PENDING",
                "UNCERTAIN",
                "FAILED",
            )

            if fresh_committed and runner_owns_position(fresh_position_state):
                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {
                        "early_invalidation_exit_status":
                        "CANCELLED_RUNNER_OWNERSHIP",
                        "position_exit_owner": "",
                    },
                ):
                    shutdown_event.set()
                return True

            if fresh_committed and not durable_exit_retry_ready(
                fresh_position_state,
                "early_invalidation_exit_last_attempt_at",
                "early_invalidation_exit_pending_at",
                check_seconds,
            ):
                return True

            fresh_context = (
                self._committed_early_invalidation_context(
                    fresh_position_state,
                    mark_price,
                )
                if fresh_committed
                else self._route_early_invalidation_context(
                    fresh_position_state,
                    mark_price,
                )
            )

            if not fresh_context:
                return True

            if fresh_committed:
                saved_evidence = (
                    fresh_position_state.get(
                        "early_invalidation_exit_evidence"
                    ) or {}
                )
                info = {
                    "should_exit": True,
                    "reason": (
                        fresh_position_state.get(
                            "early_invalidation_exit_reason"
                        ) or "EARLY_INVALIDATION_COMMITTED_RETRY"
                    ),
                    **saved_evidence,
                }
            else:
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

            evidence = {
                "fast_failure": bool(info.get("fast_failure")),
                "slow_failure": bool(info.get("slow_failure")),
                "fast_adverse": bool(info.get("fast_adverse")),
                "slow_adverse": bool(info.get("slow_adverse")),
                "dual_opposition": bool(info.get("dual_opposition")),
                "reference_broken": bool(info.get("reference_broken")),
                "fast_support_score": (
                    (info.get("fast") or {}).get("support_score")
                    if info.get("fast") is not None
                    else info.get("fast_support_score")
                ),
                "slow_support_score": (
                    (info.get("slow") or {}).get("support_score")
                    if info.get("slow") is not None
                    else info.get("slow_support_score")
                ),
            }
            pending_updates = {
                "early_invalidation_exit_status": "PENDING",
                "position_exit_owner": "EARLY_INVALIDATION",
                "early_invalidation_exit_pending_at": (
                    fresh_position_state.get(
                        "early_invalidation_exit_pending_at"
                    ) or datetime.now().isoformat(timespec="seconds")
                ),
                "early_invalidation_exit_last_attempt_at": (
                    datetime.now().isoformat(timespec="seconds")
                ),
                "early_invalidation_exit_price": mark_price,
                "early_invalidation_exit_roi": fresh_context["current_roi"],
                "early_invalidation_exit_max_roi": fresh_context.get("max_roi"),
                "early_invalidation_exit_reason": info.get("reason"),
                "early_invalidation_exit_route": fresh_context["route"],
                "early_invalidation_exit_evidence": evidence,
            }

            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                pending_updates,
            ):
                log_error(
                    f"{symbol} early invalidation intent was not persisted"
                )
                shutdown_event.set()
                return True

            exit_intent_persisted = True

            details = get_open_position_details(symbol, force=True)

            if details is None:
                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {"early_invalidation_exit_status": "UNCERTAIN"},
                ):
                    shutdown_event.set()
                log_warning(
                    f"{symbol} early invalidation deferred | "
                    "position snapshot unavailable"
                )

                with self.protection_lock:
                    self.route_exit_pending.discard(symbol)

                return True

            position_detail = details.get(symbol)

            if not position_detail:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {"early_invalidation_exit_status": final_status},
                ):
                    shutdown_event.set()

                if not cleanup_ok:
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()
                    log_error(
                        f"{symbol} early invalidation found position flat but "
                        "protection cleanup was not verified"
                    )
                else:
                    log_warning(
                        f"{symbol} early invalidation exit already completed"
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
                reference_price=mark_price,
                context="EARLY_INVALIDATION_EXIT",
            )

            if closed:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"
                final_updates = {
                    "early_invalidation_exit_status": final_status,
                    "early_invalidation_exit_price": mark_price,
                    "early_invalidation_exit_roi": fresh_context["current_roi"],
                    "early_invalidation_exit_reason": info.get("reason"),
                    "early_invalidation_exit_route": fresh_context["route"],
                    "early_invalidation_exit_evidence": evidence,
                }

                if not cleanup_ok:
                    final_updates["early_invalidation_cleanup_error"] = (
                        "PROTECTION_CLEANUP_UNCONFIRMED"
                    )
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    final_updates,
                ):
                    log_error(
                        f"{symbol} early invalidation completion was not persisted"
                    )
                    shutdown_event.set()

                if cleanup_ok:
                    send_telegram_message(
                        f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                        f"{symbol} {fresh_context['route'].lower()} "
                        "early invalidation exit\n"
                        f"ROI: {fresh_context['current_roi']}%\n"
                        f"Reason: {info.get('reason')}"
                    )
                else:
                    with self.protection_lock:
                        self.route_exit_pending.discard(symbol)

                return True

            log_error(f"{symbol} early invalidation exit order failed")
            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                {"early_invalidation_exit_status": "FAILED"},
            ):
                shutdown_event.set()

            with self.protection_lock:
                self.route_exit_pending.discard(symbol)

            return True

        except Exception as e:
            log_error(f"{symbol} early invalidation exit error: {e}")

            status_saved = bool(
                exit_intent_persisted and
                fresh_state is not None and
                update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {"early_invalidation_exit_status": "UNCERTAIN"},
                )
            )

            if not status_saved:
                shutdown_event.set()
            else:
                with self.protection_lock:
                    self.route_exit_pending.discard(symbol)

            return True

        finally:
            lock.release()

    def _time_exit_context(self, position_state, mark_price):
        if not getattr(config, "TIME_EXIT_ENABLED", False):
            return None

        if (
            not position_state or
            not position_state.get("managed_by_bot") or
            not coordinated_position_management_enabled(position_state) or
            runner_owns_position(position_state)
        ):
            return None

        confirmation_type = str(
            position_state.get("confirmation_type") or
            position_state.get("signal_type") or
            ""
        ).upper()

        if confirmation_type == "REVERSAL":
            route = "REVERSAL"
        elif confirmation_type == "RANGE_REVERSION":
            route = "RANGE_REVERSION"
        else:
            route = "TREND"

        route_enabled = bool(
            getattr(
                config,
                f"TIME_EXIT_{route}_ENABLED",
                route == "TREND",
            )
        )

        if not route_enabled:
            return None

        side = str(position_state.get("side") or "").upper()
        avg_entry = _safe_float(position_state.get("avg_entry"))

        if side not in ("BUY", "SELL") or avg_entry <= 0 or mark_price <= 0:
            return None

        opened_elapsed = seconds_since(position_state.get("opened_at"))
        time_exit_minutes = (
            getattr(
                config,
                "TIME_EXIT_RANGE_REVERSION_MINUTES",
                config.TIME_EXIT_MINUTES,
            )
            if route == "RANGE_REVERSION"
            else config.TIME_EXIT_MINUTES
        )
        minimum_seconds = max(float(time_exit_minutes), 0) * 60

        if opened_elapsed is None or opened_elapsed < minimum_seconds:
            return None

        post_dca_grace = max(
            float(getattr(config, "TIME_EXIT_POST_DCA_GRACE_MINUTES", 0)),
            0,
        ) * 60
        last_dca_elapsed = seconds_since(position_state.get("last_dca_at"))

        if (
            position_state.get("last_dca_at") and
            post_dca_grace > 0 and
            last_dca_elapsed is not None and
            last_dca_elapsed < post_dca_grace
        ):
            return None

        current_roi = -get_position_adverse_roi(side, avg_entry, mark_price)
        max_roi_setting = (
            getattr(
                config,
                "TIME_EXIT_RANGE_REVERSION_MAX_ROI",
                getattr(config, "TIME_EXIT_MAX_ROI", 0),
            )
            if route == "RANGE_REVERSION"
            else getattr(config, "TIME_EXIT_MAX_ROI", 0)
        )
        max_roi = min(float(max_roi_setting), 0)

        if current_roi > max_roi:
            return None

        return {
            "route": route,
            "side": side,
            "avg_entry": avg_entry,
            "current_roi": current_roi,
            "max_roi": max_roi,
            "elapsed_minutes": round(opened_elapsed / 60, 1),
        }

    def _committed_time_exit_context(self, position_state, mark_price):
        if not position_state or runner_owns_position(position_state):
            return None

        side = str(position_state.get("side") or "").upper()
        avg_entry = _safe_float(position_state.get("avg_entry"))

        if side not in ("BUY", "SELL") or avg_entry <= 0 or mark_price <= 0:
            return None

        committed_confirmation_type = str(
            position_state.get("confirmation_type") or
            position_state.get("signal_type") or
            ""
        ).upper()

        if committed_confirmation_type == "REVERSAL":
            route = "REVERSAL"
        elif committed_confirmation_type == "RANGE_REVERSION":
            route = "RANGE_REVERSION"
        else:
            route = "TREND"

        opened_elapsed = seconds_since(position_state.get("opened_at"))
        return {
            "route": route,
            "side": side,
            "avg_entry": avg_entry,
            "current_roi": -get_position_adverse_roi(
                side,
                avg_entry,
                mark_price,
            ),
            "max_roi": min(float(getattr(config, "TIME_EXIT_MAX_ROI", 0)), 0),
            "elapsed_minutes": round((opened_elapsed or 0) / 60, 1),
        }

    def _handle_time_exit(self, symbol, mark_price, state):
        position_state = get_position_state(state, symbol)

        if not position_state:
            return False

        exit_owner = committed_position_exit_owner(position_state)

        if exit_owner and exit_owner != "TIME":
            return False

        time_exit_status = str(
            position_state.get("time_exit_status") or ""
        ).upper()
        committed_exit = time_exit_status in (
            "PENDING",
            "UNCERTAIN",
            "FAILED",
        )

        if time_exit_status == "SUBMITTED":
            return True

        if committed_exit and runner_owns_position(position_state):
            if not update_position_runtime_fields(
                state,
                symbol,
                {
                    "time_exit_status": "CANCELLED_RUNNER_OWNERSHIP",
                    "position_exit_owner": "",
                },
            ):
                log_error(f"{symbol} time-exit runner handoff was not persisted")
                shutdown_event.set()
            return True

        retry_seconds = max(
            float(getattr(config, "TIME_EXIT_PENDING_RETRY_SECONDS", 60)),
            1,
        )

        if committed_exit and not durable_exit_retry_ready(
            position_state,
            "time_exit_last_attempt_at",
            "time_exit_pending_at",
            retry_seconds,
        ):
            return True

        context = (
            self._committed_time_exit_context(position_state, mark_price)
            if committed_exit
            else self._time_exit_context(position_state, mark_price)
        )

        if not context:
            return committed_exit

        now = time.monotonic()
        check_seconds = max(
            float(getattr(config, "TIME_EXIT_CHECK_SECONDS", 60)),
            1,
        )

        with self.protection_lock:
            if symbol in self.time_exit_pending:
                return True

            last_check = float(self.time_exit_check_times.get(symbol, 0) or 0)

            if now - last_check < check_seconds:
                return True

            self.time_exit_check_times[symbol] = now

        if committed_exit:
            weakness = {
                "should_exit": True,
                "reason": (
                    position_state.get("time_exit_reason") or
                    "TIME_EXIT_COMMITTED_RETRY"
                ),
                "evidence": position_state.get("time_exit_evidence") or [],
                "weakness_score": position_state.get(
                    "time_exit_weakness_score",
                    0,
                ),
            }
        else:
            trend_df, confirm_df, _ = get_signal_frames(symbol, None)
            weakness = evaluate_time_exit_weakness(
                context["side"],
                trend_df,
                confirm_df,
            )
        require_weakness = bool(
            getattr(config, "TIME_EXIT_REQUIRE_WEAKNESS", True)
        )

        if weakness.get("reason") == "TIME_EXIT_DATA_UNAVAILABLE":
            if getattr(config, "TIME_EXIT_REQUIRE_DATA", True):
                log_warning(
                    f"{symbol} time exit deferred | confirmation data unavailable"
                )
                return True

        if require_weakness and not weakness.get("should_exit"):
            return False

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(f"{symbol} time exit deferred | position busy")
            return True

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)
            fresh_status = str(
                (fresh_position_state or {}).get("time_exit_status") or ""
            ).upper()

            if fresh_status == "SUBMITTED":
                return True

            fresh_exit_owner = committed_position_exit_owner(
                fresh_position_state
            )

            if fresh_exit_owner and fresh_exit_owner != "TIME":
                return False

            fresh_committed = fresh_status in (
                "PENDING",
                "UNCERTAIN",
                "FAILED",
            )

            if fresh_committed and runner_owns_position(fresh_position_state):
                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {
                        "time_exit_status": "CANCELLED_RUNNER_OWNERSHIP",
                        "position_exit_owner": "",
                    },
                ):
                    shutdown_event.set()
                return True

            if fresh_committed and not durable_exit_retry_ready(
                fresh_position_state,
                "time_exit_last_attempt_at",
                "time_exit_pending_at",
                retry_seconds,
            ):
                return True

            fresh_context = (
                self._committed_time_exit_context(
                    fresh_position_state,
                    mark_price,
                )
                if fresh_committed
                else self._time_exit_context(
                    fresh_position_state,
                    mark_price,
                )
            )

            if not fresh_context:
                return True

            if fresh_committed:
                weakness = {
                    "should_exit": True,
                    "reason": (
                        fresh_position_state.get("time_exit_reason") or
                        "TIME_EXIT_COMMITTED_RETRY"
                    ),
                    "evidence": (
                        fresh_position_state.get("time_exit_evidence") or []
                    ),
                    "weakness_score": fresh_position_state.get(
                        "time_exit_weakness_score",
                        0,
                    ),
                }

            with self.protection_lock:
                if symbol in self.time_exit_pending:
                    return True

                self.time_exit_pending.add(symbol)

            pending_updates = {
                "time_exit_status": "PENDING",
                "position_exit_owner": "TIME",
                "time_exit_pending_at": (
                    fresh_position_state.get("time_exit_pending_at") or
                    datetime.now().isoformat(timespec="seconds")
                ),
                "time_exit_last_attempt_at": (
                    datetime.now().isoformat(timespec="seconds")
                ),
                "time_exit_reason": weakness.get("reason"),
                "time_exit_evidence": weakness.get("evidence", []),
                "time_exit_weakness_score": weakness.get("weakness_score", 0),
                "time_exit_elapsed_minutes": fresh_context["elapsed_minutes"],
                "time_exit_roi": fresh_context["current_roi"],
            }

            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                pending_updates,
            ):
                log_error(f"{symbol} time-exit state persistence failed")
                shutdown_event.set()
                return True

            details = get_open_position_details(symbol, force=True)

            if details is None:
                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {"time_exit_status": "UNCERTAIN"},
                ):
                    shutdown_event.set()
                log_warning(
                    f"{symbol} time exit deferred | position snapshot unavailable"
                )
                return True

            position_detail = details.get(symbol)

            if not position_detail:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {"time_exit_status": final_status},
                ):
                    shutdown_event.set()

                if not cleanup_ok:
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()
                    log_error(
                        f"{symbol} time exit found position flat but protection "
                        "cleanup was not verified"
                    )
                return True

            amount = float(position_detail.get("amount", 0) or 0)
            log_warning(
                f"{symbol} {fresh_context['route']} TIME EXIT | "
                f"AGE={fresh_context['elapsed_minutes']}m | "
                f"ROI={fresh_context['current_roi']}% | "
                f"WEAKNESS={weakness.get('weakness_score')}"
            )
            closed = close_position_market(
                symbol,
                amount,
                position_side=position_detail.get("position_side"),
                reference_price=mark_price,
                context="TIME_EXIT",
            )

            if closed:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"
                final_updates = {
                    "time_exit_status": final_status,
                    "time_exit_price": mark_price,
                }

                if not cleanup_ok:
                    final_updates["time_exit_cleanup_error"] = (
                        "PROTECTION_CLEANUP_UNCONFIRMED"
                    )
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    final_updates,
                ):
                    shutdown_event.set()

                if cleanup_ok:
                    send_telegram_message(
                        f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                        f"{symbol} time exit\n"
                        f"Age: {fresh_context['elapsed_minutes']} minutes\n"
                        f"ROI: {fresh_context['current_roi']}%\n"
                        f"Evidence: {', '.join(weakness.get('evidence', []))}"
                    )
                return True

            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                {"time_exit_status": "FAILED"},
            ):
                shutdown_event.set()
            log_error(f"{symbol} time exit order was not confirmed")
            return True
        finally:
            with self.protection_lock:
                self.time_exit_pending.discard(symbol)

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
        position_state = get_position_state(state, symbol)

        if not position_state or not position_state.get("managed_by_bot"):
            return False

        exit_status_field = f"{route_key}_profit_exit_status"
        current_exit_owner = f"{route}_PROFIT"
        exit_owner = committed_position_exit_owner(position_state)

        if exit_owner and exit_owner != current_exit_owner:
            return False

        exit_status = str(position_state.get(exit_status_field) or "").upper()
        committed_exit = exit_status in ("PENDING", "UNCERTAIN", "FAILED")

        if exit_status == "SUBMITTED":
            return True

        if not enabled and not committed_exit:
            return False

        if runner_owns_position(position_state):
            # TP2 and the runner SL exclusively own profit-taking only after
            # the TP1 partial fill has been confirmed.
            if committed_exit and not update_position_runtime_fields(
                state,
                symbol,
                {
                    exit_status_field: "CANCELLED_RUNNER_OWNERSHIP",
                    "position_exit_owner": "",
                },
            ):
                log_error(
                    f"{symbol} {route_key} profit runner handoff was not persisted"
                )
                shutdown_event.set()
            return committed_exit

        signal_type = str(
            position_state.get("confirmation_type") or
            position_state.get("signal_type") or
            ""
        ).upper()

        if signal_type != route and not committed_exit:
            return False

        retry_seconds = max(
            float(
                getattr(
                    config,
                    "PROFIT_EXIT_PENDING_RETRY_SECONDS",
                    60,
                )
            ),
            1,
        )

        if committed_exit and not durable_exit_retry_ready(
            position_state,
            f"{route_key}_profit_exit_last_attempt_at",
            f"{route_key}_profit_exit_pending_at",
            retry_seconds,
        ):
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

        if committed_exit:
            info = {
                "should_exit": True,
                "armed": True,
                "current_roi": -get_position_adverse_roi(
                    side,
                    avg_entry,
                    mark_price,
                ),
                "peak_roi": saved_peak,
                "floor_roi": position_state.get(
                    f"{route_key}_profit_floor_roi"
                ),
                "reason": (
                    position_state.get(f"{route_key}_profit_exit_reason") or
                    f"{route}_PROFIT_EXIT_COMMITTED_RETRY"
                ),
                "trigger_roi": 0,
            }
        else:
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
        should_persist = not committed_exit and (
            peak_roi >= saved_peak + persist_step or
            (peak_roi >= trigger_roi > saved_peak) or
            abs(saved_basis - avg_entry) > basis_tolerance
        )

        if not info.get("should_exit") and not should_persist:
            return False

        lock = get_dca_lock(symbol)

        if not lock.acquire(blocking=False):
            log_info(
                f"{symbol} {route_key} profit exit deferred | position busy"
            )
            return bool(info.get("should_exit"))

        fresh_state = None
        exit_intent_persisted = False

        try:
            fresh_state = load_trade_state()
            fresh_position_state = get_position_state(fresh_state, symbol)

            if not fresh_position_state:
                return True

            fresh_status = str(
                fresh_position_state.get(exit_status_field) or ""
            ).upper()

            if fresh_status == "SUBMITTED":
                return True

            fresh_exit_owner = committed_position_exit_owner(
                fresh_position_state
            )

            if fresh_exit_owner and fresh_exit_owner != current_exit_owner:
                return False

            if runner_owns_position(fresh_position_state):
                if fresh_status in ("PENDING", "UNCERTAIN", "FAILED"):
                    if not update_position_runtime_fields(
                        fresh_state,
                        symbol,
                        {
                            exit_status_field: "CANCELLED_RUNNER_OWNERSHIP",
                            "position_exit_owner": "",
                        },
                    ):
                        shutdown_event.set()
                return True

            fresh_committed = fresh_status in (
                "PENDING",
                "UNCERTAIN",
                "FAILED",
            )

            if fresh_committed and not durable_exit_retry_ready(
                fresh_position_state,
                f"{route_key}_profit_exit_last_attempt_at",
                f"{route_key}_profit_exit_pending_at",
                retry_seconds,
            ):
                return True

            if not enabled and not fresh_committed:
                return False

            fresh_signal_type = str(
                fresh_position_state.get("confirmation_type") or
                fresh_position_state.get("signal_type") or
                ""
            ).upper()

            if fresh_signal_type != route and not fresh_committed:
                return False

            side = fresh_position_state.get("side")
            avg_entry = float(fresh_position_state.get("avg_entry") or 0)
            saved_peak = float(fresh_position_state.get(peak_field) or 0)
            saved_basis = float(
                fresh_position_state.get(basis_field) or avg_entry
            )
            basis_tolerance = max(abs(avg_entry) * 1e-10, 1e-10)

            if abs(saved_basis - avg_entry) > basis_tolerance:
                saved_peak = 0

            with self.protection_lock:
                memory_basis = float(
                    basis_map.get(symbol, avg_entry) or avg_entry
                )
                memory_peak = float(peak_map.get(symbol, 0) or 0)

                if abs(memory_basis - avg_entry) > basis_tolerance:
                    memory_peak = 0

                previous_peak = max(saved_peak, memory_peak)

            if fresh_committed:
                info = {
                    "should_exit": True,
                    "armed": True,
                    "current_roi": -get_position_adverse_roi(
                        side,
                        avg_entry,
                        mark_price,
                    ),
                    "peak_roi": saved_peak,
                    "floor_roi": fresh_position_state.get(
                        f"{route_key}_profit_floor_roi"
                    ),
                    "reason": (
                        fresh_position_state.get(
                            f"{route_key}_profit_exit_reason"
                        ) or f"{route}_PROFIT_EXIT_COMMITTED_RETRY"
                    ),
                    "trigger_roi": 0,
                }
            else:
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

            trigger_roi = float(info.get("trigger_roi", 0) or 0)
            should_persist = not fresh_committed and (
                peak_roi >= saved_peak + persist_step or
                (peak_roi >= trigger_roi > saved_peak) or
                abs(saved_basis - avg_entry) > basis_tolerance
            )

            if not info.get("should_exit"):
                if should_persist and not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {
                        peak_field: round(peak_roi, 2),
                        basis_field: avg_entry,
                        f"{route_key}_profit_floor_roi": info.get("floor_roi"),
                        f"{route_key}_profit_armed": bool(info.get("armed")),
                    },
                ):
                    log_error(
                        f"{symbol} {route_key} profit peak was not persisted"
                    )
                return False

            with self.protection_lock:
                if symbol in pending:
                    return True

                pending.add(symbol)

            pending_updates = {
                peak_field: round(peak_roi, 2),
                basis_field: avg_entry,
                exit_status_field: "PENDING",
                "position_exit_owner": current_exit_owner,
                f"{route_key}_profit_exit_pending_at": (
                    fresh_position_state.get(
                        f"{route_key}_profit_exit_pending_at"
                    ) or datetime.now().isoformat(timespec="seconds")
                ),
                f"{route_key}_profit_exit_last_attempt_at": (
                    datetime.now().isoformat(timespec="seconds")
                ),
                f"{route_key}_profit_exit_price": mark_price,
                f"{route_key}_profit_exit_roi": info.get("current_roi"),
                f"{route_key}_profit_exit_reason": info.get("reason"),
                f"{route_key}_profit_floor_roi": info.get("floor_roi"),
            }

            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                pending_updates,
            ):
                log_error(
                    f"{symbol} {route_key} profit exit intent was not persisted"
                )
                shutdown_event.set()
                return True

            exit_intent_persisted = True
            details = get_open_position_details(symbol, force=True)

            if details is None:
                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {exit_status_field: "UNCERTAIN"},
                ):
                    shutdown_event.set()

                with self.protection_lock:
                    pending.discard(symbol)

                return True

            position_detail = details.get(symbol)

            if not position_detail:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {exit_status_field: final_status},
                ):
                    shutdown_event.set()

                if not cleanup_ok:
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()
                    log_error(
                        f"{symbol} {route_key} profit exit found position flat "
                        "but protection cleanup was not verified"
                    )
                else:
                    log_warning(
                        f"{symbol} {route_key} profit exit already completed"
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
                reference_price=mark_price,
                context=f"{route}_PROFIT_EXIT",
            )

            if closed:
                cleanup_ok = cancel_open_protection_orders(symbol)
                final_status = "SUBMITTED" if cleanup_ok else "UNCERTAIN"
                final_updates = {
                    peak_field: round(peak_roi, 2),
                    basis_field: avg_entry,
                    exit_status_field: final_status,
                    f"{route_key}_profit_exit_price": mark_price,
                    f"{route_key}_profit_exit_roi": info.get("current_roi"),
                    f"{route_key}_profit_exit_reason": info.get("reason"),
                }

                if not cleanup_ok:
                    final_updates[f"{route_key}_profit_cleanup_error"] = (
                        "PROTECTION_CLEANUP_UNCONFIRMED"
                    )
                    entry_quarantined_symbols.add(symbol)
                    shutdown_event.set()

                if not update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    final_updates,
                ):
                    log_error(
                        f"{symbol} {route_key} profit exit completion was not "
                        "persisted"
                    )
                    shutdown_event.set()

                if cleanup_ok:
                    send_telegram_message(
                        f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                        f"{symbol} {route_key} profit protected\n"
                        f"ROI: {info.get('current_roi')}%\n"
                        f"Peak ROI: {info.get('peak_roi')}%\n"
                        f"Protection floor: {info.get('floor_roi')}%"
                    )
                else:
                    with self.protection_lock:
                        pending.discard(symbol)

                return True

            log_error(f"{symbol} {route_key} profit exit order failed")
            if not update_position_runtime_fields(
                fresh_state,
                symbol,
                {exit_status_field: "FAILED"},
            ):
                shutdown_event.set()

            with self.protection_lock:
                pending.discard(symbol)

            return True

        except Exception as e:
            log_error(f"{symbol} {route_key} profit protection error: {e}")

            status_saved = bool(
                exit_intent_persisted and
                fresh_state is not None and
                update_position_runtime_fields(
                    fresh_state,
                    symbol,
                    {exit_status_field: "UNCERTAIN"},
                )
            )

            if not status_saved:
                shutdown_event.set()
            else:
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
    uncapped_limit = max(
        _safe_float(
            getattr(config, "SIGNAL_RANKING_UNCAPPED_INDEX_MAX", 140),
            140,
        ),
        100,
    )
    score_index = _safe_float(
        side_data.get("uncapped_score_index"),
        side_data.get("confidence", analysis.get("best_confidence", 0)),
    )
    rank = min(score_index, uncapped_limit)

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

    market_context = candidate.get("market_context") or {}
    flow = market_context.get("flow") or {}
    breadth = market_context.get("breadth") or {}
    transition = market_context.get("transition") or {}
    calibration = market_context.get("calibration") or {}
    order_flow = market_context.get("order_flow") or {}
    side_key = (signal or "").lower()
    rank += _safe_float(flow.get(f"{side_key}_score")) * _safe_float(
        getattr(config, "MARKET_FLOW_RANK_WEIGHT", 1),
        1,
    )
    rank += _safe_float(breadth.get(f"{side_key}_score")) * _safe_float(
        getattr(config, "MARKET_BREADTH_RANK_WEIGHT", 1),
        1,
    )
    rank += _safe_float(transition.get(f"{side_key}_score")) * _safe_float(
        getattr(config, "REGIME_TRANSITION_RANK_WEIGHT", 1),
        1,
    )

    if getattr(config, "SIGNAL_RANKING_ORDERFLOW_ENABLED", False):
        rank += _safe_float(
            order_flow.get(f"{side_key}_shadow_score")
        ) * _safe_float(
            getattr(config, "SIGNAL_RANKING_ORDERFLOW_WEIGHT", 1.0),
            1.0,
        )

    if calibration.get("available"):
        probability = _safe_float(calibration.get("probability"), 0.5)
        rank += (probability - 0.5) * _safe_float(
            getattr(config, "SIGNAL_CALIBRATION_RANK_WEIGHT", 8),
            8,
        )

    return round(rank, 2)


def enrich_candidate_market_context(
    candidate,
    flow_monitor,
    breadth_context,
    shadow_monitor=None,
):
    signal = str(candidate.get("signal") or "").upper()
    analysis = candidate.get("analysis") or {}
    side_data = analysis.get(signal.lower(), {}) or {}
    flow = (
        flow_monitor.snapshot(candidate.get("symbol"))
        if flow_monitor and getattr(config, "MARKET_FLOW_ENABLED", True)
        else {"available": False, "buy_score": 0, "sell_score": 0}
    )
    breadth = (
        breadth_context
        if getattr(config, "MARKET_BREADTH_ENABLED", True)
        else {"available": False, "buy_score": 0, "sell_score": 0}
    )
    transition = (
        calculate_regime_transition(candidate.get("entry_df"))
        if getattr(config, "REGIME_TRANSITION_ENABLED", True)
        else {"available": False, "buy_score": 0, "sell_score": 0}
    )
    order_flow = (
        shadow_monitor.snapshot(candidate.get("symbol"), emit_telemetry=False)
        if shadow_monitor and getattr(
            config,
            "SIGNAL_RANKING_ORDERFLOW_ENABLED",
            False,
        )
        else {"available": False, "buy_shadow_score": 0, "sell_shadow_score": 0}
    )
    route = get_candidate_signal_type(candidate)

    if (side_data.get("continuation_pullback") or {}).get("active"):
        route = "CONTINUATION_PULLBACK"
    elif (side_data.get("trend_timing_rescue") or {}).get("active"):
        route = "TREND_TIMING_RESCUE"

    calibration = calibration_probability(route, side_data.get("score", 0))
    candidate["market_context"] = {
        "flow": flow,
        "breadth": breadth,
        "transition": transition,
        "calibration": calibration,
        "order_flow": order_flow,
        "route": route,
    }
    candidate["rank_score"] = calculate_signal_rank(candidate)
    return candidate


def attach_candidate_shadow_order_flow(candidate, shadow_monitor):
    """Attach observation-only analytics after decision scores are final."""
    if shadow_monitor is None:
        candidate["shadow_order_flow"] = {
            "shadow_only": True,
            "decision_effect": False,
            "ranking_effect": False,
            "available": False,
            "reason": "SHADOW_MONITOR_DISABLED",
        }
        return candidate

    try:
        candidate["shadow_order_flow"] = shadow_monitor.snapshot(
            candidate.get("symbol"),
            context={
                "signal_side": candidate.get("signal"),
                "route": (candidate.get("market_context") or {}).get("route"),
                "rank_score": candidate.get("rank_score"),
            },
        )
    except Exception as exc:
        log_warning(
            f"{candidate.get('symbol')} shadow order-flow snapshot warning: {exc}"
        )
        candidate["shadow_order_flow"] = {
            "shadow_only": True,
            "decision_effect": False,
            "ranking_effect": False,
            "available": False,
            "reason": "SHADOW_SNAPSHOT_ERROR",
            "error": str(exc),
        }

    return candidate


def market_flow_hard_veto(candidate):
    if not getattr(config, "MARKET_FLOW_HARD_VETO_ENABLED", False):
        return False, ""

    signal = str(candidate.get("signal") or "").upper()
    flow = (candidate.get("market_context") or {}).get("flow") or {}

    if not flow.get("available"):
        return False, ""

    side_key = signal.lower()
    score = _safe_float(flow.get(f"{side_key}_score"))
    conflicts = int(flow.get(f"{side_key}_conflicts", 0) or 0)
    score_limit = _safe_float(
        getattr(config, "MARKET_FLOW_HARD_VETO_SCORE", -4),
        -4,
    )
    conflict_limit = max(
        int(getattr(config, "MARKET_FLOW_HARD_VETO_MIN_CONFLICTS", 3)),
        1,
    )
    blocked = score <= score_limit and conflicts >= conflict_limit
    reason = (
        f"PERSISTENT_MARKET_FLOW_CONFLICT SCORE={score} "
        f"CONFLICTS={conflicts}/{conflict_limit}"
        if blocked
        else ""
    )
    return blocked, reason


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

    if signal_type in ("REVERSAL", "RANGE_REVERSION"):
        return signal_type

    return "TREND"


def get_position_pool_counts(trade_state, open_positions):
    pools = {
        "TREND": _empty_position_counts(),
        "REVERSAL": _empty_position_counts(),
        "RANGE_REVERSION": _empty_position_counts(),
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
        "RANGE_REVERSION": _empty_position_counts(),
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
    range_reversion_signal = signal_type == "RANGE_REVERSION"

    if reversal_signal:
        pool = "REVERSAL"
    elif range_reversion_signal:
        pool = "RANGE_REVERSION"
    else:
        pool = "TREND"

    counts = pool_counts.get(pool, _empty_position_counts())

    if reversal_signal:
        total_limit = getattr(config, "REVERSAL_EXTRA_TOTAL_POSITIONS", 0)
        buy_limit = getattr(config, "REVERSAL_EXTRA_BUY_POSITIONS", 0)
        sell_limit = getattr(config, "REVERSAL_EXTRA_SELL_POSITIONS", 0)
    elif range_reversion_signal:
        total_limit = getattr(config, "RANGE_REVERSION_TOTAL_POSITIONS", 0)
        buy_limit = getattr(config, "RANGE_REVERSION_BUY_POSITIONS", 0)
        sell_limit = getattr(config, "RANGE_REVERSION_SELL_POSITIONS", 0)
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

        if symbol in entry_quarantined_symbols:
            log_error(
                f"{symbol} entry blocked | stale protection cleanup "
                "is not confirmed"
            )
            return position_details, open_positions, False

        if get_pending_execution(trade_state, symbol):
            log_warning(
                f"{symbol} entry skipped | unsettled execution is still pending"
            )
            return position_details, open_positions, False

        if state_requires_urgent_safety_retry(trade_state):
            log_warning(
                f"{symbol} entry skipped | urgent position-safety work "
                "must reconcile first"
            )
            return position_details, open_positions, False

        flow_blocked, flow_reason = market_flow_hard_veto(candidate)

        if flow_blocked:
            log_warning(f"{symbol} ENTRY BLOCKED | {flow_reason}")
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
                action="SKIPPED_MARKET_FLOW",
                skip_reason=flow_reason,
                news_context=news_context,
                llm_context=llm_context,
                market_context=candidate.get("market_context"),
            )
            return position_details, open_positions, False

        latest_position_details = get_open_position_details(force=True)

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
                llm_context=llm_context,
                market_context=candidate.get("market_context"),
                rank_score=candidate.get("rank_score"),
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
                require_both_override=True if require_both_live else None,
                confidence=side_analysis.get(
                    "confidence",
                    final_analysis.get("best_confidence", 0)
                )
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
                    llm_context=llm_context,
                    market_context=candidate.get("market_context"),
                    rank_score=candidate.get("rank_score"),
                    guard_context=guard_info,
                )
                return position_details, open_positions, False

            log_info(
                f"{symbol} LIVE ENTRY GUARD OK | "
                f"MARK={current_price} | {guard_info.get('reason')}"
            )

        min_room_override = None

        if side_analysis.get("confirmation_type") == "REVERSAL":
            min_room_override = config.REVERSAL_MIN_TP_ROOM_ROI
        elif side_analysis.get("confirmation_type") == "RANGE_REVERSION":
            min_room_override = config.RANGE_REVERSION_MIN_TP_ROOM_ROI

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
                llm_context=llm_context,
                market_context=candidate.get("market_context"),
                rank_score=candidate.get("rank_score"),
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
                news_context=news_context,
                market_context=candidate.get("market_context"),
                rank_score=candidate.get("rank_score"),
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
                llm_context=llm_context,
                market_context=candidate.get("market_context"),
                rank_score=candidate.get("rank_score"),
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

        side = SIDE_BUY if signal == "BUY" else SIDE_SELL
        hard_stop_price = get_entry_hard_stop(
            symbol,
            side,
            current_price,
            confirm_df,
            signal_type,
        )

        if (
            getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False) and
            hard_stop_price is None
        ):
            log_warning(
                f"{symbol} SKIPPED | mandatory hard-stop plan unavailable"
            )
            return position_details, open_positions, False

        balance = get_balance()
        risk_equity = get_conservative_risk_equity(balance)
        campaign_risk_budget = get_position_risk_budget(risk_equity)
        entry_stop_distance_roi = (
            get_stop_buffer_roi(side, current_price, hard_stop_price)
            if hard_stop_price is not None
            else 0
        )
        recovery_required_stop_roi = (
            float(config.DCA_TRIGGER_ROIS[0]) +
            max(float(getattr(config, "DCA_MIN_HARD_STOP_BUFFER_ROI", 0)), 0)
            if config.DCA_ENABLED and config.DCA_TRIGGER_ROIS
            else 0
        )
        recovery_planned = bool(
            config.DCA_ENABLED and
            getattr(config, "DCA_FIXED_RISK_ENABLED", False) and
            entry_stop_distance_roi >= recovery_required_stop_roi
        )
        recovery_disabled_reason = (
            "HARD_STOP_TOO_CLOSE_FOR_RECOVERY"
            if config.DCA_ENABLED and not recovery_planned
            else ""
        )

        if config.DCA_ENABLED and not recovery_planned:
            log_info(
                f"{symbol} recovery disabled for this trade | actual stop "
                f"distance {entry_stop_distance_roi}% < required "
                f"{recovery_required_stop_roi}% | initial entry receives the "
                "full campaign budget"
            )

        initial_risk_pct = (
            max(float(getattr(config, "DCA_INITIAL_RISK_PCT", 70)), 0)
            if recovery_planned
            else 100
        )
        initial_risk_budget = campaign_risk_budget * min(
            initial_risk_pct,
            100,
        ) / 100
        initial_margin = (
            get_initial_trade_margin()
            if recovery_planned
            else config.MARGIN_PER_TRADE
        )
        quantity = calculate_position_size(
            balance,
            current_price,
            hard_stop_price,
            symbol,
            initial_margin,
            risk_budget_override=initial_risk_budget,
        )
        notional = quantity * current_price
        log_info(
            f"{symbol} QTY={quantity} | NOTIONAL={notional:.2f} | "
            f"HARD_STOP={hard_stop_price} | "
            f"CAMPAIGN_RISK={round(campaign_risk_budget, 4)} | "
            f"INITIAL_RISK={round(initial_risk_budget, 4)}"
        )

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

        pre_order_details = get_open_position_details(symbol, force=True)

        if pre_order_details is None:
            log_warning(
                f"{symbol} entry skipped | pre-submit position refresh unavailable"
            )
            return position_details, open_positions, False

        if symbol in pre_order_details:
            log_warning(
                f"{symbol} entry skipped | position appeared before submission"
            )
            return position_details, open_positions, False

        requested_quantity = quantity
        order = submit_entry_order_with_marker(
            trade_state,
            symbol,
            side,
            requested_quantity,
            current_price,
            hard_stop_price,
            signal_type,
        )

        if not order:
            return position_details, open_positions, False

        reconciliation = get_execution_reconciliation(order)

        if not is_reconciled_execution_settled(order):
            log_error(
                f"{symbol} ENTRY EXECUTION UNSETTLED | "
                "no duplicate fallback will be submitted"
            )
            persisted = retain_entry_close_retry(
                trade_state,
                symbol,
                order,
                side,
                requested_quantity,
                current_price,
                signal_type,
                hard_stop_price,
                "ENTRY",
            )

            if persisted:
                # Reconcile immediately rather than leaving a potentially
                # filled entry without its planned hard stop until the next
                # full scan.
                reconcile_pending_executions(trade_state)

            return position_details, open_positions, False

        quantity = get_reconciled_executed_quantity(order)

        if quantity <= 0:
            log_warning(f"{symbol} entry aborted | no executed quantity confirmed")
            return position_details, open_positions, False

        entry_price = get_entry_price(symbol, order)

        if entry_price <= 0:
            entry_price = current_price
            log_warning(
                f"{symbol} ENTRY PRICE UNAVAILABLE | USING CURRENT PRICE FOR TP"
            )

        initial_margin = (
            quantity * entry_price / max(float(config.LEVERAGE), 1)
        )
        actual_initial_risk = (
            get_campaign_risk_at_stop(entry_price, quantity, hard_stop_price)
            if hard_stop_price is not None
            else 0
        )
        hard_stop_valid_for_fill = bool(
            hard_stop_price is None or
            (
                hard_stop_price < entry_price
                if side == SIDE_BUY
                else hard_stop_price > entry_price
            )
        )

        if not hard_stop_valid_for_fill:
            log_error(
                f"{symbol} entry fill crossed planned hard stop; flattening"
            )
            closed = fail_safe_close_unprotected_position(
                symbol,
                reference_price=entry_price,
                context="ENTRY_STOP_CROSSED",
            )

            if not closed:
                retain_entry_close_retry(
                    trade_state,
                    symbol,
                    order,
                    side,
                    quantity,
                    entry_price,
                    signal_type,
                    hard_stop_price,
                    "ENTRY_STOP_CROSSED",
                )

            return position_details, open_positions, False

        # The executable stop distance can change between signal evaluation and
        # the reconciled fill.  Never retain a recovery reservation when the
        # actual entry leaves less than the configured trigger-plus-buffer.
        # A trade that was sized with the full initial budget cannot be upgraded
        # to a recovery campaign after the fill, but a planned recovery may be
        # safely downgraded.
        filled_stop_distance_roi = (
            get_stop_buffer_roi(side, entry_price, hard_stop_price)
            if hard_stop_price is not None
            else 0
        )

        if (
            recovery_planned and
            filled_stop_distance_roi < recovery_required_stop_roi
        ):
            recovery_planned = False
            recovery_disabled_reason = "FILL_STOP_TOO_CLOSE_FOR_RECOVERY"
            log_warning(
                f"{symbol} recovery disabled after fill | actual stop "
                f"distance {filled_stop_distance_roi}% < required "
                f"{recovery_required_stop_roi}%"
            )

        entry_stop_distance_roi = filled_stop_distance_roi

        signal_id = register_signal_outcome(candidate, entry_price)

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
            enable_multi_tp=bool(getattr(config, "MULTI_TP_ENABLED", False)),
            sl_price_override=hard_stop_price,
            return_details=True
        )
        protection_ok = bool(protection_result.get("ok"))
        protection_degraded = False
        protection_degraded_reason = ""

        risk_tolerance = 1 + max(
            float(
                getattr(config, "POSITION_RISK_OVERRUN_TOLERANCE_PCT", 2)
            ),
            0,
        ) / 100
        entry_risk_overrun = bool(
            getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False) and
            initial_risk_budget > 0 and
            actual_initial_risk > initial_risk_budget * risk_tolerance
        )

        if entry_risk_overrun:
            log_error(
                f"{symbol} entry fill exceeded initial risk allocation | "
                f"ACTUAL={round(actual_initial_risk, 4)} > "
                f"BUDGET={round(initial_risk_budget, 4)}"
            )
            closed = fail_safe_close_unprotected_position(
                symbol,
                reference_price=entry_price,
                context="ENTRY_RISK_OVERRUN",
            )

            if not closed:
                retain_entry_close_retry(
                    trade_state,
                    symbol,
                    order,
                    side,
                    quantity,
                    entry_price,
                    signal_type,
                    hard_stop_price,
                    "ENTRY_RISK_OVERRUN",
                )
                send_telegram_message(
                    f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                    f"{symbol} entry risk overrun close is retrying"
                )

            return position_details, open_positions, False

        if not protection_ok:
            log_warning(f"{symbol} TP ORDER NOT CREATED")

            if getattr(config, "PROTECTION_FAILURE_CLOSE_ENABLED", True):
                closed = fail_safe_close_unprotected_position(
                    symbol,
                    reference_price=entry_price,
                    context="ENTRY_PROTECTION_FAILURE",
                )

                if closed:
                    log_error(
                        f"{symbol} entry closed | protection creation failed"
                    )
                    send_telegram_message(
                        f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                        f"{symbol} entry closed: protection creation failed"
                    )
                    latest_position_details = get_open_position_details(
                        force=True
                    )

                    if latest_position_details is not None:
                        position_details = latest_position_details
                        open_positions = get_open_position_amounts(
                            latest_position_details
                        )
                        dca_monitor.sync(latest_position_details)

                    return position_details, open_positions, False

                log_error(
                    f"{symbol} protection failed and emergency close "
                    "was not confirmed"
                )
                retain_entry_close_retry(
                    trade_state,
                    symbol,
                    order,
                    side,
                    quantity,
                    entry_price,
                    signal_type,
                    hard_stop_price,
                    "ENTRY_PROTECTION_FAILURE",
                )
                send_telegram_message(
                    f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                    f"{symbol} protection failure close is retrying"
                )
                return position_details, open_positions, False

            protection_degraded = True
            protection_degraded_reason = (
                protection_degraded_reason or
                "ENTRY_PROTECTION_CLOSE_UNCONFIRMED"
            )
            shutdown_event.set()

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
        position_state["signal_id"] = signal_id
        position_state["execution_status"] = "SETTLED"
        position_state["execution_client_order_ids"] = reconciliation.get(
            "client_order_ids",
            "",
        )
        position_state["execution"] = reconciliation
        position_state["signal_rank_score"] = candidate.get("rank_score")
        position_state["market_context"] = candidate.get("market_context") or {}
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
        position_state["campaign_risk_version"] = 2
        position_state["campaign_equity_snapshot"] = risk_equity
        position_state["campaign_wallet_balance_snapshot"] = balance
        position_state["campaign_risk_budget_usdt"] = round(
            campaign_risk_budget,
            8,
        )
        position_state["campaign_initial_risk_budget_usdt"] = round(
            initial_risk_budget,
            8,
        )
        position_state["campaign_initial_risk_usdt"] = round(
            actual_initial_risk,
            8,
        )
        position_state["campaign_stop_distance_roi"] = entry_stop_distance_roi
        position_state["dca_recovery_planned"] = recovery_planned
        position_state["campaign_stop_price"] = hard_stop_price
        position_state["hard_stop_price"] = hard_stop_price
        position_state["hard_stop_order_id"] = extract_order_id(
            protection_result.get("sl_order")
        )
        position_state["hard_stop_source"] = "ENTRY"
        position_state["dca_recovery_status"] = "WAITING_FOR_TRIGGER"
        position_state["dca_recovery_disabled"] = bool(
            not recovery_planned or protection_degraded
        )
        position_state["dca_recovery_disabled_reason"] = (
            protection_degraded_reason
            if protection_degraded
            else recovery_disabled_reason
        )
        position_state["time_exit_status"] = ""
        position_state["position_management_status"] = (
            "DEGRADED_CLOSE_UNCONFIRMED"
            if protection_degraded
            else "ACTIVE"
        )
        apply_multi_tp_protection_state(position_state, protection_result)
        position_state["tp_updated_at"] = datetime.now().isoformat(
            timespec="seconds"
        )
        state_saved = upsert_position_state(
            trade_state,
            symbol,
            position_state,
        )

        if not state_saved:
            log_error(
                f"{symbol} entry state persistence failed | "
                "attempting fail-safe close"
            )
            closed = fail_safe_close_unprotected_position(
                symbol,
                reference_price=entry_price,
                context="ENTRY_STATE_FAILURE",
            )

            if not closed:
                retain_entry_close_retry(
                    trade_state,
                    symbol,
                    order,
                    side,
                    quantity,
                    entry_price,
                    signal_type,
                    hard_stop_price,
                    "ENTRY_STATE_FAILURE",
                )

            send_telegram_message(
                f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                f"{symbol} state persistence failed\n"
                f"Fail-safe close confirmed: {bool(closed)}"
            )
            latest_position_details = get_open_position_details(force=True)

            if latest_position_details is not None:
                position_details = latest_position_details
                open_positions = get_open_position_amounts(
                    latest_position_details
                )
                dca_monitor.sync(latest_position_details)

            return position_details, open_positions, False

        if protection_degraded:
            send_telegram_message(
                f"{config.TELEGRAM_MESSAGE_PREFIX}\n"
                f"{symbol} protection degraded and close unconfirmed\n"
                "Bot stopped; inspect the live position and exchange orders"
            )
            return position_details, open_positions, False

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
            llm_context=llm_context,
            market_context=candidate.get("market_context"),
            signal_id=signal_id,
            rank_score=candidate.get("rank_score"),
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
        latest_position_details = get_open_position_details(force=True)

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

        if interrupted_dca_submission(
            get_position_state(trade_state, symbol)
        ):
            entry_quarantined_symbols.add(symbol)
            log_error(
                f"{symbol} entry submit marker remains unresolved | "
                "new entries are paused for fail-safe reconciliation"
            )

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

        if state_requires_urgent_safety_retry(trade_state):
            log_warning(
                "Signal ranking paused | urgent position-safety work "
                "requires reconciliation"
            )
            break

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

    try:
        record_volume_profile_telemetry(
            symbol,
            config.CONFIRMATION_TIMEFRAME,
            confirm_df,
            structure=detect_market_structure(confirm_df),
        )
    except Exception as e:
        log_warning(f"{symbol} volume profile telemetry error: {e}")

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
    install_shutdown_signal_handlers()
    sync_client_time()

    if (
        getattr(config, "REQUIRE_ONE_WAY_POSITION_MODE", True) and
        not is_one_way_position_mode()
    ):
        log_error(
            "BOT STARTUP BLOCKED | Binance one-way position mode was not "
            "confirmed. V7 state is symbol-keyed and cannot safely manage "
            "simultaneous hedge legs."
        )
        return

    if not validate_execution_telemetry_path():
        log_warning(
            "Execution telemetry is degraded; reconciliation remains enabled"
        )

    scan_symbols = get_scan_symbols()
    log_info(
        f"Scanning {len(scan_symbols)} symbols | "
        f"KLINE_LIMIT={config.KLINE_LIMIT} | "
        f"THROTTLE={config.REQUEST_THROTTLE_SECONDS}s"
    )
    log_active_dca_config()

    if not validate_position_management_config():
        log_error("BOT STARTUP BLOCKED | unsafe position-management configuration")
        return

    log_info(
        "TP1 EXTRA SLOT CONFIG | "
        f"ENABLED={getattr(config, 'TP1_EXTRA_SLOTS_ENABLED', False)} | "
        f"TOTAL_CAP={getattr(config, 'TP1_EXTRA_TOTAL_POSITIONS', 0)} | "
        f"BUY_CAP={getattr(config, 'TP1_EXTRA_BUY_POSITIONS', 0)} | "
        f"SELL_CAP={getattr(config, 'TP1_EXTRA_SELL_POSITIONS', 0)}"
    )
    dca_monitor = None
    flow_monitor = None
    shadow_flow_monitor = None
    target_margin_monitor = None

    try:
        dca_monitor = DcaWebsocketMonitor()
        dca_monitor.start()

        try:
            shadow_flow_monitor = OrderFlowShadowMonitor(
                scan_symbols,
                snapshot_provider=get_futures_depth_snapshot,
                shutdown_event=shutdown_event,
            )
            shadow_flow_monitor.start()
        except Exception as exc:
            log_warning(f"Shadow order-flow startup warning: {exc}")
            if shadow_flow_monitor is not None:
                try:
                    shadow_flow_monitor.stop()
                except Exception as cleanup_exc:
                    log_warning(
                        "Shadow order-flow partial-start cleanup warning: "
                        f"{cleanup_exc}"
                    )
            shadow_flow_monitor = None

        flow_monitor = MarketFlowMonitor(
            scan_symbols,
            shutdown_event=shutdown_event,
            shadow_monitor=shadow_flow_monitor,
        )
        flow_monitor.start()
        target_margin_monitor = TargetMarginBalanceMonitor()
        target_margin_monitor.start()

        while not shutdown_event.is_set():
            try:
                position_details = get_open_position_details()

                if position_details is None:
                    log_warning("Position snapshot unavailable; skipping this scan")
                    pending_retry = False

                    try:
                        pending_retry = state_requires_urgent_safety_retry(
                            load_trade_state()
                        )
                    except Exception:
                        pending_retry = False

                    wait_for_next_scan(
                        "PENDING_EXECUTION_POSITION_SNAPSHOT_UNAVAILABLE"
                        if pending_retry
                        else "POSITION_SNAPSHOT_UNAVAILABLE",
                        getattr(
                            config,
                            "PENDING_EXECUTION_RECONCILE_SECONDS",
                            5,
                        )
                        if pending_retry
                        else None,
                    )
                    continue

                open_positions = get_open_position_amounts(position_details)
                try:
                    trade_state = load_runtime_trade_state(open_positions)
                except TradeStateLoadError as exc:
                    log_error(f"BOT STOPPING | TRADE STATE UNSAFE: {exc}")
                    shutdown_event.set()
                    break

                if trade_state.get("pending_executions"):
                    reconcile_pending_executions(
                        trade_state,
                        position_details=position_details,
                    )
                    refreshed_details = get_open_position_details(force=True)

                    if refreshed_details is None:
                        log_warning(
                            "Pending execution reconciliation completed but "
                            "position refresh is unavailable; skipping scan"
                        )
                        wait_for_next_scan(
                            "PENDING_EXECUTION_REFRESH_UNAVAILABLE",
                            getattr(
                                config,
                                "PENDING_EXECUTION_RECONCILE_SECONDS",
                                5,
                            ),
                        )
                        continue

                    position_details = refreshed_details
                    open_positions = get_open_position_amounts(position_details)

                    if trade_state.get("pending_executions"):
                        log_warning(
                            "Pending execution remains unresolved; strategy "
                            "scan is paused until its topology is safe"
                        )
                        wait_for_next_scan(
                            "PENDING_EXECUTION_RETRY",
                            getattr(
                                config,
                                "PENDING_EXECUTION_RECONCILE_SECONDS",
                                5,
                            ),
                        )
                        continue

                untracked_attempted, untracked_unresolved = (
                    reconcile_untracked_open_positions(
                        position_details,
                        trade_state,
                    )
                )

                if untracked_attempted:
                    refreshed_details = get_open_position_details(force=True)

                    if refreshed_details is None or untracked_unresolved:
                        wait_for_next_scan(
                            "UNTRACKED_POSITION_FAIL_CLOSE_RETRY",
                            getattr(
                                config,
                                "PENDING_EXECUTION_RECONCILE_SECONDS",
                                5,
                            ),
                        )
                        continue

                    position_details = refreshed_details
                    open_positions = get_open_position_amounts(position_details)

                interrupted_attempted, interrupted_unresolved = (
                    reconcile_interrupted_dca_submissions(
                        position_details,
                        trade_state,
                    )
                )

                if interrupted_attempted:
                    refreshed_details = get_open_position_details(force=True)

                    if refreshed_details is None or interrupted_unresolved:
                        wait_for_next_scan(
                            "INTERRUPTED_DCA_FAIL_CLOSE_RETRY",
                            getattr(
                                config,
                                "PENDING_EXECUTION_RECONCILE_SECONDS",
                                5,
                            ),
                        )
                        continue

                    position_details = refreshed_details
                    open_positions = get_open_position_amounts(position_details)

                uncovered_symbols = {
                    symbol
                    for symbol in position_details
                    if symbol in configured_entry_symbol_scope() and
                    not get_position_state(trade_state, symbol) and
                    not get_pending_execution(trade_state, symbol)
                }

                if uncovered_symbols:
                    log_error(
                        "Live positions remain without durable ownership: "
                        f"{','.join(sorted(uncovered_symbols))}"
                    )
                    wait_for_next_scan(
                        "UNTRACKED_POSITION_RECHECK",
                        getattr(
                            config,
                            "PENDING_EXECUTION_RECONCILE_SECONDS",
                            5,
                        ),
                    )
                    continue

                prune_and_cleanup_closed_positions(trade_state, open_positions)
                log_closed_trades(open_positions)
                dca_monitor.sync(position_details)
                dca_monitor.reconcile_multi_tp_positions(
                    position_details,
                    trade_state,
                )
                btc_trend_df, btc_trend = get_cached_btc_context()
                log_info(f"BTC TREND: {btc_trend}")

                for open_symbol, position_detail in position_details.items():
                    try:
                        ensure_route_stop_loss(
                            open_symbol,
                            position_detail,
                            trade_state,
                            btc_trend_df,
                        )
                    except Exception as e:
                        log_error(
                            f"{open_symbol} hard-stop reconcile error: {e}"
                        )

                tp_reprice_blocked_symbols = set()

                for open_symbol, position_detail in position_details.items():
                    try:
                        if repair_pending_dca_tp_reprice(
                            open_symbol,
                            position_detail,
                            trade_state,
                            btc_trend_df,
                        ):
                            tp_reprice_blocked_symbols.add(open_symbol)
                    except Exception as e:
                        tp_reprice_blocked_symbols.add(open_symbol)
                        log_error(
                            f"{open_symbol} DCA TP reprice repair error: {e}"
                        )

                lifecycle_blocked_symbols = (
                    dca_monitor.reconcile_position_management(
                        position_details,
                        trade_state,
                    )
                )
                lifecycle_blocked_symbols.update(tp_reprice_blocked_symbols)

                futures_context_queue = []
                signal_candidates = []
                breadth_samples = []
                begin_llm_scan_budget()

                for symbol in scan_symbols:
                    if shutdown_event.is_set():
                        log_warning("Scan stopped | bot shutdown requested")
                        break

                    try:
                        log_info(f"Checking {symbol}")

                        if symbol in open_positions:
                            open_mark_price = (
                                position_details.get(symbol, {}).get("mark_price")
                            )
                            observe_signal_outcomes(
                                symbol,
                                open_mark_price,
                                open_mark_price,
                                open_mark_price,
                            )
                            if symbol not in lifecycle_blocked_symbols:
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

                        breadth_sample = build_breadth_sample(symbol, trend_df)

                        if breadth_sample:
                            breadth_samples.append(breadth_sample)

                        latest_entry_candle = entry_df.iloc[-2]
                        observe_signal_outcomes(
                            symbol,
                            latest_entry_candle.get("close"),
                            latest_entry_candle.get("high"),
                            latest_entry_candle.get("low"),
                        )

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

                breadth_context = calculate_market_breadth(breadth_samples)
                log_info(
                    "MARKET BREADTH | "
                    f"AVAILABLE={breadth_context.get('available')} | "
                    f"SAMPLES={breadth_context.get('sample_count')} | "
                    f"ABOVE_EMA20={breadth_context.get('above_ema20_pct')}% | "
                    f"ABOVE_EMA50={breadth_context.get('above_ema50_pct')}% | "
                    f"ADVANCE={breadth_context.get('advance_pct')}% | "
                    f"BUY_SCORE={breadth_context.get('buy_score')}"
                )

                for candidate in signal_candidates:
                    enrich_candidate_market_context(
                        candidate,
                        flow_monitor,
                        breadth_context,
                        shadow_flow_monitor,
                    )
                    attach_candidate_shadow_order_flow(
                        candidate,
                        shadow_flow_monitor,
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

                wait_for_next_scan(
                    "URGENT_POSITION_SAFETY_RETRY"
                    if state_requires_urgent_safety_retry(trade_state)
                    else "SCAN_COMPLETE",
                    getattr(
                        config,
                        "PENDING_EXECUTION_RECONCILE_SECONDS",
                        5,
                    )
                    if state_requires_urgent_safety_retry(trade_state)
                    else None,
                )

            except Exception as e:
                log_error(f"MAIN LOOP ERROR: {e}")
                wait_for_next_scan("MAIN_LOOP_ERROR")

    finally:
        for monitor, label in (
            (target_margin_monitor, "target margin monitor"),
            (flow_monitor, "market-flow monitor"),
            (shadow_flow_monitor, "shadow order-flow monitor"),
            (dca_monitor, "DCA monitor"),
        ):
            if monitor is None:
                continue

            try:
                monitor.stop()
            except Exception as exc:
                log_warning(f"{label} shutdown warning: {exc}")

        flush_execution_telemetry()

    log_warning("BOT STOPPED | manual restart required")

if __name__ == "__main__":
    run_bot()
