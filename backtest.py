import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

import config
from indicators import apply_indicators
from multi_tp import calculate_runner_stop
from strategy import (
    analyze_signal,
    evaluate_route_early_invalidation,
    evaluate_route_profit_protection,
    evaluate_time_exit_weakness,
    validate_adverse_zone_level,
    validate_dca_continuation_guard,
    validate_dca_recovery_confirmation,
    validate_entry_profit_room,
    validate_structure_take_profit,
)


BINANCE_FAPI_KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
BAN_UNTIL_RE = re.compile(r"banned until\s+(\d+)", re.IGNORECASE)
RATE_LIMIT_RE = re.compile(r"(code=-1003|too many requests)", re.IGNORECASE)
KLINE_COLUMNS = [
    "time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "trades",
    "tbbav",
    "tbqav",
    "ignore",
]
INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}


@dataclass
class BacktestData:
    raw: pd.DataFrame
    indicators: pd.DataFrame


def utc_now_ms():
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def parse_time_ms(value, default=None):
    if not value:
        return default if default is not None else utc_now_ms()

    value = str(value).strip()

    if value.isdigit():
        number = int(value)
        return number if number > 10_000_000_000 else number * 1000

    if len(value) == 10:
        value = f"{value}T00:00:00+00:00"
    elif value.endswith("Z"):
        value = value.replace("Z", "+00:00")

    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp() * 1000)


def ms_to_iso(value):
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc).isoformat()


def interval_to_ms(interval):
    if interval not in INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")

    return INTERVAL_MS[interval]


def parse_symbols(value):
    if not value:
        symbols = config.BACKTEST_SYMBOLS or config.SYMBOLS
    elif isinstance(value, str):
        symbols = [item.strip() for item in value.split(",") if item.strip()]
    else:
        symbols = list(value)

    result = []

    for symbol in symbols:
        symbol = str(symbol).upper().strip()

        if symbol and symbol not in result:
            result.append(symbol)

    max_symbols = int(getattr(config, "BACKTEST_MAX_SYMBOLS", 10))

    if max_symbols > 0:
        result = result[:max_symbols]

    return result


def parse_confirmation_types(value):
    if not value:
        values = getattr(config, "BACKTEST_CONFIRMATION_TYPES", [])
    elif isinstance(value, str):
        values = [item.strip() for item in value.split(",") if item.strip()]
    else:
        values = list(value)

    result = []

    for item in values:
        confirmation_type = str(item).upper().strip()

        if confirmation_type and confirmation_type not in result:
            result.append(confirmation_type)

    return result


def _reversal_reason_key(reason):
    text = str(reason or "UNKNOWN").strip()

    if not text:
        return "UNKNOWN"

    return text.split(" ", 1)[0].split("=", 1)[0]


def record_reversal_diagnostics(diagnostics, analysis):
    if diagnostics is None:
        return

    diagnostics.setdefault("evaluations", 0)
    diagnostics.setdefault("chart_confirmed", 0)
    diagnostics.setdefault("final_signals", 0)
    diagnostics.setdefault("near_misses", 0)
    diagnostics.setdefault("max_confidence", 0.0)
    reason_counts = diagnostics.setdefault("rejection_reasons", {})
    combination_counts = diagnostics.setdefault("rejection_combinations", {})

    for side in ("buy", "sell"):
        side_data = analysis.get(side, {}) or {}
        diagnostics["evaluations"] += 1
        confidence = float(side_data.get("reversal_confidence", 0) or 0)
        diagnostics["max_confidence"] = max(
            float(diagnostics["max_confidence"]),
            confidence,
        )

        if side_data.get("reversal_confirmed"):
            diagnostics["chart_confirmed"] += 1
            continue

        reasons = {
            _reversal_reason_key(reason)
            for reason in side_data.get("reversal_reasons", [])
        }

        if reasons:
            combination = "+".join(sorted(reasons))
            combination_counts[combination] = (
                combination_counts.get(combination, 0) + 1
            )

        if confidence >= 80 and len(reasons) <= 2:
            diagnostics["near_misses"] += 1

        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1


def reversal_diagnostics_summary(diagnostics, top_reasons=10):
    diagnostics = diagnostics or {}
    reason_counts = diagnostics.get("rejection_reasons", {}) or {}
    ordered_reasons = sorted(
        reason_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    combination_counts = diagnostics.get("rejection_combinations", {}) or {}
    ordered_combinations = sorted(
        combination_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    pre_entry_counts = diagnostics.get("pre_entry_rejections", {}) or {}
    ordered_pre_entry = sorted(
        pre_entry_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )
    return {
        "evaluations": int(diagnostics.get("evaluations", 0)),
        "chart_confirmed": int(diagnostics.get("chart_confirmed", 0)),
        "final_signals": int(diagnostics.get("final_signals", 0)),
        "near_misses": int(diagnostics.get("near_misses", 0)),
        "max_confidence": round(
            float(diagnostics.get("max_confidence", 0) or 0),
            2,
        ),
        "top_rejection_reasons": [
            {"reason": reason, "count": int(count)}
            for reason, count in ordered_reasons[:max(int(top_reasons), 0)]
        ],
        "top_rejection_combinations": [
            {"reasons": reasons, "count": int(count)}
            for reasons, count in ordered_combinations[:max(int(top_reasons), 0)]
        ],
        "pre_entry_rejections": [
            {"reason": reason, "count": int(count)}
            for reason, count in ordered_pre_entry[:max(int(top_reasons), 0)]
        ],
    }


def data_path(data_dir, symbol, interval):
    return Path(data_dir) / f"{symbol}_{interval}.csv"


def normalise_kline_frame(df, interval):
    df = df.copy()

    if "open_time" in df.columns and "time" not in df.columns:
        df.rename(columns={"open_time": "time"}, inplace=True)

    if "time" not in df.columns:
        raise ValueError("Historical data must contain a time column")

    for column in ("time", "close_time"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")

    if "close_time" not in df.columns:
        df["close_time"] = df["time"].astype("int64") + interval_to_ms(interval) - 1

    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.dropna(subset=["time", "open", "high", "low", "close", "volume"], inplace=True)
    df["time"] = df["time"].astype("int64")
    df["close_time"] = df["close_time"].astype("int64")
    df.drop_duplicates(subset=["time"], keep="last", inplace=True)
    df.sort_values("time", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_klines_csv(path, interval):
    df = pd.read_csv(path)
    return normalise_kline_frame(df, interval)


def save_klines_csv(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["time", "open", "high", "low", "close", "volume", "close_time"]
    df[columns].to_csv(path, index=False)


def extract_rate_limit_backoff_seconds(message):
    buffer_seconds = max(
        float(getattr(config, "PUBLIC_REST_BACKOFF_BUFFER_SECONDS", 60)),
        0.0,
    )
    match = BAN_UNTIL_RE.search(str(message))

    if match:
        try:
            banned_until_ms = int(match.group(1))
            banned_until_seconds = banned_until_ms / 1000
            return max(
                banned_until_seconds - time.time() + buffer_seconds,
                buffer_seconds,
            )
        except (TypeError, ValueError):
            pass

    if RATE_LIMIT_RE.search(str(message)):
        return max(
            float(getattr(config, "PUBLIC_REST_DEFAULT_BACKOFF_SECONDS", 300)),
            1.0,
        )

    return 0.0


def download_klines(symbol, interval, start_ms, end_ms, sleep_seconds):
    rows = []
    cursor = start_ms
    interval_ms = interval_to_ms(interval)

    while cursor < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        }
        response = requests.get(BINANCE_FAPI_KLINES_URL, params=params, timeout=20)

        if response.status_code != 200:
            backoff_seconds = extract_rate_limit_backoff_seconds(response.text)

            if backoff_seconds > 0:
                print(
                    f"{symbol} {interval} Binance rate limit backoff | "
                    f"sleep={round(backoff_seconds, 1)}s",
                    flush=True,
                )
                time.sleep(backoff_seconds)
                continue

            raise RuntimeError(
                f"{symbol} {interval} download failed: "
                f"{response.status_code} {response.text[:200]}"
            )

        batch = response.json()

        if not batch:
            break

        rows.extend(batch)
        last_open_time = int(batch[-1][0])
        next_cursor = last_open_time + interval_ms

        if next_cursor <= cursor:
            break

        cursor = next_cursor

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if not rows:
        return pd.DataFrame(columns=KLINE_COLUMNS)

    df = pd.DataFrame(rows, columns=KLINE_COLUMNS)
    return normalise_kline_frame(df, interval)


def load_or_download(symbol, interval, start_ms, end_ms, args):
    path = data_path(args.data_dir, symbol, interval)

    if path.exists() and not args.force_download:
        return load_klines_csv(path, interval)

    if not args.download:
        raise FileNotFoundError(
            f"Missing {path}. Re-run with --download or set BACKTEST_DOWNLOAD_DATA=True."
        )

    warmup_start = start_ms - (
        interval_to_ms(interval) * int(getattr(config, "BACKTEST_WARMUP_CANDLES", 260))
    )
    warmup_start = max(warmup_start, 0)
    df = download_klines(
        symbol,
        interval,
        warmup_start,
        end_ms,
        float(getattr(config, "BACKTEST_DOWNLOAD_SLEEP_SECONDS", 0.2)),
    )
    save_klines_csv(path, df)
    return df


def prepare_data(raw_df, interval):
    raw_df = normalise_kline_frame(raw_df, interval)
    indicator_df = apply_indicators(raw_df)

    if indicator_df is None or indicator_df.empty:
        return BacktestData(raw=raw_df, indicators=pd.DataFrame())

    indicator_df = normalise_kline_frame(indicator_df, interval)
    return BacktestData(raw=raw_df, indicators=indicator_df)


def closed_slice(indicator_df, decision_ms, interval, min_rows=220):
    if indicator_df is None or indicator_df.empty:
        return None

    close_times = indicator_df["close_time"].to_numpy()
    end_pos = close_times.searchsorted(decision_ms, side="left")

    if end_pos < min_rows:
        return None

    max_rows = int(getattr(config, "BACKTEST_SLICE_MAX_ROWS", 0) or 0)

    if max_rows > 0:
        start_pos = max(0, end_pos - max(max_rows, min_rows))
    else:
        start_pos = 0

    closed = indicator_df.iloc[start_pos:end_pos]
    forming = closed.iloc[[-1]].copy()
    forming["time"] = int(decision_ms)
    forming["close_time"] = int(decision_ms + interval_to_ms(interval) - 1)
    return pd.concat([closed, forming], ignore_index=True, copy=False)


def btc_trend_from_slice(btc_df):
    if btc_df is None or len(btc_df) < 2:
        return "NEUTRAL"

    latest = btc_df.iloc[-2]

    if latest["ema50"] > latest["ema200"]:
        return "BULLISH"

    if latest["ema50"] < latest["ema200"]:
        return "BEARISH"

    return "NEUTRAL"


def calculate_btc_context(symbol, trend_df, btc_df):
    if symbol == "BTCUSDT":
        return 1.0, 0.0

    if trend_df is None or btc_df is None:
        return 0.0, 0.0

    try:
        coin_close = trend_df.iloc[:-1]["close"].tail(100).reset_index(drop=True)
        btc_close = btc_df.iloc[:-1]["close"].tail(100).reset_index(drop=True)
        length = min(len(coin_close), len(btc_close))

        if length < 20:
            btc_corr = 0.0
        else:
            coin_ret = coin_close.tail(length).pct_change().dropna()
            btc_ret = btc_close.tail(length).pct_change().dropna()
            btc_corr = coin_ret.corr(btc_ret)

            if btc_corr != btc_corr:
                btc_corr = 0.0

        if length < 10:
            rs = 0.0
        else:
            coin_tail = coin_close.tail(length)
            btc_tail = btc_close.tail(length)
            coin_r = ((coin_tail.iloc[-1] - coin_tail.iloc[-10]) / coin_tail.iloc[-10]) * 100
            btc_r = ((btc_tail.iloc[-1] - btc_tail.iloc[-10]) / btc_tail.iloc[-10]) * 100
            rs = coin_r - btc_r

        return round(float(btc_corr), 2), round(float(rs), 2)

    except Exception:
        return 0.0, 0.0


def roi_to_price(side, entry_price, roi):
    move = (float(roi) / max(float(config.LEVERAGE), 1)) / 100

    if side == "BUY":
        return entry_price * (1 + move)

    return entry_price * (1 - move)


def roi_stop_loss_price(side, entry_price, roi):
    move = (float(roi) / max(float(config.LEVERAGE), 1)) / 100

    if side == "BUY":
        return entry_price * (1 - move)

    return entry_price * (1 + move)


def stop_loss_enabled(confirmation_type):
    confirmation_type = str(confirmation_type or "").upper()

    if confirmation_type == "REVERSAL":
        return bool(getattr(config, "REVERSAL_SL_ENABLED", config.SL_ENABLED))

    if confirmation_type == "TREND":
        return bool(getattr(config, "TREND_SL_ENABLED", config.SL_ENABLED))

    return bool(getattr(config, "SL_ENABLED", False))


def max_sl_roi(confirmation_type):
    confirmation_type = str(confirmation_type or "").upper()

    if confirmation_type == "REVERSAL":
        return float(getattr(config, "REVERSAL_MAX_SL_ROI", config.MAX_SL_ROI))

    return float(getattr(config, "TREND_MAX_SL_ROI", config.MAX_SL_ROI))


def structure_stop_loss_price(side, confirm_df):
    try:
        if confirm_df is None or len(confirm_df) < 12:
            return None

        atr = float(confirm_df["atr"].iloc[-1])

        if atr <= 0:
            return None

        if side == "BUY":
            return float(confirm_df["low"].iloc[-10:-1].min()) - (atr * 0.5)

        return float(confirm_df["high"].iloc[-10:-1].max()) + (atr * 0.5)

    except Exception:
        return None


def compute_stop_loss(side, avg_entry, confirm_df, confirmation_type):
    if not stop_loss_enabled(confirmation_type):
        return None, "SL_DISABLED"

    structure_sl = structure_stop_loss_price(side, confirm_df)
    cap_roi = max_sl_roi(confirmation_type)
    capped_sl = roi_stop_loss_price(side, avg_entry, cap_roi) if cap_roi > 0 else None

    if structure_sl is None:
        return capped_sl, f"ROI_SL_{cap_roi}%"

    if capped_sl is None:
        return structure_sl, "STRUCTURE_SL"

    if side == "BUY":
        if structure_sl >= avg_entry:
            return capped_sl, f"ROI_SL_{cap_roi}%"

        return max(structure_sl, capped_sl), f"STRUCTURE_SL_CAPPED_{cap_roi}%"

    if structure_sl <= avg_entry:
        return capped_sl, f"ROI_SL_{cap_roi}%"

    return min(structure_sl, capped_sl), f"STRUCTURE_SL_CAPPED_{cap_roi}%"


def candle_hits_sl(side, candle, sl_price):
    if sl_price is None:
        return False

    if side == "BUY":
        return float(candle["low"]) <= sl_price

    return float(candle["high"]) >= sl_price


def apply_entry_slippage(side, price):
    slip = float(getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.02)) / 100

    if side == "BUY":
        return price * (1 + slip)

    return price * (1 - slip)


def apply_exit_slippage(side, price):
    slip = float(getattr(config, "BACKTEST_SLIPPAGE_PCT", 0.02)) / 100

    if side == "BUY":
        return price * (1 - slip)

    return price * (1 + slip)


def margin_per_trade():
    override = float(getattr(config, "BACKTEST_MARGIN_PER_TRADE", 0))
    return override if override > 0 else float(config.MARGIN_PER_TRADE)


def initial_margin():
    base = margin_per_trade()

    if getattr(config, "BACKTEST_USE_DCA", False) and config.DCA_ENABLED:
        return base * max(float(config.DCA_INITIAL_MARGIN_PCT), 0) / 100

    return base


def dca_margin(dca_index):
    if not getattr(config, "BACKTEST_USE_DCA", False) or not config.DCA_ENABLED:
        return 0.0

    if dca_index >= config.DCA_MAX_ORDERS:
        return 0.0

    if dca_index >= len(config.DCA_MARGIN_PCTS):
        return 0.0

    return margin_per_trade() * max(float(config.DCA_MARGIN_PCTS[dca_index]), 0) / 100


def dca_trigger_roi(dca_index):
    if dca_index >= config.DCA_MAX_ORDERS:
        return None

    if dca_index >= len(config.DCA_TRIGGER_ROIS):
        return None

    return float(config.DCA_TRIGGER_ROIS[dca_index])


def position_average_entry(fills):
    qty = sum(item["qty"] for item in fills)

    if qty <= 0:
        return 0.0

    return sum(item["qty"] * item["price"] for item in fills) / qty


def calculate_trade_pnl(side, fills, exit_price, partial_exits=None):
    fee_rate = float(getattr(config, "BACKTEST_FEE_RATE", 0.0004))
    gross = 0.0
    entry_fees = 0.0
    exit_fees = 0.0

    for fill in fills:
        qty = fill["qty"]

        if side == "BUY":
            gross += qty * (exit_price - fill["price"])
        else:
            gross += qty * (fill["price"] - exit_price)

        entry_fees += qty * fill["price"] * fee_rate
        exit_fees += qty * exit_price * fee_rate

    for partial in partial_exits or []:
        qty = float(partial["qty"])
        entry = float(partial["entry_price"])
        partial_exit = float(partial["exit_price"])

        if side == "BUY":
            gross += qty * (partial_exit - entry)
        else:
            gross += qty * (entry - partial_exit)

        entry_fees += qty * entry * fee_rate
        exit_fees += qty * partial_exit * fee_rate

    fees = entry_fees + exit_fees
    margin = sum(item["margin"] for item in fills)
    net = gross - fees
    roi = (net / margin * 100) if margin > 0 else 0.0
    return gross, fees, net, roi


def realize_partial_exit(fills, close_pct, exit_price):
    fraction = min(max(float(close_pct) / 100, 0), 1)
    partial_exits = []

    if fraction <= 0 or fraction >= 1:
        return partial_exits

    for fill in fills:
        quantity = float(fill["qty"])
        close_quantity = quantity * fraction
        fill["qty"] = quantity - close_quantity
        partial_exits.append({
            "qty": close_quantity,
            "entry_price": float(fill["price"]),
            "exit_price": float(exit_price),
        })

    return partial_exits


def compute_take_profit(
    side,
    avg_entry,
    trend_df,
    confirm_df,
    confirmation_type=None,
    dca_context=False,
):
    reversal = str(confirmation_type or "").upper() == "REVERSAL"
    reversal_max_roi = max(
        float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
        0,
    )

    if dca_context and config.DCA_TP_MODE in ("roi", "fixed_roi", "fallback_roi"):
        roi = float(config.DCA_TP_ROI)

        if reversal and reversal_max_roi > 0:
            roi = min(roi, reversal_max_roi)

        return roi_to_price(side, avg_entry, roi), f"DCA_ROI_{roi}%"

    if config.STATIC_TP_ENABLED:
        roi = float(config.STATIC_TP_ROI)

        if reversal and reversal_max_roi > 0:
            roi = min(roi, reversal_max_roi)

        return roi_to_price(side, avg_entry, roi), f"STATIC_ROI_{roi}%"

    ok, target = validate_structure_take_profit(
        side,
        avg_entry,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE,
    )

    if ok and target.get("target_price"):
        target_roi = float(target.get("target_roi") or 0)

        if reversal and reversal_max_roi > 0 and target_roi > reversal_max_roi:
            return (
                roi_to_price(side, avg_entry, reversal_max_roi),
                f"REVERSAL_STRUCTURE_CAPPED_{reversal_max_roi}%",
            )

        return float(target["target_price"]), f"STRUCTURE_{target['source']}"

    roi = float(
        getattr(config, "REVERSAL_TP_FALLBACK_ROI", 35)
        if reversal
        else config.STRUCTURE_TP_FALLBACK_ROI
    )

    if reversal and reversal_max_roi > 0:
        roi = min(roi, reversal_max_roi)

    return roi_to_price(side, avg_entry, roi), f"FALLBACK_ROI_{roi}%"


def compute_runner_take_profit(
    side,
    basis_price,
    trend_df,
    confirm_df,
    confirmation_type=None,
):
    ok, target = validate_structure_take_profit(
        side,
        basis_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE,
    )
    reversal = str(confirmation_type or "").upper() == "REVERSAL"
    reversal_max_roi = max(
        float(getattr(config, "REVERSAL_TP_MAX_ROI", 45)),
        0,
    )

    if ok and target.get("target_price"):
        target_roi = float(target.get("target_roi") or 0)

        if reversal and reversal_max_roi > 0 and target_roi > reversal_max_roi:
            return (
                roi_to_price(side, basis_price, reversal_max_roi),
                f"TP2_REVERSAL_STRUCTURE_CAPPED_{reversal_max_roi}%",
            )

        return float(target["target_price"]), f"TP2_STRUCTURE_{target['source']}"

    fallback_roi = max(float(getattr(config, "TP2_FALLBACK_ROI", 35)), 0)

    if reversal and reversal_max_roi > 0:
        fallback_roi = min(fallback_roi, reversal_max_roi)

    return (
        roi_to_price(side, basis_price, fallback_roi),
        f"TP2_FALLBACK_ROI_{fallback_roi}%",
    )


def candle_hits_tp(side, candle, tp_price):
    if side == "BUY":
        return float(candle["high"]) >= tp_price

    return float(candle["low"]) <= tp_price


def candle_hits_dca(side, candle, trigger_price):
    if side == "BUY":
        return float(candle["low"]) <= trigger_price

    return float(candle["high"]) >= trigger_price


def dca_trigger_price(side, anchor_price, roi):
    move = (float(roi) / max(float(config.LEVERAGE), 1)) / 100

    if side == "BUY":
        return anchor_price * (1 - move)

    return anchor_price * (1 + move)


def simulate_trade(
    symbol,
    side,
    confirmation_type,
    confidence,
    entry_time,
    entry_price,
    entry_index,
    frames,
):
    entry_price = apply_entry_slippage(side, float(entry_price))
    init_margin = initial_margin()
    fills = [{
        "time": int(entry_time),
        "price": entry_price,
        "margin": init_margin,
        "qty": (init_margin * float(config.LEVERAGE)) / entry_price,
    }]
    dca_count = 0
    max_hold = int(getattr(config, "BACKTEST_MAX_HOLD_CANDLES", 240))
    signal_entry_df = frames["entry"].indicators
    fast_frame = frames.get("fast") or frames["entry"]
    slow_frame = frames.get("slow") or frames["confirm"]
    fast_signal_df = fast_frame.indicators
    slow_signal_df = slow_frame.indicators
    exit_frame = frames.get("exit") or frames["entry"]
    exit_df = exit_frame.indicators
    trend_interval = config.TREND_TIMEFRAME
    confirm_interval = config.CONFIRMATION_TIMEFRAME
    entry_interval = config.ENTRY_TIMEFRAME
    fast_interval = config.LIVE_ENTRY_FAST_TIMEFRAME
    slow_interval = config.LIVE_ENTRY_SLOW_TIMEFRAME
    decision_trend = closed_slice(
        frames["trend"].indicators,
        int(entry_time),
        trend_interval,
    )
    decision_confirm = closed_slice(
        frames["confirm"].indicators,
        int(entry_time),
        confirm_interval,
    )
    avg_entry = position_average_entry(fills)
    tp_price, tp_mode = compute_take_profit(
        side,
        avg_entry,
        decision_trend,
        decision_confirm,
        confirmation_type=confirmation_type,
    )
    tp1_close_pct = float(getattr(config, "TP1_CLOSE_POSITION_PCT", 50))
    multi_tp_enabled = bool(
        getattr(config, "BACKTEST_MULTI_TP_ENABLED", False) and
        getattr(config, "MULTI_TP_ENABLED", False) and
        0 < tp1_close_pct < 100
    )
    tp1_price = float(tp_price)
    tp1_mode = tp_mode
    tp1_hit = False
    tp1_exit_price = None
    tp1_time = None
    tp2_price = None
    tp2_mode = ""
    partial_exits = []
    sl_price, sl_mode = compute_stop_loss(
        side,
        avg_entry,
        decision_confirm,
        confirmation_type,
    )
    entry_stop_distance_roi = (
        (
            ((entry_price - sl_price) / entry_price) *
            float(config.LEVERAGE) * 100
            if side == "BUY"
            else ((sl_price - entry_price) / entry_price) *
            float(config.LEVERAGE) * 100
        )
        if sl_price is not None and entry_price > 0
        else 0
    )
    recovery_required_stop_roi = (
        float(config.DCA_TRIGGER_ROIS[0]) +
        max(float(getattr(config, "DCA_MIN_HARD_STOP_BUFFER_ROI", 0)), 0)
        if (
            getattr(config, "BACKTEST_USE_DCA", False) and
            config.DCA_ENABLED and
            config.DCA_TRIGGER_ROIS
        )
        else 0
    )
    recovery_planned = bool(
        getattr(config, "BACKTEST_USE_DCA", False) and
        config.DCA_ENABLED and
        getattr(config, "DCA_FIXED_RISK_ENABLED", False) and
        entry_stop_distance_roi >= recovery_required_stop_roi
    )

    if (
        getattr(config, "BACKTEST_USE_DCA", False) and
        config.DCA_ENABLED and
        getattr(config, "DCA_FIXED_RISK_ENABLED", False) and
        not recovery_planned
    ):
        fills[0]["margin"] = float(config.MARGIN_PER_TRADE)
        fills[0]["qty"] = (
            fills[0]["margin"] * float(config.LEVERAGE) / entry_price
        )

    campaign_risk_budget = 0.0

    if (
        getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False) and
        sl_price is not None
    ):
        campaign_risk_budget = (
            float(getattr(config, "BACKTEST_INITIAL_BALANCE", 1000)) *
            max(float(getattr(config, "POSITION_RISK_PCT", 0)), 0) /
            100
        )
        max_risk = max(
            float(getattr(config, "POSITION_RISK_MAX_USDT", 0)),
            0,
        )

        if max_risk > 0:
            campaign_risk_budget = min(campaign_risk_budget, max_risk)

        initial_risk_budget = campaign_risk_budget * (
            max(float(getattr(config, "DCA_INITIAL_RISK_PCT", 70)), 0) /
            100
            if recovery_planned
            else 1
        )
        risk_quantity = initial_risk_budget / max(
            abs(entry_price - sl_price),
            1e-12,
        )
        fills[0]["qty"] = min(fills[0]["qty"], risk_quantity)
        fills[0]["margin"] = (
            fills[0]["qty"] * entry_price / max(float(config.LEVERAGE), 1)
        )
        avg_entry = position_average_entry(fills)

    max_seen_adverse_roi = 0.0
    max_seen_favorable_roi = 0.0
    profit_protection_peak_roi = 0.0
    profit_protection_floor_roi = 0.0
    exit_candle = None
    exit_reason = "OPEN_AT_DATA_END"
    exit_price = None
    exit_time_override = None
    dca_blocked_count = 0
    last_dca_block_reason = ""
    early_invalidation_reason = ""
    time_exit_reason = ""
    recovery_armed = False
    recovery_disabled = False
    recovery_extreme_price = None
    recovery_armed_time = None

    exit_times = exit_df["time"].to_numpy()
    start_index = exit_times.searchsorted(int(entry_time), side="left")

    if start_index >= len(exit_df):
        start_index = max(len(exit_df) - 1, 0)

    max_hold_ms = max_hold * interval_to_ms(config.ENTRY_TIMEFRAME)
    max_exit_time = int(entry_time) + max_hold_ms
    end_index = exit_times.searchsorted(max_exit_time, side="right")
    end_index = min(max(end_index, start_index + 1), len(exit_df))

    for row_index in range(start_index, end_index):
        candle = exit_df.iloc[row_index]
        candle_time = int(candle["time"])
        candle_close_decision_ms = int(candle["close_time"]) + 1
        dca_filled_this_candle = False

        # Conservative intrabar ordering: an exchange hard stop always wins
        # before any software exit or recovery add. Gap-through exits use the
        # adverse candle open rather than the configured trigger price.
        if candle_hits_sl(side, candle, sl_price):
            candle_open = float(candle["open"])
            stop_fill = (
                min(candle_open, sl_price)
                if side == "BUY"
                else max(candle_open, sl_price)
            )
            exit_price = apply_exit_slippage(side, stop_fill)
            exit_candle = candle
            exit_reason = "RUNNER_SL" if tp1_hit else "SL"
            break

        if (
            not tp1_hit and
            getattr(config, "EARLY_FLOW_EXIT_ENABLED", False)
        ):
            route = (
                "REVERSAL"
                if str(confirmation_type or "").upper() == "REVERSAL"
                else "TREND"
            )
            route_enabled = (
                getattr(config, "EARLY_FLOW_EXIT_REVERSAL_ENABLED", True)
                if route == "REVERSAL"
                else getattr(config, "EARLY_FLOW_EXIT_TREND_ENABLED", True)
            )
            last_activity_time = int(fills[-1]["time"])
            grace_minutes = float(
                getattr(
                    config,
                    "EARLY_FLOW_EXIT_POST_DCA_GRACE_MINUTES",
                    config.EARLY_FLOW_EXIT_MINUTES,
                )
                if dca_count > 0
                else config.EARLY_FLOW_EXIT_MINUTES
            )
            elapsed_ms = candle_time - last_activity_time
            route_max_roi = min(
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
            current_price = float(candle["open"])
            current_roi = (
                ((current_price - avg_entry) / avg_entry) *
                float(config.LEVERAGE) * 100
                if side == "BUY"
                else ((avg_entry - current_price) / avg_entry) *
                float(config.LEVERAGE) * 100
            )

            if (
                route_enabled and
                elapsed_ms >= max(grace_minutes, 0) * 60_000 and
                current_roi <= route_max_roi
            ):
                fast_slice = closed_slice(
                    fast_signal_df,
                    candle_time,
                    fast_interval,
                    min_rows=40,
                )
                slow_slice = closed_slice(
                    slow_signal_df,
                    candle_time,
                    slow_interval,
                    min_rows=40,
                )
                early_info = evaluate_route_early_invalidation(
                    side,
                    fast_slice,
                    slow_slice,
                    current_price,
                    confirmation_type=route,
                    reference_price=entry_price,
                )

                if early_info.get("should_exit"):
                    exit_price = apply_exit_slippage(side, current_price)
                    exit_candle = candle
                    exit_time_override = candle_time
                    exit_reason = f"{route}_EARLY_INVALIDATION"
                    early_invalidation_reason = early_info.get("reason", "")
                    max_seen_adverse_roi = max(
                        max_seen_adverse_roi,
                        abs(float(current_roi)),
                    )
                    break

        if not tp1_hit and getattr(config, "TIME_EXIT_ENABLED", False):
            route = (
                "REVERSAL"
                if str(confirmation_type or "").upper() == "REVERSAL"
                else "TREND"
            )
            route_enabled = bool(
                getattr(config, f"TIME_EXIT_{route}_ENABLED", route == "TREND")
            )
            elapsed_minutes = (candle_time - int(entry_time)) / 60_000
            post_dca_grace_minutes = max(
                float(getattr(config, "TIME_EXIT_POST_DCA_GRACE_MINUTES", 0)),
                0,
            )
            post_dca_grace_complete = bool(
                dca_count <= 0 or
                post_dca_grace_minutes <= 0 or
                candle_time - int(fills[-1]["time"]) >=
                post_dca_grace_minutes * 60_000
            )
            current_price = float(candle["open"])
            current_roi = (
                ((current_price - avg_entry) / avg_entry) *
                float(config.LEVERAGE) * 100
                if side == "BUY"
                else ((avg_entry - current_price) / avg_entry) *
                float(config.LEVERAGE) * 100
            )

            if (
                route_enabled and
                elapsed_minutes >= max(float(config.TIME_EXIT_MINUTES), 0) and
                post_dca_grace_complete and
                current_roi <= min(float(config.TIME_EXIT_MAX_ROI), 0)
            ):
                trend_slice = closed_slice(
                    frames["trend"].indicators,
                    candle_time,
                    trend_interval,
                )
                confirm_slice = closed_slice(
                    frames["confirm"].indicators,
                    candle_time,
                    confirm_interval,
                )
                weakness = evaluate_time_exit_weakness(
                    side,
                    trend_slice,
                    confirm_slice,
                )
                should_exit = (
                    weakness.get("should_exit")
                    if getattr(config, "TIME_EXIT_REQUIRE_WEAKNESS", True)
                    else True
                )

                if should_exit:
                    exit_price = apply_exit_slippage(side, current_price)
                    exit_candle = candle
                    exit_time_override = candle_time
                    exit_reason = f"{route}_TIME_EXIT"
                    time_exit_reason = weakness.get("reason", "")
                    break

        while not tp1_hit:
            if recovery_disabled:
                break

            trigger_roi = dca_trigger_roi(dca_count)
            next_margin = dca_margin(dca_count)

            if trigger_roi is None or next_margin <= 0:
                break

            cooldown_ms = max(
                int(getattr(config, "DCA_MIN_SECONDS_BETWEEN_ORDERS", 0)),
                0,
            ) * 1000

            if cooldown_ms and candle_time - int(fills[-1]["time"]) < cooldown_ms:
                break

            trigger_anchor_price = fills[0]["price"]
            spacing_anchor_price = fills[-1]["price"]
            max_adverse_roi = max(
                float(getattr(config, "DCA_MAX_ADVERSE_ROI", 0)),
                0,
            )

            adverse_extreme_price = (
                float(candle["low"])
                if side == "BUY"
                else float(candle["high"])
            )
            actual_adverse_roi = abs(
                ((trigger_anchor_price - adverse_extreme_price) /
                 trigger_anchor_price) * float(config.LEVERAGE) * 100
                if side == "BUY"
                else ((adverse_extreme_price - trigger_anchor_price) /
                      trigger_anchor_price) * float(config.LEVERAGE) * 100
            )

            if max_adverse_roi and actual_adverse_roi > max_adverse_roi:
                dca_blocked_count += 1
                last_dca_block_reason = "DCA_MAX_RISK_EXCEEDED"
                recovery_armed = False
                recovery_disabled = True
                break

            trigger_price = dca_trigger_price(
                side,
                trigger_anchor_price,
                trigger_roi,
            )

            trigger_hit = candle_hits_dca(side, candle, trigger_price)
            recovery_mode = bool(
                getattr(config, "DCA_RECOVERY_CONFIRMATION_ENABLED", False)
            )

            if recovery_mode:
                adverse_price = (
                    float(candle["low"])
                    if side == "BUY"
                    else float(candle["high"])
                )

                if not recovery_armed:
                    if not trigger_hit:
                        break

                    recovery_armed = True
                    recovery_extreme_price = adverse_price
                    recovery_armed_time = candle_time
                    last_dca_block_reason = "DCA_RECOVERY_ARMED"
                    break

                arm_timeout_ms = max(
                    float(
                        getattr(
                            config,
                            "DCA_RECOVERY_ARM_TIMEOUT_MINUTES",
                            240,
                        )
                    ),
                    0,
                ) * 60_000

                if (
                    arm_timeout_ms and
                    recovery_armed_time is not None and
                    candle_time - recovery_armed_time > arm_timeout_ms
                ):
                    recovery_armed = False
                    recovery_disabled = True
                    dca_blocked_count += 1
                    last_dca_block_reason = "DCA_RECOVERY_ARM_TIMEOUT"
                    break

                if side == "BUY":
                    recovery_extreme_price = min(
                        float(recovery_extreme_price),
                        adverse_price,
                    )
                    rebound_move = float(candle["close"]) - recovery_extreme_price
                else:
                    recovery_extreme_price = max(
                        float(recovery_extreme_price),
                        adverse_price,
                    )
                    rebound_move = recovery_extreme_price - float(candle["close"])

                rebound_roi = (
                    max(rebound_move, 0) /
                    max(float(recovery_extreme_price), 1e-12) *
                    float(config.LEVERAGE) * 100
                )

                if rebound_roi < max(
                    float(getattr(config, "DCA_RECOVERY_MIN_REBOUND_ROI", 5)),
                    0,
                ):
                    last_dca_block_reason = "DCA_RECOVERY_REBOUND_INCOMPLETE"
                    break

                candidate_fill_price = float(candle["close"])

                recovery_price_gap_roi = abs(
                    ((trigger_anchor_price - candidate_fill_price) /
                     trigger_anchor_price) * float(config.LEVERAGE) * 100
                    if side == "BUY"
                    else ((candidate_fill_price - trigger_anchor_price) /
                          trigger_anchor_price) * float(config.LEVERAGE) * 100
                )
                minimum_price_gap_roi = max(
                    float(getattr(config, "DCA_MIN_PRICE_GAP_ROI", 0)),
                    0,
                )

                if recovery_price_gap_roi < minimum_price_gap_roi:
                    dca_blocked_count += 1
                    last_dca_block_reason = "DCA_RECOVERY_PRICE_GAP_TOO_SMALL"
                    break

                fast_slice = closed_slice(
                    fast_signal_df,
                    candle_close_decision_ms,
                    fast_interval,
                    min_rows=40,
                )
                slow_slice = closed_slice(
                    slow_signal_df,
                    candle_close_decision_ms,
                    slow_interval,
                    min_rows=40,
                )
                recovery_ok, recovery_info = validate_dca_recovery_confirmation(
                    side,
                    fast_slice,
                    slow_slice,
                    candidate_fill_price,
                )

                if not recovery_ok:
                    dca_blocked_count += 1
                    last_dca_block_reason = recovery_info.get(
                        "reason",
                        "DCA_RECOVERY_NOT_CONFIRMED",
                    )
                    break
            else:
                if not trigger_hit:
                    break

                candidate_fill_price = trigger_price

            if sl_price is not None:
                stop_buffer_roi = (
                    ((candidate_fill_price - sl_price) / candidate_fill_price) *
                    float(config.LEVERAGE) * 100
                    if side == "BUY"
                    else ((sl_price - candidate_fill_price) / candidate_fill_price) *
                    float(config.LEVERAGE) * 100
                )

                if stop_buffer_roi < max(
                    float(getattr(config, "DCA_MIN_HARD_STOP_BUFFER_ROI", 0)),
                    0,
                ):
                    dca_blocked_count += 1
                    last_dca_block_reason = "DCA_HARD_STOP_BUFFER_TOO_SMALL"
                    break

            if (
                getattr(config, "DCA_STRICT_GUARD_ENABLED", True) or
                getattr(config, "DCA_TRIGGER_MODE", "static_roi") ==
                "adaptive_hybrid"
            ):
                trend_slice = closed_slice(
                    frames["trend"].indicators,
                    candle_close_decision_ms,
                    trend_interval,
                )
                confirm_slice = closed_slice(
                    frames["confirm"].indicators,
                    candle_close_decision_ms,
                    confirm_interval,
                )
                entry_slice = closed_slice(
                    signal_entry_df,
                    candle_close_decision_ms,
                    entry_interval,
                )
                position_adverse_roi = abs(
                    ((avg_entry - candidate_fill_price) / avg_entry) *
                    float(config.LEVERAGE) * 100
                    if side == "BUY"
                    else ((candidate_fill_price - avg_entry) / avg_entry) *
                    float(config.LEVERAGE) * 100
                )
                candidate_adverse_roi = abs(
                    ((trigger_anchor_price - candidate_fill_price) /
                     trigger_anchor_price) * float(config.LEVERAGE) * 100
                    if side == "BUY"
                    else ((candidate_fill_price - trigger_anchor_price) /
                          trigger_anchor_price) * float(config.LEVERAGE) * 100
                )
                guard_ok, guard_info = validate_dca_continuation_guard(
                    side,
                    candidate_fill_price,
                    avg_entry,
                    trend_slice,
                    confirm_slice,
                    entry_slice,
                    leverage=config.LEVERAGE,
                    confirmation_type=confirmation_type,
                    dca_level=dca_count + 1,
                    adverse_roi=round(float(candidate_adverse_roi), 2),
                    position_adverse_roi=round(float(position_adverse_roi), 2),
                    trigger_roi=trigger_roi,
                    spacing_anchor_price=spacing_anchor_price,
                )

                if not guard_ok:
                    dca_blocked_count += 1
                    last_dca_block_reason = guard_info.get(
                        "reason",
                        "DCA_STRICT_GUARD_BLOCKED"
                    )
                    break

            fill_price = apply_entry_slippage(side, candidate_fill_price)
            fill_quantity = (next_margin * float(config.LEVERAGE)) / fill_price

            if campaign_risk_budget > 0 and sl_price is not None:
                existing_risk = sum(
                    item["qty"] * abs(item["price"] - sl_price)
                    for item in fills
                )
                recovery_cap = campaign_risk_budget * max(
                    float(getattr(config, "DCA_RECOVERY_RISK_PCT", 30)),
                    0,
                ) / 100
                remaining_risk = min(
                    max(campaign_risk_budget - existing_risk, 0),
                    recovery_cap,
                )
                fill_quantity = min(
                    fill_quantity,
                    remaining_risk / max(abs(fill_price - sl_price), 1e-12),
                )

            if fill_quantity <= 0:
                dca_blocked_count += 1
                last_dca_block_reason = "DCA_FIXED_RISK_EXHAUSTED"
                break

            fills.append({
                "time": candle_close_decision_ms,
                "price": fill_price,
                "margin": (
                    fill_quantity * fill_price / max(float(config.LEVERAGE), 1)
                ),
                "qty": fill_quantity,
            })
            dca_count += 1
            dca_filled_this_candle = True
            recovery_armed = False
            avg_entry = position_average_entry(fills)
            profit_protection_peak_roi = 0.0
            profit_protection_floor_roi = 0.0

            if getattr(config, "DCA_REPRICE_TP_AFTER_FILL", True):
                trend_slice = closed_slice(
                    frames["trend"].indicators,
                    candle_close_decision_ms,
                    trend_interval,
                )
                confirm_slice = closed_slice(
                    frames["confirm"].indicators,
                    candle_close_decision_ms,
                    confirm_interval,
                )

                if trend_slice is not None and confirm_slice is not None:
                    tp_price, tp_mode = compute_take_profit(
                        side,
                        avg_entry,
                        trend_slice,
                        confirm_slice,
                        confirmation_type=confirmation_type,
                        dca_context=True,
                    )
                    tp1_price = float(tp_price)
                    tp1_mode = tp_mode
                    # The campaign hard stop is immutable. DCA may reprice the
                    # target from the new average, but it must never loosen or
                    # replace the original exchange risk boundary.

            if (
                getattr(config, "DCA_TRIGGER_MODE", "static_roi") ==
                "adaptive_hybrid"
            ):
                break

        if dca_filled_this_candle:
            # The recovery order is modeled at this candle's close. Its new
            # quantity cannot profit from high/low extrema that occurred
            # earlier in the same candle.
            continue

        adverse_price = float(candle["low"]) if side == "BUY" else float(candle["high"])
        adverse_roi = abs(
            ((avg_entry - adverse_price) / avg_entry) * float(config.LEVERAGE) * 100
            if side == "BUY"
            else ((adverse_price - avg_entry) / avg_entry) * float(config.LEVERAGE) * 100
        )
        max_seen_adverse_roi = max(max_seen_adverse_roi, adverse_roi)
        favorable_price = (
            float(candle["high"])
            if side == "BUY"
            else float(candle["low"])
        )
        favorable_roi = (
            ((favorable_price - avg_entry) / avg_entry) *
            float(config.LEVERAGE) *
            100
            if side == "BUY"
            else ((avg_entry - favorable_price) / avg_entry) *
            float(config.LEVERAGE) *
            100
        )
        max_seen_favorable_roi = max(
            max_seen_favorable_roi,
            favorable_roi,
            0,
        )
        profit_protection_peak_roi = max(
            profit_protection_peak_roi,
            favorable_roi,
            0,
        )
        route = (
            "REVERSAL"
            if str(confirmation_type or "").upper() == "REVERSAL"
            else "TREND"
        )
        if tp1_hit:
            profit_info = {"armed": False}
        else:
            profit_info = evaluate_route_profit_protection(
                side,
                avg_entry,
                favorable_price,
                peak_roi=profit_protection_peak_roi,
                leverage=config.LEVERAGE,
                confirmation_type=route,
            )
            profit_protection_peak_roi = max(
                profit_protection_peak_roi,
                float(profit_info.get("peak_roi", 0) or 0),
            )
            profit_protection_floor_roi = float(
                profit_info.get("floor_roi", 0) or 0
            )

        if not tp1_hit and profit_info.get("armed"):
            profit_floor_price = roi_to_price(
                side,
                avg_entry,
                profit_protection_floor_roi,
            )
            profit_floor_hit = (
                float(candle["low"]) <= profit_floor_price
                if side == "BUY"
                else float(candle["high"]) >= profit_floor_price
            )

            if profit_floor_hit:
                exit_price = apply_exit_slippage(side, profit_floor_price)
                exit_candle = candle
                exit_reason = f"{route}_PROFIT_PROTECTION"
                break

        if candle_hits_tp(side, candle, tp_price):
            if multi_tp_enabled and not tp1_hit:
                tp1_exit_price = apply_exit_slippage(side, tp_price)
                partial_exits.extend(
                    realize_partial_exit(
                        fills,
                        tp1_close_pct,
                        tp1_exit_price,
                    )
                )
                tp1_hit = True
                tp1_time = int(candle["close_time"])
                trend_slice = closed_slice(
                    frames["trend"].indicators,
                    candle_time,
                    trend_interval,
                )
                confirm_slice = closed_slice(
                    frames["confirm"].indicators,
                    candle_time,
                    confirm_interval,
                )
                tp2_price, tp2_mode = compute_runner_take_profit(
                    side,
                    float(tp_price),
                    trend_slice,
                    confirm_slice,
                    confirmation_type=confirmation_type,
                )
                runner_sl, runner_sl_info = calculate_runner_stop(
                    side,
                    avg_entry,
                    float(tp_price),
                    confirm_slice,
                    leverage=config.LEVERAGE,
                )

                if (
                    getattr(config, "TP1_RUNNER_STOP_ENABLED", True) and
                    runner_sl is not None
                ):
                    sl_price = runner_sl
                    sl_mode = f"RUNNER_{runner_sl_info.get('source', 'PROFIT_LOCK')}"

                tp_price = float(tp2_price)
                tp_mode = tp2_mode
                continue

            exit_price = apply_exit_slippage(side, tp_price)
            exit_candle = candle
            exit_reason = "TP2" if tp1_hit else "TP"
            break

    if exit_price is None:
        if exit_candle is None:
            if end_index > start_index:
                exit_candle = exit_df.iloc[end_index - 1]
            else:
                exit_candle = exit_df.iloc[min(start_index, len(exit_df) - 1)]

        exit_price = apply_exit_slippage(side, float(exit_candle["close"]))
        exit_reason = (
            "RUNNER_TIMEOUT"
            if tp1_hit and end_index < len(exit_df)
            else "RUNNER_DATA_END"
            if tp1_hit
            else "TIMEOUT"
            if end_index < len(exit_df)
            else "DATA_END"
        )

    gross, fees, net, roi = calculate_trade_pnl(
        side,
        fills,
        exit_price,
        partial_exits=partial_exits,
    )
    exit_time = int(
        exit_time_override
        if exit_time_override is not None
        else exit_candle["close_time"]
    )
    duration_hours = (exit_time - int(entry_time)) / 3_600_000

    return {
        "symbol": symbol,
        "side": side,
        "confirmation_type": confirmation_type,
        "confidence": round(float(confidence), 2),
        "entry_time": ms_to_iso(entry_time),
        "exit_time": ms_to_iso(exit_time),
        "entry_ms": int(entry_time),
        "exit_ms": exit_time,
        "entry_price": round(entry_price, 8),
        "avg_entry": round(position_average_entry(fills), 8),
        "exit_price": round(float(exit_price), 8),
        "tp_price": round(float(tp_price), 8),
        "tp_mode": tp_mode,
        "tp1_hit": tp1_hit,
        "tp1_price": round(float(tp1_price), 8),
        "tp1_mode": tp1_mode,
        "tp1_exit_price": (
            round(float(tp1_exit_price), 8)
            if tp1_exit_price is not None
            else ""
        ),
        "tp1_time": ms_to_iso(tp1_time) if tp1_time is not None else "",
        "tp1_ms": int(tp1_time) if tp1_time is not None else None,
        "tp1_close_pct": (
            tp1_close_pct
            if tp1_hit
            else 0
        ),
        "tp2_price": round(float(tp2_price), 8) if tp2_price is not None else "",
        "tp2_mode": tp2_mode,
        "sl_price": round(float(sl_price), 8) if sl_price is not None else "",
        "sl_mode": sl_mode,
        "exit_reason": exit_reason,
        "dca_count": dca_count,
        "fill_count": len(fills),
        "margin_used": round(sum(item["margin"] for item in fills), 4),
        "gross_pnl": round(gross, 4),
        "fees": round(fees, 4),
        "net_pnl": round(net, 4),
        "roi_on_margin": round(roi, 2),
        "result": "WIN" if net > 0 else "LOSS",
        "duration_hours": round(duration_hours, 2),
        "max_adverse_roi": round(max_seen_adverse_roi, 2),
        "max_favorable_roi": round(max_seen_favorable_roi, 2),
        "profit_protection_floor_roi": round(
            profit_protection_floor_roi,
            2,
        ),
        "reversal_profit_floor_roi": round(
            profit_protection_floor_roi
            if str(confirmation_type or "").upper() == "REVERSAL"
            else 0,
            2,
        ),
        "trend_profit_floor_roi": round(
            profit_protection_floor_roi
            if str(confirmation_type or "").upper() != "REVERSAL"
            else 0,
            2,
        ),
        "dca_blocked_count": dca_blocked_count,
        "last_dca_block_reason": last_dca_block_reason,
        "early_invalidation_reason": early_invalidation_reason,
        "time_exit_reason": time_exit_reason,
        "campaign_risk_budget_usdt": round(campaign_risk_budget, 4),
        "recovery_armed_at_data_end": bool(recovery_armed),
        "recovery_disabled_at_data_end": bool(recovery_disabled),
        "recovery_planned": bool(recovery_planned),
    }


def passes_pre_entry_filters(signal, current_price, trend_df, confirm_df, side_analysis):
    min_room_override = None

    if side_analysis.get("confirmation_type") == "REVERSAL":
        min_room_override = config.REVERSAL_MIN_TP_ROOM_ROI

    room_ok, room_info = validate_entry_profit_room(
        signal,
        current_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE,
        min_roi_override=min_room_override,
    )

    if not room_ok:
        return False, room_info.get("reason", "PROFIT_ROOM_BLOCKED"), {}

    level_ok, level_info = validate_adverse_zone_level(
        signal,
        current_price,
        trend_df,
        confirm_df,
        leverage=config.LEVERAGE,
    )

    if not level_ok:
        return False, level_info.get("reason", "ADVERSE_LEVEL_BLOCKED"), {}

    return True, "OK", level_info


def generate_symbol_trades(
    symbol,
    frames,
    btc_frames,
    start_ms,
    end_ms,
    allowed_confirmation_types=None,
    reversal_diagnostics=None,
):
    trades = []
    entry_df = frames["entry"].indicators
    trend_df = frames["trend"].indicators
    confirm_df = frames["confirm"].indicators
    btc_trend_df = btc_frames["trend"].indicators if btc_frames else None
    entry_interval = config.ENTRY_TIMEFRAME
    active_until_ms = 0
    rows = entry_df[
        (entry_df["time"] >= start_ms) &
        (entry_df["time"] <= end_ms)
    ]
    step_candles = max(int(getattr(config, "BACKTEST_SIGNAL_STEP_CANDLES", 4)), 1)
    progress_every = max(int(getattr(config, "BACKTEST_PROGRESS_EVERY_CANDLES", 500)), 0)
    scan_rows = rows.iloc[::step_candles] if step_candles > 1 else rows
    total_rows = len(scan_rows)

    print(
        f"{symbol}: replaying {total_rows} decision candles "
        f"(step={step_candles})",
        flush=True,
    )

    for checked_count, row in enumerate(scan_rows.itertuples(), start=1):
        entry_index = int(row.Index)
        decision_ms = int(row.time)

        if progress_every and checked_count % progress_every == 0:
            print(
                f"{symbol}: checked {checked_count}/{total_rows} "
                f"decision candles | trades={len(trades)}",
                flush=True,
            )

        if getattr(config, "BACKTEST_ONE_POSITION_PER_SYMBOL", True):
            if decision_ms <= active_until_ms:
                continue

        trend_slice = closed_slice(trend_df, decision_ms, config.TREND_TIMEFRAME)
        confirm_slice = closed_slice(confirm_df, decision_ms, config.CONFIRMATION_TIMEFRAME)
        entry_slice = closed_slice(entry_df, decision_ms, entry_interval)

        if trend_slice is None or confirm_slice is None or entry_slice is None:
            continue

        btc_slice = None

        if getattr(config, "BACKTEST_USE_BTC_CONTEXT", True) and btc_trend_df is not None:
            btc_slice = closed_slice(btc_trend_df, decision_ms, config.TREND_TIMEFRAME)

        btc_trend = btc_trend_from_slice(btc_slice)
        btc_corr, rs = calculate_btc_context(symbol, trend_slice, btc_slice)
        analysis = analyze_signal(
            trend_slice,
            confirm_slice,
            entry_slice,
            btc_trend,
            btc_corr,
            rs,
            participation=None,
            log_details=False,
        )
        record_reversal_diagnostics(reversal_diagnostics, analysis)
        signal = analysis.get("signal")

        if not signal:
            continue

        side_analysis = analysis.get(signal.lower(), {})
        confirmation_type = str(
            side_analysis.get("confirmation_type", "UNKNOWN")
        ).upper()

        if confirmation_type == "REVERSAL" and reversal_diagnostics is not None:
            reversal_diagnostics["final_signals"] = (
                reversal_diagnostics.get("final_signals", 0) + 1
            )

        if (
            allowed_confirmation_types
            and confirmation_type not in allowed_confirmation_types
        ):
            continue

        current_price = float(row.open)
        filters_ok, reason, _ = passes_pre_entry_filters(
            signal,
            current_price,
            trend_slice,
            confirm_slice,
            side_analysis,
        )

        if not filters_ok:
            if confirmation_type == "REVERSAL" and reversal_diagnostics is not None:
                pre_entry_rejections = reversal_diagnostics.setdefault(
                    "pre_entry_rejections",
                    {},
                )
                reason_key = str(reason or "PRE_ENTRY_FILTER_BLOCKED").strip()
                pre_entry_rejections[reason_key] = (
                    pre_entry_rejections.get(reason_key, 0) + 1
                )
            continue

        trade = simulate_trade(
            symbol,
            signal,
            confirmation_type,
            side_analysis.get("confidence", 0),
            decision_ms,
            current_price,
            int(entry_index),
            frames,
        )
        trade["skip_reason"] = reason
        trades.append(trade)
        active_until_ms = int(trade["exit_ms"])

    return trades


def backtest_position_pool(trade):
    confirmation_type = str(
        trade.get("confirmation_type") or "TREND"
    ).upper()
    return "REVERSAL" if confirmation_type == "REVERSAL" else "TREND"


def backtest_position_limits(pool):
    if pool == "REVERSAL":
        return (
            getattr(config, "REVERSAL_EXTRA_TOTAL_POSITIONS", 0),
            getattr(config, "REVERSAL_EXTRA_BUY_POSITIONS", 0),
            getattr(config, "REVERSAL_EXTRA_SELL_POSITIONS", 0),
        )

    return (
        config.MAX_TOTAL_POSITIONS,
        config.MAX_BUY_POSITIONS,
        config.MAX_SELL_POSITIONS,
    )


def _backtest_limit_with_tp1_capacity(base_limit, runner_count, extra_cap):
    if base_limit is None:
        return None

    earned = min(
        max(int(runner_count or 0), 0),
        max(int(extra_cap or 0), 0),
    )
    return base_limit + earned


def apply_position_limits(trades):
    if not getattr(config, "BACKTEST_APPLY_POSITION_LIMITS", True):
        return trades, 0

    accepted = []
    skipped = 0

    for trade in sorted(trades, key=lambda item: item["entry_ms"]):
        pool = backtest_position_pool(trade)
        max_total, max_buy, max_sell = backtest_position_limits(pool)

        open_trades = [
            item for item in accepted
            if (
                item["entry_ms"] <= trade["entry_ms"] < item["exit_ms"]
                and backtest_position_pool(item) == pool
            )
        ]
        total_count = len(open_trades)
        buy_count = sum(1 for item in open_trades if item["side"] == "BUY")
        sell_count = sum(1 for item in open_trades if item["side"] == "SELL")

        if getattr(config, "TP1_EXTRA_SLOTS_ENABLED", False):
            runners = [
                item for item in open_trades
                if (
                    item.get("tp1_hit") and
                    item.get("tp1_ms") is not None and
                    int(item["tp1_ms"]) <= int(trade["entry_ms"])
                )
            ]
            runner_buy_count = sum(
                1 for item in runners if item["side"] == "BUY"
            )
            runner_sell_count = sum(
                1 for item in runners if item["side"] == "SELL"
            )
            max_total = _backtest_limit_with_tp1_capacity(
                max_total,
                len(runners),
                getattr(config, "TP1_EXTRA_TOTAL_POSITIONS", 0),
            )
            max_buy = _backtest_limit_with_tp1_capacity(
                max_buy,
                runner_buy_count,
                getattr(config, "TP1_EXTRA_BUY_POSITIONS", 0),
            )
            max_sell = _backtest_limit_with_tp1_capacity(
                max_sell,
                runner_sell_count,
                getattr(config, "TP1_EXTRA_SELL_POSITIONS", 0),
            )

        if max_total is not None and total_count >= max_total:
            skipped += 1
            continue

        if trade["side"] == "BUY" and max_buy is not None and buy_count >= max_buy:
            skipped += 1
            continue

        if trade["side"] == "SELL" and max_sell is not None and sell_count >= max_sell:
            skipped += 1
            continue

        accepted.append(trade)

    return sorted(accepted, key=lambda item: item["entry_ms"]), skipped


def summarise_trades(trades, skipped_by_limits):
    balance = float(getattr(config, "BACKTEST_INITIAL_BALANCE", 1000))
    equity = balance
    peak = balance
    max_drawdown = 0.0
    wins = 0
    losses = 0
    gross_profit = 0.0
    gross_loss = 0.0
    dca_trades = 0
    durations = []
    equity_curve = []

    for trade in sorted(trades, key=lambda item: item["exit_ms"]):
        pnl = float(trade["net_pnl"])
        equity += pnl
        peak = max(peak, equity)
        drawdown = ((peak - equity) / peak * 100) if peak > 0 else 0
        max_drawdown = max(max_drawdown, drawdown)
        equity_curve.append({
            "time": trade["exit_time"],
            "equity": round(equity, 4),
            "drawdown_pct": round(drawdown, 2),
        })

        if pnl > 0:
            wins += 1
            gross_profit += pnl
        else:
            losses += 1
            gross_loss += abs(pnl)

        if int(trade["dca_count"]) > 0:
            dca_trades += 1

        durations.append(float(trade["duration_hours"]))

    total = len(trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    net_pnl = equity - balance
    notes = [
        "Backtest uses historical OHLCV candles and simulated fills.",
        "News, LLM, realtime websocket confirmation, order-book flow, and private account state are not replayed.",
        "Intracandle order is conservative: DCA, SL, then TP are evaluated inside the same candle.",
    ]

    if getattr(config, "BACKTEST_MULTI_TP_ENABLED", False):
        notes.append(
            "Multi-stage TP realizes TP1 pro rata, disables later DCA, then "
            "models TP2 and the runner profit-lock stop from the next candle."
        )

    return {
        "initial_balance": round(balance, 4),
        "final_balance": round(equity, 4),
        "net_pnl": round(net_pnl, 4),
        "net_return_pct": round((net_pnl / balance * 100) if balance else 0, 2),
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate_pct": round((wins / total * 100) if total else 0, 2),
        "profit_factor": round(profit_factor, 3) if profit_factor is not None else None,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "max_drawdown_pct": round(max_drawdown, 2),
        "avg_roi_on_margin_pct": round(
            sum(float(item["roi_on_margin"]) for item in trades) / total,
            2,
        ) if total else 0,
        "avg_duration_hours": round(sum(durations) / len(durations), 2) if durations else 0,
        "dca_trades": dca_trades,
        "skipped_by_position_limits": skipped_by_limits,
        "notes": notes,
        "equity_curve": equity_curve,
    }


def summarise_by_symbol(trades):
    rows = []

    for symbol in sorted({item["symbol"] for item in trades}):
        subset = [item for item in trades if item["symbol"] == symbol]
        wins = sum(1 for item in subset if item["result"] == "WIN")
        pnl = sum(float(item["net_pnl"]) for item in subset)
        rows.append({
            "symbol": symbol,
            "trades": len(subset),
            "wins": wins,
            "losses": len(subset) - wins,
            "win_rate_pct": round((wins / len(subset) * 100) if subset else 0, 2),
            "net_pnl": round(pnl, 4),
            "avg_roi_on_margin_pct": round(
                sum(float(item["roi_on_margin"]) for item in subset) / len(subset),
                2,
            ) if subset else 0,
            "dca_trades": sum(1 for item in subset if int(item["dca_count"]) > 0),
        })

    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_path = path

    if not rows:
        try:
            path.write_text("", encoding="utf-8")
        except PermissionError:
            actual_path = timestamped_output_path(path)
            print(f"{path.name} is locked; writing {actual_path.name} instead")
            actual_path.write_text("", encoding="utf-8")
        return actual_path

    try:
        handle = path.open("w", newline="", encoding="utf-8")
    except PermissionError:
        actual_path = timestamped_output_path(path)
        print(f"{path.name} is locked; writing {actual_path.name} instead")
        handle = actual_path.open("w", newline="", encoding="utf-8")

    with handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    return actual_path


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_path = path
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except PermissionError:
        actual_path = timestamped_output_path(path)
        print(f"{path.name} is locked; writing {actual_path.name} instead")
        actual_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return actual_path


def timestamped_output_path(path):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return path.with_name(f"{path.stem}_{stamp}{path.suffix}")


def load_symbol_frames(symbol, args, start_ms, end_ms):
    intervals = {
        "trend": config.TREND_TIMEFRAME,
        "confirm": config.CONFIRMATION_TIMEFRAME,
        "entry": config.ENTRY_TIMEFRAME,
        "exit": getattr(config, "BACKTEST_EXIT_TIMEFRAME", config.ENTRY_TIMEFRAME),
        "fast": config.LIVE_ENTRY_FAST_TIMEFRAME,
        "slow": config.LIVE_ENTRY_SLOW_TIMEFRAME,
    }
    frames = {}
    loaded_by_interval = {}

    for key, interval in intervals.items():
        if interval in loaded_by_interval:
            frames[key] = loaded_by_interval[interval]
            continue

        raw = load_or_download(symbol, interval, start_ms, end_ms, args)
        frames[key] = prepare_data(raw, interval)
        loaded_by_interval[interval] = frames[key]

        if frames[key].indicators.empty:
            raise RuntimeError(f"{symbol} {interval} has no indicator-ready data")

    return frames


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Offline historical backtest for binance_ai_bot_v7."
    )
    parser.add_argument("--symbols", default="")
    parser.add_argument("--start", default=config.BACKTEST_START)
    parser.add_argument("--end", default=config.BACKTEST_END)
    parser.add_argument("--data-dir", default=config.BACKTEST_DATA_DIR)
    parser.add_argument("--results-dir", default=config.BACKTEST_RESULTS_DIR)
    parser.add_argument(
        "--download",
        action="store_true",
        default=bool(config.BACKTEST_DOWNLOAD_DATA),
    )
    parser.add_argument("--no-download", action="store_false", dest="download")
    parser.add_argument(
        "--force-download",
        action="store_true",
        default=bool(config.BACKTEST_FORCE_DOWNLOAD),
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        default=bool(getattr(config, "BACKTEST_FAST_MODE", False)),
        help="Use faster rough-test settings unless explicitly overridden.",
    )
    parser.add_argument("--signal-step", type=int, default=None)
    parser.add_argument("--exit-timeframe", default="")
    parser.add_argument("--slice-max-rows", type=int, default=None)
    parser.add_argument(
        "--confirmation-types",
        default="",
        help="Comma-separated confirmation types to include, e.g. TREND or REVERSAL.",
    )
    parser.add_argument(
        "--no-btc-context",
        action="store_true",
        default=not bool(getattr(config, "BACKTEST_USE_BTC_CONTEXT", True)),
    )
    return parser


def apply_runtime_options(args):
    if args.fast:
        config.BACKTEST_SIGNAL_STEP_CANDLES = max(
            int(getattr(config, "BACKTEST_SIGNAL_STEP_CANDLES", 4)),
            8,
        )
        config.BACKTEST_SLICE_MAX_ROWS = min(
            int(getattr(config, "BACKTEST_SLICE_MAX_ROWS", 420)),
            320,
        )

        if not args.exit_timeframe:
            config.BACKTEST_EXIT_TIMEFRAME = "4h"

    if args.signal_step is not None:
        config.BACKTEST_SIGNAL_STEP_CANDLES = max(int(args.signal_step), 1)

    if args.exit_timeframe:
        config.BACKTEST_EXIT_TIMEFRAME = args.exit_timeframe.strip()

    if args.slice_max_rows is not None:
        config.BACKTEST_SLICE_MAX_ROWS = max(int(args.slice_max_rows), 0)

    if args.no_btc_context:
        config.BACKTEST_USE_BTC_CONTEXT = False

    config.BACKTEST_CONFIRMATION_TYPES = parse_confirmation_types(
        args.confirmation_types
    )


def run_backtest(args):
    apply_runtime_options(args)
    start_ms = parse_time_ms(args.start)
    end_ms = parse_time_ms(args.end, default=utc_now_ms())

    if end_ms <= start_ms:
        raise ValueError("Backtest end time must be after start time")

    symbols = parse_symbols(args.symbols)
    allowed_confirmation_types = parse_confirmation_types(
        getattr(config, "BACKTEST_CONFIRMATION_TYPES", [])
    )
    print(
        "Backtest settings | "
        f"STEP={getattr(config, 'BACKTEST_SIGNAL_STEP_CANDLES', 4)} | "
        f"EXIT_TF={getattr(config, 'BACKTEST_EXIT_TIMEFRAME', config.ENTRY_TIMEFRAME)} | "
        f"SLICE_ROWS={getattr(config, 'BACKTEST_SLICE_MAX_ROWS', 0)} | "
        f"BTC_CONTEXT={getattr(config, 'BACKTEST_USE_BTC_CONTEXT', True)} | "
        f"MULTI_TP={getattr(config, 'BACKTEST_MULTI_TP_ENABLED', False)} | "
        f"CONFIRMATION_TYPES={','.join(allowed_confirmation_types) if allowed_confirmation_types else 'ALL'}",
        flush=True,
    )

    all_trades = []
    failed_symbols = {}
    reversal_diagnostics = {}
    btc_frames = None

    if getattr(config, "BACKTEST_USE_BTC_CONTEXT", True):
        try:
            btc_frames = load_symbol_frames("BTCUSDT", args, start_ms, end_ms)
        except Exception as exc:
            failed_symbols["BTCUSDT"] = str(exc)

    for symbol in symbols:
        try:
            symbol_reversal_diagnostics = {}
            if symbol == "BTCUSDT" and btc_frames is not None:
                frames = btc_frames
            else:
                frames = load_symbol_frames(symbol, args, start_ms, end_ms)

            trades = generate_symbol_trades(
                symbol,
                frames,
                btc_frames,
                start_ms,
                end_ms,
                allowed_confirmation_types=allowed_confirmation_types,
                reversal_diagnostics=symbol_reversal_diagnostics,
            )
            reversal_diagnostics[symbol] = reversal_diagnostics_summary(
                symbol_reversal_diagnostics
            )
            all_trades.extend(trades)
            print(f"{symbol}: {len(trades)} simulated trades")

            diagnostic = reversal_diagnostics[symbol]
            top_reasons = ", ".join(
                f"{item['reason']}={item['count']}"
                for item in diagnostic["top_rejection_reasons"][:5]
            ) or "NONE"
            print(
                f"{symbol}: reversal diagnostics | "
                f"confirmed={diagnostic['chart_confirmed']} | "
                f"final_signals={diagnostic['final_signals']} | "
                f"near_misses={diagnostic['near_misses']} | "
                f"max_confidence={diagnostic['max_confidence']} | "
                f"top_reasons={top_reasons}",
                flush=True,
            )

        except Exception as exc:
            failed_symbols[symbol] = str(exc)
            print(f"{symbol}: skipped - {exc}")

    accepted_trades, skipped_by_limits = apply_position_limits(all_trades)
    summary = summarise_trades(accepted_trades, skipped_by_limits)
    summary.update({
        "symbols": symbols,
        "start": ms_to_iso(start_ms),
        "end": ms_to_iso(end_ms),
        "failed_symbols": failed_symbols,
        "generated_trades_before_position_limits": len(all_trades),
        "reversal_diagnostics": reversal_diagnostics,
        "settings": {
            "signal_step_candles": int(getattr(config, "BACKTEST_SIGNAL_STEP_CANDLES", 4)),
            "exit_timeframe": getattr(config, "BACKTEST_EXIT_TIMEFRAME", config.ENTRY_TIMEFRAME),
            "slice_max_rows": int(getattr(config, "BACKTEST_SLICE_MAX_ROWS", 0) or 0),
            "use_btc_context": bool(getattr(config, "BACKTEST_USE_BTC_CONTEXT", True)),
            "use_dca": bool(getattr(config, "BACKTEST_USE_DCA", False)),
            "dca_trigger_mode": getattr(
                config,
                "DCA_TRIGGER_MODE",
                "static_roi",
            ),
            "dca_trigger_rois": list(getattr(config, "DCA_TRIGGER_ROIS", [])),
            "dca_max_adverse_roi": float(
                getattr(config, "DCA_MAX_ADVERSE_ROI", 0)
            ),
            "dca_min_seconds_between_orders": int(
                getattr(config, "DCA_MIN_SECONDS_BETWEEN_ORDERS", 0)
            ),
            "dca_adaptive_atr_multipliers": list(
                getattr(config, "DCA_ADAPTIVE_ATR_MULTIPLIERS", [])
            ),
            "dca_adaptive_require_structure": bool(
                getattr(config, "DCA_ADAPTIVE_REQUIRE_STRUCTURE", True)
            ),
            "dca_adaptive_require_recovery": bool(
                getattr(config, "DCA_ADAPTIVE_REQUIRE_RECOVERY", True)
            ),
            "multi_tp_enabled": bool(
                getattr(config, "BACKTEST_MULTI_TP_ENABLED", False) and
                getattr(config, "MULTI_TP_ENABLED", False)
            ),
            "tp1_extra_slots_enabled": bool(
                getattr(config, "TP1_EXTRA_SLOTS_ENABLED", False)
            ),
            "tp1_extra_total_positions": int(
                getattr(config, "TP1_EXTRA_TOTAL_POSITIONS", 0)
            ),
            "tp1_extra_buy_positions": int(
                getattr(config, "TP1_EXTRA_BUY_POSITIONS", 0)
            ),
            "tp1_extra_sell_positions": int(
                getattr(config, "TP1_EXTRA_SELL_POSITIONS", 0)
            ),
            "tp1_close_position_pct": float(
                getattr(config, "TP1_CLOSE_POSITION_PCT", 50)
            ),
            "tp2_fallback_roi": float(
                getattr(config, "TP2_FALLBACK_ROI", 35)
            ),
            "runner_min_lock_roi": float(
                getattr(config, "TP1_RUNNER_MIN_LOCK_ROI", 5)
            ),
            "trend_sl_enabled": bool(
                getattr(config, "TREND_SL_ENABLED", getattr(config, "SL_ENABLED", False))
            ),
            "reversal_sl_enabled": bool(
                getattr(config, "REVERSAL_SL_ENABLED", getattr(config, "SL_ENABLED", False))
            ),
            "trend_max_sl_roi": float(
                getattr(config, "TREND_MAX_SL_ROI", getattr(config, "MAX_SL_ROI", 0))
            ),
            "reversal_max_sl_roi": float(
                getattr(config, "REVERSAL_MAX_SL_ROI", getattr(config, "MAX_SL_ROI", 0))
            ),
            "reversal_tp_max_roi": float(
                getattr(config, "REVERSAL_TP_MAX_ROI", 0)
            ),
            "reversal_profit_protection_enabled": bool(
                getattr(config, "REVERSAL_PROFIT_PROTECTION_ENABLED", True)
            ),
            "reversal_profit_trigger_roi": float(
                getattr(config, "REVERSAL_PROFIT_PROTECTION_TRIGGER_ROI", 12)
            ),
            "reversal_profit_lock_roi": float(
                getattr(config, "REVERSAL_PROFIT_PROTECTION_LOCK_ROI", 3)
            ),
            "reversal_profit_retrace_pct": float(
                getattr(config, "REVERSAL_PROFIT_PROTECTION_RETRACE_PCT", 50)
            ),
            "trend_profit_protection_enabled": bool(
                getattr(config, "TREND_PROFIT_PROTECTION_ENABLED", False)
            ),
            "trend_profit_trigger_roi": float(
                getattr(config, "TREND_PROFIT_PROTECTION_TRIGGER_ROI", 15)
            ),
            "trend_profit_lock_roi": float(
                getattr(config, "TREND_PROFIT_PROTECTION_LOCK_ROI", 5)
            ),
            "trend_profit_retrace_pct": float(
                getattr(config, "TREND_PROFIT_PROTECTION_RETRACE_PCT", 45)
            ),
            "reversal_entry_enabled": bool(
                getattr(config, "REVERSAL_ENTRY_ENABLED", True)
            ),
            "trend_exhaustion_guard_enabled": bool(
                getattr(config, "TREND_EXHAUSTION_GUARD_ENABLED", True)
            ),
            "trend_exhaustion_min_warning_points": float(
                getattr(config, "TREND_EXHAUSTION_MIN_WARNING_POINTS", 4)
            ),
            "confirmation_types": (
                allowed_confirmation_types
                if allowed_confirmation_types
                else ["ALL"]
            ),
        },
    })
    symbol_summary = summarise_by_symbol(accepted_trades)
    results_dir = Path(args.results_dir)
    written_files = [
        write_csv(results_dir / "trades.csv", accepted_trades),
        write_csv(results_dir / "symbol_summary.csv", symbol_summary),
        write_json(results_dir / "summary.json", summary),
    ]

    print("")
    print("Backtest complete")
    print(f"Trades: {summary['total_trades']}")
    print(f"Win rate: {summary['win_rate_pct']}%")
    print(f"Net PnL: {summary['net_pnl']}")
    print(f"Net return: {summary['net_return_pct']}%")
    print(f"Max drawdown: {summary['max_drawdown_pct']}%")
    print(f"Results: {results_dir.resolve()}")
    if any(item.name not in {"trades.csv", "symbol_summary.csv", "summary.json"} for item in written_files):
        print("Locked output fallback files:")
        for item in written_files:
            print(f"  {item.name}")


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    run_backtest(args)


if __name__ == "__main__":
    main()
