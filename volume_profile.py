"""Volume Profile (POC/VAH/VAL) approximated from OHLCV candles.

This module is observation-only: it is not imported by strategy.py and does
not participate in signal scoring, ranking, filtering, or execution. It
exists to log a real, independently-computed alternative to
detect_market_structure()'s crude high/low window (see strategy.py:100)
alongside the live scan, so there is comparative evidence before any
decision is made about promoting or replacing that logic.

Real volume-at-price data isn't available from klines (only OHLCV per
candle), so each candle's volume is distributed across the price bins its
high-low range overlaps, weighted by the fraction of the candle's range each
bin covers. This is the standard practical approximation for building a
volume profile from OHLCV bars rather than tick/order-book data.
"""

import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import config

_LAST_TELEMETRY_WRITE_AT = {}

TELEMETRY_FIELDNAMES = [
    "timestamp",
    "symbol",
    "timeframe",
    "close",
    "poc",
    "vah",
    "val",
    "price_low",
    "price_high",
    "total_volume",
    "bin_width",
    "price_vs_value_area",
    "structure_bullish",
    "structure_bearish",
    "structure_breakout",
    "structure_breakdown",
]


def compute_volume_profile(df, lookback=120, bins=24, value_area_pct=0.70):
    result = {
        "available": False,
        "reason": "",
        "poc": None,
        "vah": None,
        "val": None,
        "price_low": None,
        "price_high": None,
        "total_volume": 0.0,
        "bin_width": None,
        "lookback": lookback,
        "bins": bins,
    }

    if df is None or len(df) < 2:
        result["reason"] = "INSUFFICIENT_DATA"
        return result

    window = df.iloc[:-1].tail(lookback)

    if len(window) < 5:
        result["reason"] = "INSUFFICIENT_DATA"
        return result

    if bins < 2:
        result["reason"] = "INVALID_BIN_COUNT"
        return result

    highs = window["high"].to_numpy(dtype=float)
    lows = window["low"].to_numpy(dtype=float)
    volumes = window["volume"].to_numpy(dtype=float)

    price_low = float(np.min(lows))
    price_high = float(np.max(highs))

    if price_high <= price_low:
        result["reason"] = "DEGENERATE_RANGE"
        return result

    bin_edges = np.linspace(price_low, price_high, bins + 1)
    bin_volume = np.zeros(bins, dtype=float)

    for low, high, volume in zip(lows, highs, volumes):
        if volume <= 0:
            continue

        candle_range = max(high - low, 1e-12)
        start_bin = max(int(np.searchsorted(bin_edges, low, side="right") - 1), 0)
        end_bin = min(int(np.searchsorted(bin_edges, high, side="right") - 1), bins - 1)

        if start_bin > end_bin:
            continue

        if start_bin == end_bin:
            bin_volume[start_bin] += volume
            continue

        for bucket in range(start_bin, end_bin + 1):
            overlap_low = max(low, bin_edges[bucket])
            overlap_high = min(high, bin_edges[bucket + 1])
            overlap = max(overlap_high - overlap_low, 0.0)
            bin_volume[bucket] += volume * (overlap / candle_range)

    total_volume = float(bin_volume.sum())

    if total_volume <= 0:
        result["reason"] = "NO_VOLUME"
        return result

    poc_bin = int(np.argmax(bin_volume))
    bin_width = float(bin_edges[1] - bin_edges[0])
    poc_price = float((bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2)

    included_low = poc_bin
    included_high = poc_bin
    cumulative = float(bin_volume[poc_bin])
    target = total_volume * min(max(value_area_pct, 0.01), 1.0)

    while cumulative < target and (included_low > 0 or included_high < bins - 1):
        next_low = included_low - 1 if included_low > 0 else None
        next_high = included_high + 1 if included_high < bins - 1 else None
        low_volume = bin_volume[next_low] if next_low is not None else -1.0
        high_volume = bin_volume[next_high] if next_high is not None else -1.0

        if high_volume >= low_volume:
            included_high = next_high
            cumulative += bin_volume[included_high]
        else:
            included_low = next_low
            cumulative += bin_volume[included_low]

    result.update({
        "available": True,
        "reason": "OK",
        "poc": round(poc_price, 8),
        "vah": round(float(bin_edges[included_high + 1]), 8),
        "val": round(float(bin_edges[included_low]), 8),
        "price_low": round(price_low, 8),
        "price_high": round(price_high, 8),
        "total_volume": round(total_volume, 4),
        "bin_width": round(bin_width, 8),
    })
    return result


def classify_price_vs_value_area(price, profile):
    if not profile or not profile.get("available") or price is None:
        return ""

    vah = profile.get("vah")
    val = profile.get("val")

    if vah is None or val is None:
        return ""

    if price > vah:
        return "ABOVE_VALUE"

    if price < val:
        return "BELOW_VALUE"

    return "INSIDE_VALUE"


def _telemetry_path():
    path = Path(
        getattr(config, "VOLUME_PROFILE_TELEMETRY_PATH", "data/volume_profile_v7.csv")
    )

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def record_volume_profile_telemetry(symbol, timeframe, df, structure=None):
    """Log a volume-profile snapshot alongside detect_market_structure's read.

    Observation-only: wrapped so a failure here can never propagate into the
    scan/signal path, and rate-limited per (symbol, timeframe) so it can't
    grow unbounded across a 139-symbol universe scanned every few minutes.
    Returns the computed profile dict (or None if skipped/disabled/failed) -
    callers may use the return value, but nothing in this module writes to
    or reads from strategy.py's decision path.
    """
    if not getattr(config, "VOLUME_PROFILE_TELEMETRY_ENABLED", True):
        return None

    try:
        key = f"{symbol}:{timeframe}"
        min_interval = max(
            float(
                getattr(
                    config,
                    "VOLUME_PROFILE_TELEMETRY_MIN_INTERVAL_SECONDS",
                    240,
                )
            ),
            0,
        )
        now = time.monotonic()
        last_write = _LAST_TELEMETRY_WRITE_AT.get(key, 0.0)

        if min_interval > 0 and (now - last_write) < min_interval:
            return None

        profile = compute_volume_profile(
            df,
            lookback=int(getattr(config, "VOLUME_PROFILE_LOOKBACK", 120)),
            bins=int(getattr(config, "VOLUME_PROFILE_BINS", 24)),
            value_area_pct=float(
                getattr(config, "VOLUME_PROFILE_VALUE_AREA_PCT", 0.70)
            ),
        )

        if not profile.get("available"):
            return profile

        close = (
            float(df["close"].iloc[-2])
            if df is not None and len(df) > 1
            else None
        )
        structure = structure or {}
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "symbol": symbol,
            "timeframe": timeframe,
            "close": close,
            "poc": profile.get("poc"),
            "vah": profile.get("vah"),
            "val": profile.get("val"),
            "price_low": profile.get("price_low"),
            "price_high": profile.get("price_high"),
            "total_volume": profile.get("total_volume"),
            "bin_width": profile.get("bin_width"),
            "price_vs_value_area": classify_price_vs_value_area(close, profile),
            "structure_bullish": structure.get("bullish_structure", ""),
            "structure_bearish": structure.get("bearish_structure", ""),
            "structure_breakout": structure.get("bullish_breakout", ""),
            "structure_breakdown": structure.get("bearish_breakdown", ""),
        }

        path = _telemetry_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists() or path.stat().st_size == 0

        with path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=TELEMETRY_FIELDNAMES)

            if write_header:
                writer.writeheader()

            writer.writerow(row)

        _LAST_TELEMETRY_WRITE_AT[key] = now
        return profile

    except Exception:
        # Telemetry must never affect the scan/signal path.
        return None
