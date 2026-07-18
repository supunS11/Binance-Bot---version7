import csv
from datetime import datetime
import json
import os
from pathlib import Path
import threading
import time
import uuid

import config
from logger import log_error
from signal_calibration import record_calibration_outcome


OUTCOME_FIELDS = [
    "signal_id",
    "opened_at",
    "observed_at",
    "symbol",
    "side",
    "route",
    "entry_price",
    "raw_score",
    "score_index",
    "rank_score",
    "horizon_hours",
    "observed_price",
    "directional_return_pct",
    "mfe_pct",
    "mae_pct",
    "flow_score",
    "breadth_score",
    "transition_score",
]


_lock = threading.RLock()
_state_cache = None


def _resolve_path(value):
    path = Path(value)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _state_path():
    return _resolve_path(
        getattr(
            config,
            "SIGNAL_OUTCOME_STATE_PATH",
            "data/signal_outcomes_pending_v7.json",
        )
    )


def _csv_path():
    return _resolve_path(
        getattr(
            config,
            "SIGNAL_OUTCOME_PATH",
            "data/signal_outcomes_v7.csv",
        )
    )


def _load_state():
    global _state_cache

    if _state_cache is not None:
        return _state_cache

    path = _state_path()

    if not path.exists():
        _state_cache = {"pending": {}}
        return _state_cache

    try:
        with path.open("r", encoding="utf-8") as handle:
            state = json.load(handle)

        state.setdefault("pending", {})
        _state_cache = state
        return state
    except Exception as exc:
        log_error(f"signal outcome state load error: {exc}")
        _state_cache = {"pending": {}}
        return _state_cache


def _save_state(state):
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, indent=2, sort_keys=True)

    os.replace(temporary, path)


def _route(candidate):
    signal = str(candidate.get("signal") or "").upper()
    side_data = (candidate.get("analysis") or {}).get(signal.lower(), {}) or {}

    if (side_data.get("continuation_pullback") or {}).get("active"):
        return "CONTINUATION_PULLBACK"

    if (side_data.get("trend_timing_rescue") or {}).get("active"):
        return "TREND_TIMING_RESCUE"

    return str(side_data.get("confirmation_type") or "TREND").upper()


def register_signal_outcome(candidate, entry_price):
    if not getattr(config, "SIGNAL_OUTCOME_TRACKING_ENABLED", True):
        return ""

    side = str(candidate.get("signal") or "").upper()
    side_data = (candidate.get("analysis") or {}).get(side.lower(), {}) or {}
    market_context = candidate.get("market_context") or {}
    flow = market_context.get("flow") or {}
    breadth = market_context.get("breadth") or {}
    transition = market_context.get("transition") or {}
    signal_id = f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}"
    now = time.time()
    route = _route(candidate)
    item = {
        "signal_id": signal_id,
        "opened_at": datetime.now().isoformat(timespec="seconds"),
        "opened_at_epoch": now,
        "symbol": candidate.get("symbol"),
        "side": side,
        "route": route,
        "entry_price": float(entry_price),
        "raw_score": float(side_data.get("score", 0) or 0),
        "score_index": float(side_data.get("uncapped_score_index", 0) or 0),
        "rank_score": float(candidate.get("rank_score", 0) or 0),
        "flow_score": float(flow.get(f"{side.lower()}_score", 0) or 0),
        "breadth_score": float(breadth.get(f"{side.lower()}_score", 0) or 0),
        "transition_score": float(transition.get(f"{side.lower()}_score", 0) or 0),
        "mfe_pct": 0.0,
        "mae_pct": 0.0,
        "completed_horizons": [],
    }

    try:
        with _lock:
            state = _load_state()
            state.setdefault("pending", {})[signal_id] = item
            _save_state(state)
        return signal_id
    except Exception as exc:
        log_error(f"{candidate.get('symbol')} signal outcome register error: {exc}")
        return ""


def _directional_return(side, entry_price, price):
    if entry_price <= 0:
        return 0.0

    raw = ((price - entry_price) / entry_price) * 100
    return raw if side == "BUY" else -raw


def _append_outcome(item, horizon, observed_price, directional_return):
    path = _csv_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    row = {
        "signal_id": item["signal_id"],
        "opened_at": item["opened_at"],
        "observed_at": datetime.now().isoformat(timespec="seconds"),
        "symbol": item["symbol"],
        "side": item["side"],
        "route": item["route"],
        "entry_price": item["entry_price"],
        "raw_score": item["raw_score"],
        "score_index": item["score_index"],
        "rank_score": item["rank_score"],
        "horizon_hours": horizon,
        "observed_price": observed_price,
        "directional_return_pct": round(directional_return, 4),
        "mfe_pct": round(float(item.get("mfe_pct", 0)), 4),
        "mae_pct": round(float(item.get("mae_pct", 0)), 4),
        "flow_score": item["flow_score"],
        "breadth_score": item["breadth_score"],
        "transition_score": item["transition_score"],
    }

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTCOME_FIELDS)

        if write_header:
            writer.writeheader()

        writer.writerow(row)


def observe_signal_outcomes(symbol, close_price, high_price=None, low_price=None):
    if not getattr(config, "SIGNAL_OUTCOME_TRACKING_ENABLED", True):
        return

    symbol = str(symbol or "").upper()
    close_price = float(close_price or 0)

    if close_price <= 0:
        return

    high_price = float(high_price or close_price)
    low_price = float(low_price or close_price)
    horizons = sorted(
        {
            max(float(value), 0.01)
            for value in getattr(config, "SIGNAL_OUTCOME_HORIZON_HOURS", [1, 4, 12, 24])
        }
    )
    now = time.time()

    try:
        with _lock:
            state = _load_state()
            pending = state.setdefault("pending", {})
            changed = False

            for signal_id, item in list(pending.items()):
                if item.get("symbol") != symbol:
                    continue

                side = item.get("side")
                entry_price = float(item.get("entry_price", 0) or 0)
                favorable_price = high_price if side == "BUY" else low_price
                adverse_price = low_price if side == "BUY" else high_price
                item["mfe_pct"] = max(
                    float(item.get("mfe_pct", 0) or 0),
                    _directional_return(side, entry_price, favorable_price),
                )
                item["mae_pct"] = min(
                    float(item.get("mae_pct", 0) or 0),
                    _directional_return(side, entry_price, adverse_price),
                )
                elapsed_hours = (
                    now - float(item.get("opened_at_epoch", now))
                ) / 3600
                completed = set(float(value) for value in item.get("completed_horizons", []))

                for horizon in horizons:
                    if horizon in completed or elapsed_hours < horizon:
                        continue

                    directional_return = _directional_return(
                        side,
                        entry_price,
                        close_price,
                    )
                    _append_outcome(
                        item,
                        horizon,
                        close_price,
                        directional_return,
                    )
                    completed.add(horizon)

                    if horizon == horizons[-1]:
                        threshold = float(
                            getattr(
                                config,
                                "SIGNAL_CALIBRATION_SUCCESS_RETURN_PCT",
                                0.1,
                            )
                        )
                        record_calibration_outcome(
                            item.get("route"),
                            item.get("raw_score"),
                            directional_return >= threshold,
                            directional_return,
                        )

                item["completed_horizons"] = sorted(completed)
                changed = True

                if horizons and horizons[-1] in completed:
                    del pending[signal_id]

            if changed:
                _save_state(state)

    except Exception as exc:
        log_error(f"{symbol} signal outcome observation error: {exc}")
