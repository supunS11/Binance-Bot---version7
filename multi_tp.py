import math

import config


TP1_PENDING = "TP1_PENDING"
RUNNER_PENDING = "RUNNER_PENDING"
RUNNER_ACTIVE = "RUNNER_ACTIVE"
MULTI_TP_DISABLED = "DISABLED"


def _safe_float(value, default=0.0):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def extract_order_id(order):
    order = order or {}
    status = str(
        order.get("algoStatus") or order.get("status") or ""
    ).upper()

    if status in {"REJECTED", "EXPIRED", "CANCELED", "CANCELLED", "FAILED"}:
        return ""

    return order.get("algoId") or order.get("orderId") or ""


def build_multi_tp_state(protection_result):
    result = protection_result or {}

    if not result.get("multi_tp_active"):
        return {
            "multi_tp_active": False,
            "multi_tp_stage": MULTI_TP_DISABLED,
        }

    return {
        "multi_tp_active": True,
        "multi_tp_stage": TP1_PENDING,
        "tp1_price": result.get("tp_price"),
        "tp1_close_pct": result.get("tp1_close_pct"),
        "tp1_requested_close_pct": result.get("tp1_requested_close_pct"),
        "tp1_quantity": result.get("tp1_quantity"),
        "tp1_order_quantity": result.get("tp1_quantity"),
        "tp1_base_quantity": result.get("tp1_base_quantity"),
        "tp1_order_id": extract_order_id(result.get("tp_order")),
        "tp1_accounted_order_ids": [],
        "tp1_executed_quantity": 0.0,
        "tp1_executed_quote": 0.0,
        "tp1_repair_count": 0,
        "tp1_original_price": result.get("tp_price"),
        "tp1_rearmed_from_price": None,
        "initial_sl_order_id": extract_order_id(result.get("sl_order")),
        "tp1_trigger_seen_at": None,
        "tp1_order_status": "NEW",
        "tp1_filled_at": None,
        "tp1_fill_price": None,
        "runner_basis_price": None,
        "runner_quantity": None,
        "runner_tp_price": None,
        "runner_tp_mode": "",
        "runner_tp_context": {},
        "runner_tp_order_id": "",
        "runner_sl_price": None,
        "runner_sl_mode": "",
        "runner_sl_order_id": "",
        "runner_protection_error": "",
    }


def tp1_trigger_reached(side, mark_price, tp1_price):
    mark_price = _safe_float(mark_price)
    tp1_price = _safe_float(tp1_price)

    if mark_price <= 0 or tp1_price <= 0:
        return False

    if str(side).upper() == "BUY":
        return mark_price >= tp1_price

    return mark_price <= tp1_price


def tp1_fill_confirmed(base_quantity, tp1_quantity, live_quantity):
    base_quantity = abs(_safe_float(base_quantity))
    tp1_quantity = abs(_safe_float(tp1_quantity))
    live_quantity = abs(_safe_float(live_quantity))

    if base_quantity <= 0 or tp1_quantity <= 0:
        return False

    tolerance = max(base_quantity * 1e-6, 1e-12)
    reduction = max(base_quantity - live_quantity, 0)
    return reduction + tolerance >= tp1_quantity


def roi_to_price(side, basis_price, roi, leverage=None):
    basis_price = _safe_float(basis_price)
    leverage = max(_safe_float(leverage, config.LEVERAGE), 1)
    move = (_safe_float(roi) / leverage) / 100

    if str(side).upper() == "BUY":
        return basis_price * (1 + move)

    return basis_price * (1 - move)


def calculate_runner_stop(
    side,
    original_entry,
    current_price,
    confirm_df,
    leverage=None,
):
    side = str(side or "").upper()
    original_entry = _safe_float(original_entry)
    current_price = _safe_float(current_price)
    leverage = max(_safe_float(leverage, config.LEVERAGE), 1)

    if side not in ("BUY", "SELL") or original_entry <= 0 or current_price <= 0:
        return None, {"reason": "RUNNER_STOP_INPUT_INVALID"}

    min_lock_roi = max(
        _safe_float(getattr(config, "TP1_RUNNER_MIN_LOCK_ROI", 5)),
        _safe_float(
            getattr(config, "TP1_RUNNER_BREAKEVEN_BUFFER_PCT", 0.12)
        ) * leverage,
        0,
    )
    lock_price = roi_to_price(side, original_entry, min_lock_roi, leverage)
    atr = 0.0
    structure_price = None

    try:
        closed_index = -2 if len(confirm_df) > 1 else -1
        closed = confirm_df.iloc[closed_index]
        atr = max(_safe_float(closed.get("atr")), 0)
        lookback = max(
            int(getattr(config, "TP1_RUNNER_STRUCTURE_LOOKBACK", 8)),
            2,
        )
        end = len(confirm_df) - 1 if len(confirm_df) > 1 else len(confirm_df)
        start = max(end - lookback, 0)
        window = confirm_df.iloc[start:end]
        structure_buffer = atr * max(
            _safe_float(
                getattr(config, "TP1_RUNNER_STRUCTURE_BUFFER_ATR", 0.15)
            ),
            0,
        )

        if not window.empty:
            if side == "BUY":
                structure_price = _safe_float(window["low"].min()) - structure_buffer
            else:
                structure_price = _safe_float(window["high"].max()) + structure_buffer
    except (AttributeError, KeyError, TypeError, ValueError):
        structure_price = None

    atr_buffer = atr * max(
        _safe_float(getattr(config, "TP1_RUNNER_ATR_BUFFER_MULT", 0.50)),
        0,
    )
    if atr > 0:
        atr_price = (
            current_price - atr_buffer
            if side == "BUY"
            else current_price + atr_buffer
        )
    else:
        atr_price = lock_price
    min_distance = max(
        atr * max(
            _safe_float(
                getattr(config, "TP1_RUNNER_MIN_STOP_DISTANCE_ATR", 0.20)
            ),
            0,
        ),
        current_price * 0.0005,
    )

    if side == "BUY":
        max_stop = current_price - min_distance

        if lock_price >= max_stop:
            return None, {
                "reason": "RUNNER_STOP_NO_SAFE_ROOM",
                "lock_price": lock_price,
                "max_stop": max_stop,
            }

        candidate = structure_price if structure_price else atr_price
        stop_price = min(max(lock_price, candidate), max_stop)
    else:
        min_stop = current_price + min_distance

        if lock_price <= min_stop:
            return None, {
                "reason": "RUNNER_STOP_NO_SAFE_ROOM",
                "lock_price": lock_price,
                "min_stop": min_stop,
            }

        candidate = structure_price if structure_price else atr_price
        stop_price = max(min(lock_price, candidate), min_stop)

    source = "STRUCTURE"

    if structure_price is None:
        source = "ATR"
    elif abs(stop_price - lock_price) <= max(original_entry * 1e-9, 1e-12):
        source = "PROFIT_LOCK_FLOOR"

    return stop_price, {
        "reason": "RUNNER_STOP_OK",
        "source": source,
        "min_lock_roi": round(min_lock_roi, 4),
        "lock_price": lock_price,
        "structure_price": structure_price,
        "atr_price": atr_price,
        "atr": atr,
    }
