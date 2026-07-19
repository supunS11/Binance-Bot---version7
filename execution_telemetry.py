import csv
import threading
from datetime import datetime, timezone
from pathlib import Path

import config
from logger import log_error


EXECUTION_FIELDS = (
    "timestamp_utc",
    "context",
    "symbol",
    "order_side",
    "position_side",
    "requested_quantity",
    "submitted_quantity",
    "executed_quantity",
    "residual_quantity",
    "fill_ratio_pct",
    "fully_filled",
    "position_verified",
    "position_closed",
    "status",
    "reference_price",
    "average_fill_price",
    "slippage_bps",
    "latency_ms",
    "submission_attempts",
    "verification_attempts",
    "pre_position_amount",
    "post_position_amount",
    "order_ids",
    "client_order_ids",
    "commission",
    "commission_asset",
    "error",
)


_journal_lock = threading.Lock()


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def calculate_slippage_bps(order_side, reference_price, average_fill_price):
    reference = _safe_float(reference_price)
    fill = _safe_float(average_fill_price)

    if reference <= 0 or fill <= 0:
        return None

    direction = 1 if str(order_side or "").upper() == "BUY" else -1
    return round(((fill - reference) / reference) * direction * 10000, 4)


def order_execution_values(order):
    order = order if isinstance(order, dict) else {}
    executed_quantity = _safe_float(order.get("executedQty"))
    cumulative_quote = _safe_float(
        order.get("cumQuote") or order.get("cumQuoteQty")
    )
    average_fill_price = _safe_float(order.get("avgPrice"))

    if (
        average_fill_price <= 0 and
        executed_quantity > 0 and
        cumulative_quote > 0
    ):
        average_fill_price = cumulative_quote / executed_quantity

    commission = (
        _safe_float(order.get("commission"))
        if order.get("commission") not in (None, "")
        else None
    )
    return {
        "executed_quantity": max(executed_quantity, 0),
        "cumulative_quote": max(cumulative_quote, 0),
        "average_fill_price": max(average_fill_price, 0),
        "commission": commission,
        "commission_asset": str(order.get("commissionAsset") or ""),
        "order_id": str(
            order.get("orderId") or ""
        ),
        "client_order_id": str(order.get("clientOrderId") or ""),
        "status": str(order.get("status") or "").upper(),
    }


def aggregate_order_execution(orders):
    values = [order_execution_values(order) for order in (orders or [])]
    executed_quantity = sum(item["executed_quantity"] for item in values)
    cumulative_quote = sum(item["cumulative_quote"] for item in values)
    weighted_total = 0.0
    weighted_quantity = 0.0

    for item in values:
        quantity = item["executed_quantity"]

        if quantity <= 0:
            continue

        quote = item["cumulative_quote"]

        if quote <= 0 and item["average_fill_price"] > 0:
            quote = item["average_fill_price"] * quantity

        if quote > 0:
            weighted_total += quote
            weighted_quantity += quantity

    average_fill_price = (
        weighted_total / weighted_quantity
        if weighted_quantity > 0
        else 0
    )

    statuses = [item["status"] for item in values if item["status"]]
    commission_assets = {
        item["commission_asset"]
        for item in values
        if item["commission_asset"]
    }
    return {
        "executed_quantity": round(executed_quantity, 12),
        "cumulative_quote": round(cumulative_quote, 12),
        "average_fill_price": round(average_fill_price, 12),
        "commission": (
            round(
                sum(
                    item["commission"]
                    for item in values
                    if item["commission"] is not None
                ),
                12,
            )
            if any(item["commission"] is not None for item in values)
            else None
        ),
        "commission_asset": (
            next(iter(commission_assets))
            if len(commission_assets) == 1
            else ",".join(sorted(commission_assets))
        ),
        "order_ids": ",".join(
            item["order_id"] for item in values if item["order_id"]
        ),
        "client_order_ids": ",".join(
            item["client_order_id"]
            for item in values
            if item["client_order_id"]
        ),
        "status": statuses[-1] if statuses else "",
    }


def _telemetry_path():
    path = Path(
        getattr(
            config,
            "EXECUTION_TELEMETRY_PATH",
            "data/execution_telemetry_v7.csv",
        )
    )

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def append_execution_telemetry(record):
    if not bool(getattr(config, "EXECUTION_TELEMETRY_ENABLED", True)):
        return False

    path = _telemetry_path()
    row = {field: "" for field in EXECUTION_FIELDS}
    row.update(record or {})
    row["timestamp_utc"] = row.get("timestamp_utc") or datetime.now(
        timezone.utc
    ).isoformat()

    try:
        with _journal_lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists() or path.stat().st_size == 0

            with path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=EXECUTION_FIELDS,
                    extrasaction="ignore",
                )

                if write_header:
                    writer.writeheader()

                writer.writerow(row)

        return True

    except Exception as exc:
        log_error(f"Execution telemetry write error: {exc}")
        return False
