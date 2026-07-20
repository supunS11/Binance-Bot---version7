import math

import config
from exchange import get_symbol_precision


MIN_NOTIONAL = 5.0


def get_position_risk_budget(balance):
    """Return the maximum planned dollar loss for one complete campaign."""
    try:
        balance = max(float(balance or 0), 0)
        risk_pct = max(float(getattr(config, "POSITION_RISK_PCT", 0)), 0)
        risk_budget = balance * risk_pct / 100
        max_risk = max(
            float(getattr(config, "POSITION_RISK_MAX_USDT", 0)),
            0,
        )

        if max_risk > 0:
            risk_budget = min(risk_budget, max_risk)

        return max(risk_budget, 0)
    except Exception:
        return 0


def _round_quantity_down(quantity, precision):
    precision = max(int(precision or 0), 0)
    factor = 10 ** precision
    return math.floor(max(float(quantity or 0), 0) * factor) / factor


def calculate_position_size(
    balance,
    entry_price,
    sl_price,
    symbol,
    margin_override=None,
    risk_budget_override=None,
):

    try:

        margin = margin_override if margin_override is not None else config.MARGIN_PER_TRADE
        margin = max(float(margin or 0), 0)
        entry_price = float(entry_price or 0)

        if margin <= 0 or entry_price <= 0:
            return 0

        # =========================
        # BASE NOTIONAL FROM MARGIN
        # =========================
        base_notional = margin * config.LEVERAGE

        # convert to quantity
        qty_from_margin = base_notional / entry_price

        quantity = qty_from_margin

        if getattr(config, "RISK_BASED_POSITION_SIZING_ENABLED", False):
            stop_price = float(sl_price or 0)
            stop_distance = abs(entry_price - stop_price)
            risk_budget = (
                float(risk_budget_override)
                if risk_budget_override is not None
                else get_position_risk_budget(balance)
            )

            if stop_price <= 0 or stop_distance <= 0 or risk_budget <= 0:
                return 0

            quantity = min(quantity, risk_budget / stop_distance)

        # =========================
        # ENSURE BINANCE MINIMUM
        # =========================
        qty_min_notional = MIN_NOTIONAL / entry_price

        # =========================
        # OPTIONAL SAFETY (avoid huge spikes)
        # =========================
        max_notional = base_notional * 1.5
        max_qty = max_notional / entry_price

        quantity = min(quantity, max_qty)

        precision = get_symbol_precision(symbol)
        quantity = _round_quantity_down(quantity, precision)

        # Never force the Binance minimum if doing so would exceed either the
        # configured margin cap or the fixed campaign risk budget. The normal
        # min-notional validation in main will reject an undersized order.
        if quantity * entry_price < MIN_NOTIONAL:
            return 0

        if quantity <= 0:
            return 0

        return quantity

    except Exception:
        return 0
