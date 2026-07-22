import json
import os
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import config
from logger import log_error, log_info, log_warning
from multi_tp import build_multi_tp_state, MULTI_TP_DISABLED


_MULTI_TP_FIELDS = {
    "multi_tp_active",
    "multi_tp_stage",
    "tp1_price",
    "tp1_close_pct",
    "tp1_requested_close_pct",
    "tp1_quantity",
    "tp1_order_quantity",
    "tp1_base_quantity",
    "tp1_order_id",
    "tp1_accounted_order_ids",
    "tp1_executed_quantity",
    "tp1_executed_quote",
    "tp1_repair_count",
    "tp1_original_price",
    "tp1_rearmed_from_price",
    "initial_sl_order_id",
    "tp1_trigger_seen_at",
    "tp1_order_status",
    "tp1_filled_at",
    "tp1_fill_price",
    "runner_basis_price",
    "runner_quantity",
    "runner_tp_price",
    "runner_tp_mode",
    "runner_tp_context",
    "runner_tp_order_id",
    "runner_sl_price",
    "runner_sl_mode",
    "runner_sl_order_id",
    "runner_protection_error",
}


class TradeStateLoadError(RuntimeError):
    """Raised when existing runtime state cannot be read safely."""


def apply_multi_tp_protection_state(item, tp_info):
    for field in _MULTI_TP_FIELDS:
        item.pop(field, None)

    item.update(build_multi_tp_state(tp_info))
    return item


def _state_path():
    path = Path(config.DCA_STATE_PATH)

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _lock_path():
    path = _state_path()
    return path.with_name(f"{path.name}.lock")


def _backup_path():
    path = _state_path()
    return path.with_name(f"{path.stem}.bak{path.suffix}")


def trade_state_file_exists():
    """Return whether primary state or its recoverable backup exists."""
    return _state_path().exists() or _backup_path().exists()


@contextmanager
def _state_file_lock():
    path = _lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    timeout = max(float(getattr(config, "DCA_STATE_LOCK_TIMEOUT_SECONDS", 10)), 0.1)
    stale_seconds = max(timeout * 3, 30)
    start = time.time()
    fd = None

    while fd is None:
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {time.time()}".encode("utf-8"))
        except FileExistsError:
            try:
                age = time.time() - path.stat().st_mtime

                if age > stale_seconds:
                    path.unlink()
                    continue

            except FileNotFoundError:
                continue
            except Exception as e:
                log_warning(f"trade state lock inspect warning: {e}")

            if time.time() - start >= timeout:
                raise TimeoutError(f"timed out waiting for trade state lock: {path}")

            time.sleep(0.05)

    try:
        yield
    finally:
        if fd is not None:
            os.close(fd)

        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except Exception as e:
            log_warning(f"trade state lock cleanup warning: {e}")


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def _state_write_retry_settings():
    return (
        max(int(getattr(config, "STATE_UPSERT_RETRY_ATTEMPTS", 3)), 1),
        max(
            float(getattr(config, "STATE_UPSERT_RETRY_DELAY_SECONDS", 0.25)),
            0,
        ),
    )


def _load_trade_state_unlocked():
    path = _state_path()

    if not path.exists():
        backup_path = _backup_path()

        if backup_path.exists():
            try:
                with backup_path.open("r", encoding="utf-8") as file:
                    state = json.load(file)

                if not isinstance(state, dict):
                    raise ValueError("backup state root is not an object")

                positions = state.setdefault("positions", {})
                pending = state.setdefault("pending_executions", {})

                if not isinstance(positions, dict) or not isinstance(
                    pending,
                    dict,
                ):
                    raise ValueError("backup state collections are invalid")

                log_warning(
                    "trade state primary file missing; recovered previous "
                    f"state from backup: {backup_path}"
                )
                return state

            except Exception as e:
                log_error(f"trade state backup load error: {e}")
                raise TradeStateLoadError(
                    f"runtime state backup is unreadable: {backup_path}"
                ) from e

        return {"positions": {}, "pending_executions": {}}

    try:
        with path.open("r", encoding="utf-8") as file:
            state = json.load(file)

        if not isinstance(state, dict):
            raise ValueError("state root is not an object")

        positions = state.setdefault("positions", {})
        pending = state.setdefault("pending_executions", {})

        if not isinstance(positions, dict) or not isinstance(pending, dict):
            raise ValueError("state collections are invalid")

        return state

    except Exception as e:
        log_error(f"trade state load error: {e}")
        backup_path = _backup_path()
        backup_hint = (
            f"; previous backup is available at {backup_path}"
            if backup_path.exists()
            else ""
        )
        raise TradeStateLoadError(
            f"runtime state is unreadable: {path}{backup_hint}"
        ) from e


def load_trade_state():
    return _load_trade_state_unlocked()


def _save_trade_state_unlocked(state):
    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    backup_path = _backup_path()
    backup_temp_path = path.with_name(
        f"{path.stem}.bak.tmp{path.suffix}"
    )

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(state, file, indent=2, sort_keys=True, default=str)
        file.flush()
        os.fsync(file.fileno())

    if path.exists():
        shutil.copyfile(path, backup_temp_path)

        with backup_temp_path.open("r+b") as backup_file:
            backup_file.flush()
            os.fsync(backup_file.fileno())

        os.replace(backup_temp_path, backup_path)

    os.replace(temp_path, path)

    if os.name != "nt":
        directory_fd = None

        try:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
            directory_fd = os.open(str(path.parent), flags)
            os.fsync(directory_fd)
        except OSError as exc:
            log_warning(f"trade state directory fsync warning: {exc}")
        finally:
            if directory_fd is not None:
                os.close(directory_fd)


def save_trade_state(state):
    try:
        with _state_file_lock():
            _save_trade_state_unlocked(state)
        return True

    except Exception as e:
        log_error(f"trade state save error: {e}")
        return False


def get_position_state(state, symbol):
    return state.get("positions", {}).get(symbol)


def get_pending_execution(state, symbol):
    return state.get("pending_executions", {}).get(symbol)


def upsert_pending_execution(state, symbol, data):
    attempts = max(
        int(getattr(config, "STATE_UPSERT_RETRY_ATTEMPTS", 3)),
        1,
    )
    retry_delay = max(
        float(getattr(config, "STATE_UPSERT_RETRY_DELAY_SECONDS", 0.25)),
        0,
    )

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                latest_state.setdefault("pending_executions", {})[symbol] = data
                _save_trade_state_unlocked(latest_state)
                state["pending_executions"] = latest_state.get(
                    "pending_executions",
                    {},
                )
            return True

        except Exception as e:
            log_error(
                f"{symbol} pending execution upsert error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False

def remove_pending_execution(state, symbol):
    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                pending = latest_state.setdefault("pending_executions", {})
                pending.pop(symbol, None)
                _save_trade_state_unlocked(latest_state)
                state["pending_executions"] = pending
            return True

        except Exception as e:
            log_error(
                f"{symbol} pending execution remove error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False


def _normalized_execution_client_ids(value):
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


def clear_confirmed_absent_entry_execution(
    state,
    symbol,
    expected_client_order_ids,
):
    """Atomically clear a proven-never-accepted ENTRY and its intent marker.

    The expected client IDs are a compare-and-delete token.  A concurrent
    pending-execution replacement, a non-ENTRY record, or any position marker
    that could represent live/managed exposure makes this operation fail
    closed without changing either collection.
    """
    expected_ids = _normalized_execution_client_ids(expected_client_order_ids)

    if not expected_ids:
        return False, "EXPECTED_CLIENT_ORDER_IDS_MISSING"

    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                pending_executions = latest_state.setdefault(
                    "pending_executions",
                    {},
                )
                positions = latest_state.setdefault("positions", {})
                pending = pending_executions.get(symbol)
                state["pending_executions"] = pending_executions
                state["positions"] = positions

                if not isinstance(pending, dict):
                    return False, "PENDING_EXECUTION_MISSING"

                current_ids = _normalized_execution_client_ids(
                    pending.get("client_order_ids")
                )

                if current_ids != expected_ids:
                    return False, "PENDING_CLIENT_ORDER_IDS_CHANGED"

                if (
                    str(pending.get("symbol") or "").strip().upper() !=
                    str(symbol or "").strip().upper()
                ):
                    return False, "PENDING_SYMBOL_CHANGED"

                if str(pending.get("context") or "").upper() != "ENTRY":
                    return False, "PENDING_CONTEXT_NOT_ENTRY"

                try:
                    pre_position_amount = abs(
                        float(pending.get("pre_position_amount", 0) or 0)
                    )
                except (TypeError, ValueError):
                    return False, "PENDING_PRE_POSITION_INVALID"

                if pre_position_amount > 1e-12:
                    return False, "PENDING_PRE_POSITION_NOT_FLAT"

                marker = positions.get(symbol)

                if marker is not None:
                    if not isinstance(marker, dict):
                        return False, "POSITION_MARKER_INVALID"

                    submission = marker.get("pending_submission")

                    try:
                        initial_quantity = abs(
                            float(marker.get("initial_quantity", 0) or 0)
                        )
                    except (TypeError, ValueError):
                        return False, "POSITION_MARKER_QUANTITY_INVALID"

                    safe_marker = bool(
                        marker.get("managed_by_bot") is True and
                        str(
                            marker.get("position_management_status") or ""
                        ).upper() == "ENTRY_READY_TO_SUBMIT" and
                        initial_quantity <= 1e-12 and
                        isinstance(submission, dict) and
                        str(submission.get("context") or "").upper() ==
                        "ENTRY" and
                        str(
                            submission.get("submission_phase") or ""
                        ).upper() == "READY_TO_SUBMIT"
                    )

                    if not safe_marker:
                        return False, "POSITION_MARKER_NOT_SAFE_TO_CLEAR"

                    marker_ids = _normalized_execution_client_ids(
                        submission.get("client_order_ids")
                    )

                    if marker_ids and marker_ids != expected_ids:
                        return False, "POSITION_MARKER_CLIENT_IDS_CHANGED"

                pending_executions.pop(symbol, None)

                if marker is not None:
                    positions.pop(symbol, None)

                _save_trade_state_unlocked(latest_state)
                state["pending_executions"] = pending_executions
                state["positions"] = positions
            return True, "CLEARED"

        except Exception as e:
            log_error(
                f"{symbol} confirmed-absent ENTRY clear error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False, "STATE_WRITE_FAILED"


def upsert_position_state(state, symbol, data):
    attempts = max(
        int(getattr(config, "STATE_UPSERT_RETRY_ATTEMPTS", 3)),
        1,
    )
    retry_delay = max(
        float(getattr(config, "STATE_UPSERT_RETRY_DELAY_SECONDS", 0.25)),
        0,
    )

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                latest_state.setdefault("positions", {})[symbol] = data
                _save_trade_state_unlocked(latest_state)
                state["positions"] = latest_state.get("positions", {})
            return True

        except Exception as e:
            log_error(
                f"{symbol} trade state upsert error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False


def remove_position_state(state, symbol):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            positions = latest_state.setdefault("positions", {})

            if symbol in positions:
                del positions[symbol]
                _save_trade_state_unlocked(latest_state)

            state["positions"] = latest_state.get("positions", {})
            return True

    except Exception as e:
        log_error(f"{symbol} trade state remove error: {e}")
        return False


def update_position_tp_status(state, symbol, tp_info, context=""):
    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                item = get_position_state(latest_state, symbol)

                if not item:
                    return False

                tp_info = tp_info or {}
                ok = bool(tp_info.get("ok"))
                item["tp_status"] = "CREATED" if ok else "FAILED"
                item["tp_price"] = tp_info.get("tp_price")
                item["tp_mode"] = tp_info.get("tp_mode")
                item["tp_context"] = context
                item["tp_updated_at"] = now_iso()

                if "sl_created" in tp_info or "sl_enabled" in tp_info:
                    sl_created = bool(tp_info.get("sl_created"))
                    sl_order = tp_info.get("sl_order") or {}
                    sl_order_id = (
                        sl_order.get("algoId") or
                        sl_order.get("orderId") or
                        ""
                    )
                    item["sl_status"] = (
                        "CREATED" if sl_created else "DISABLED"
                    )
                    item["sl_enabled"] = sl_created
                    item["sl_price"] = tp_info.get("sl_price")
                    item["sl_source"] = context

                    if sl_created:
                        item["hard_stop_price"] = tp_info.get("sl_price")
                        item["hard_stop_order_id"] = sl_order_id

                apply_multi_tp_protection_state(item, tp_info)

                if ok and str(context or "").upper().startswith("DCA_LEVEL_"):
                    item["tp_reprice_status"] = "COMPLETE"
                    item["tp_reprice_completed_at"] = now_iso()

                latest_state.setdefault("positions", {})[symbol] = item
                _save_trade_state_unlocked(latest_state)
                state["positions"] = latest_state.get("positions", {})
            return True

        except Exception as e:
            log_error(
                f"{symbol} TP status update error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False


def update_position_runtime_fields(state, symbol, updates):
    if not updates:
        return False

    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                item = get_position_state(latest_state, symbol)

                if not item:
                    return False

                item.update(updates)
                item["updated_at"] = now_iso()
                latest_state.setdefault("positions", {})[symbol] = item
                _save_trade_state_unlocked(latest_state)
                state["positions"] = latest_state.get("positions", {})
            return True

        except Exception as e:
            log_error(
                f"{symbol} runtime state update error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False


def prune_closed_positions(state, open_positions):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            positions = latest_state.setdefault("positions", {})
            closed_symbols = [
                symbol
                for symbol in positions
                if symbol not in open_positions
            ]

            for symbol in closed_symbols:
                log_info(f"{symbol} removed from trade state; position is closed")
                del positions[symbol]

            if closed_symbols:
                _save_trade_state_unlocked(latest_state)

            state["positions"] = latest_state.get("positions", {})
            return closed_symbols

    except Exception as e:
        log_error(f"trade state prune error: {e}")
        return []


def create_position_state(
    symbol,
    side,
    entry_price,
    quantity,
    planned_margin,
    used_margin,
    reference_price,
    level_info=None
):
    return {
        "symbol": symbol,
        "side": side,
        "managed_by_bot": True,
        "opened_at": now_iso(),
        "updated_at": now_iso(),
        "initial_entry": entry_price,
        "avg_entry": entry_price,
        "quantity": quantity,
        "planned_margin": planned_margin,
        "used_margin": used_margin,
        "dca_count": 0,
        "last_dca_price": None,
        "last_dca_at": None,
        "tp_status": "PENDING",
        "tp_price": None,
        "tp_mode": "",
        "reference_price": reference_price,
        "level_info": level_info or {},
        "reversal_peak_roi": 0,
        "reversal_profit_basis_entry": entry_price,
        "reversal_profit_exit_status": "",
        "trend_peak_roi": 0,
        "trend_profit_basis_entry": entry_price,
        "trend_profit_exit_status": "",
        "multi_tp_active": False,
        "multi_tp_stage": MULTI_TP_DISABLED,
    }


def _record_dca_fill_once(
    state,
    symbol,
    avg_entry,
    quantity,
    used_margin,
    dca_price,
    level_info=None,
    dca_count_increment=1
):
    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                log_error(f"{symbol} DCA fill not recorded | state missing")
                return False

            current_count = int(item.get("dca_count", 0) or 0)
            completed_level = None

            if level_info:
                try:
                    completed_level = int(level_info.get("dca_level") or 0)
                except Exception:
                    completed_level = None

            item["avg_entry"] = avg_entry
            item["quantity"] = quantity
            item["used_margin"] = round(
                float(item.get("used_margin", 0)) + used_margin,
                8
            )
            item["dca_count"] = current_count + dca_count_increment

            if completed_level:
                item["dca_count"] = max(item["dca_count"], completed_level)

            item["last_dca_price"] = dca_price
            item["last_dca_at"] = now_iso()
            if getattr(config, "DCA_FIXED_RISK_ENABLED", False):
                item["dca_recovery_status"] = "FILLED"
                item["dca_recovery_disabled"] = True
                item["dca_recovery_disabled_reason"] = "RECOVERY_ADD_CONSUMED"

                if getattr(config, "DCA_REPRICE_TP_AFTER_FILL", True):
                    item["tp_reprice_status"] = "PENDING"
                    item["tp_reprice_pending_at"] = now_iso()
                    item["tp_reprice_avg_entry"] = avg_entry
                    item["tp_reprice_quantity"] = quantity
                    item["tp_reprice_hard_stop_price"] = (
                        (level_info or {}).get("hard_stop_price") or
                        item.get("campaign_stop_price")
                    )
            item["updated_at"] = now_iso()
            item.pop("pending_dca", None)

            if getattr(config, "DCA_FIXED_RISK_ENABLED", False):
                for field in (
                    "dca_recovery_arm_price",
                    "dca_recovery_extreme_price",
                    "dca_recovery_extreme_at",
                    "dca_recovery_peak_adverse_roi",
                ):
                    item.pop(field, None)

            # A DCA fill changes the ROI basis, so route profit peaks must
            # restart from the new average entry.
            for route in ("reversal", "trend"):
                item[f"{route}_peak_roi"] = 0
                item[f"{route}_profit_floor_roi"] = 0
                item[f"{route}_profit_armed"] = False
                item[f"{route}_profit_basis_entry"] = avg_entry
                item[f"{route}_profit_exit_status"] = ""

            if level_info:
                item["last_dca_level_info"] = level_info
                completed_levels = item.setdefault("completed_dca_levels", [])

                if completed_level and completed_level not in completed_levels:
                    completed_levels.append(completed_level)
                    completed_levels.sort()

            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})
            return True

    except Exception as e:
        log_error(f"{symbol} DCA fill record error: {e}")
        return False


def record_dca_fill(
    state,
    symbol,
    avg_entry,
    quantity,
    used_margin,
    dca_price,
    level_info=None,
    dca_count_increment=1,
):
    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        if _record_dca_fill_once(
            state,
            symbol,
            avg_entry,
            quantity,
            used_margin,
            dca_price,
            level_info=level_info,
            dca_count_increment=dca_count_increment,
        ):
            return True

        if attempt < attempts and retry_delay > 0:
            time.sleep(retry_delay)

    return False


def _pending_dca_is_active(pending):
    if not pending:
        return False

    # An ambiguous exchange acknowledgement must never age out into a duplicate
    # DCA order. It stays reserved until reconciliation or operator repair.
    if pending.get("execution_unsettled"):
        return True

    # A fixed-risk recovery reservation is an exactly-once safety token. It
    # must never expire into a second add after a process crash in the narrow
    # window between an exchange fill and the durable fill record.
    if pending.get("fixed_risk_recovery"):
        return True

    timeout = max(float(getattr(config, "DCA_PENDING_TIMEOUT_SECONDS", 300)), 1)

    try:
        started_at = datetime.fromisoformat(pending.get("started_at", ""))
        return (datetime.now() - started_at).total_seconds() < timeout
    except Exception:
        return False


def has_active_dca_reservation(state, symbol):
    item = get_position_state(state, symbol)

    if not item:
        return False

    return _pending_dca_is_active(item.get("pending_dca"))


def reserve_dca_level(state, symbol, expected_dca_count, level_info=None):
    level = int(expected_dca_count) + 1

    try:
        with _state_file_lock():
            latest_state = _load_trade_state_unlocked()
            item = get_position_state(latest_state, symbol)

            if not item:
                return False, "state missing"

            current_count = int(item.get("dca_count", 0) or 0)

            if current_count != int(expected_dca_count):
                state["positions"] = latest_state.get("positions", {})
                return False, f"state count changed to {current_count}"

            pending = item.get("pending_dca")

            if _pending_dca_is_active(pending):
                return False, f"level {pending.get('level')} already pending"

            if getattr(config, "DCA_FIXED_RISK_ENABLED", False):
                recovery_status = str(
                    item.get("dca_recovery_status") or ""
                ).upper()
                recovery_level = int(
                    item.get("dca_recovery_level", 0) or 0
                )

                if recovery_status != "ARMED" or recovery_level != level:
                    return False, "recovery confirmation state changed"

                multi_tp_stage = str(item.get("multi_tp_stage") or "").upper()

                if (
                    multi_tp_stage in ("RUNNER_PENDING", "RUNNER_ACTIVE") or
                    (
                        multi_tp_stage == "TP1_PENDING" and
                        item.get("tp1_trigger_seen_at")
                    )
                ):
                    return False, "TP1 runner owns position"

                if any(
                    item.get(field) in (
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
                ):
                    return False, "position exit is pending"

                expected_stop = float(
                    (level_info or {}).get("hard_stop_price") or 0
                )
                persisted_stop = float(
                    item.get("campaign_stop_price") or 0
                )

                if (
                    expected_stop <= 0 or
                    persisted_stop <= 0 or
                    abs(expected_stop - persisted_stop) >
                    max(abs(persisted_stop) * 1e-10, 1e-10)
                ):
                    return False, "campaign stop changed"

            item["pending_dca"] = {
                "level": level,
                "started_at": now_iso(),
                "fixed_risk_recovery": bool(
                    getattr(config, "DCA_FIXED_RISK_ENABLED", False)
                ),
                "trigger_roi": (level_info or {}).get("trigger_roi"),
                "adverse_roi": (level_info or {}).get("adverse_roi"),
                "price_source": (level_info or {}).get("price_source"),
                "hard_stop_price": (level_info or {}).get("hard_stop_price"),
                "campaign_risk_budget_usdt": (level_info or {}).get(
                    "campaign_risk_budget_usdt"
                ),
                "recovery_risk_budget_usdt": (level_info or {}).get(
                    "recovery_risk_budget_usdt"
                ),
            }
            if getattr(config, "DCA_FIXED_RISK_ENABLED", False):
                item["dca_recovery_status"] = "RESERVED"
            item["updated_at"] = now_iso()
            latest_state.setdefault("positions", {})[symbol] = item
            _save_trade_state_unlocked(latest_state)
            state["positions"] = latest_state.get("positions", {})
            return True, "reserved"

    except Exception as e:
        log_error(f"{symbol} DCA reservation error: {e}")
        return False, str(e)


def clear_dca_reservation(state, symbol, level=None):
    attempts, retry_delay = _state_write_retry_settings()

    for attempt in range(1, attempts + 1):
        try:
            with _state_file_lock():
                latest_state = _load_trade_state_unlocked()
                item = get_position_state(latest_state, symbol)

                if not item:
                    return True

                pending = item.get("pending_dca")

                if not pending:
                    return True

                if (
                    level is not None and
                    int(pending.get("level", 0) or 0) != int(level)
                ):
                    return False

                item.pop("pending_dca", None)

                if (
                    str(item.get("dca_recovery_status") or "").upper() ==
                    "RESERVED" and
                    not item.get("dca_recovery_disabled")
                ):
                    item["dca_recovery_status"] = "ARMED"

                item["updated_at"] = now_iso()
                latest_state.setdefault("positions", {})[symbol] = item
                _save_trade_state_unlocked(latest_state)
                state["positions"] = latest_state.get("positions", {})
            return True

        except Exception as e:
            log_error(
                f"{symbol} DCA reservation clear error | "
                f"ATTEMPT={attempt}/{attempts}: {e}"
            )

            if attempt < attempts and retry_delay > 0:
                time.sleep(retry_delay)

    return False
