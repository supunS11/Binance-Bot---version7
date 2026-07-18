import json
import os
from pathlib import Path
import threading

import config
from logger import log_error


_lock = threading.RLock()
_cache = None


def _path():
    path = Path(
        getattr(
            config,
            "SIGNAL_CALIBRATION_PATH",
            "data/signal_calibration_v7.json",
        )
    )

    if not path.is_absolute():
        path = Path(__file__).resolve().parent / path

    return path


def _load_unlocked():
    global _cache

    if _cache is not None:
        return _cache

    path = _path()

    if not path.exists():
        _cache = {"version": 1, "buckets": {}}
        return _cache

    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        data.setdefault("version", 1)
        data.setdefault("buckets", {})
        _cache = data

    except Exception as exc:
        log_error(f"signal calibration load error: {exc}")
        _cache = {"version": 1, "buckets": {}}

    return _cache


def _save_unlocked(data):
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")

    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)

    os.replace(temporary, path)


def _route(route):
    route = str(route or "TREND").strip().upper()
    return route or "TREND"


def _bucket(raw_score):
    width = max(
        float(getattr(config, "SIGNAL_CALIBRATION_BUCKET_WIDTH", 5)),
        0.5,
    )
    value = max(float(raw_score or 0), 0)
    lower = int(value // width) * width
    return f"{lower:g}-{lower + width:g}"


def calibration_probability(route, raw_score):
    if not getattr(config, "SIGNAL_CALIBRATION_ENABLED", True):
        return {
            "available": False,
            "probability": None,
            "samples": 0,
            "source": "DISABLED",
        }

    key = f"{_route(route)}|{_bucket(raw_score)}"

    with _lock:
        item = _load_unlocked().get("buckets", {}).get(key, {})

    samples = int(item.get("samples", 0) or 0)
    minimum = max(
        int(getattr(config, "SIGNAL_CALIBRATION_MIN_SAMPLES", 30)),
        1,
    )

    if samples < minimum:
        return {
            "available": False,
            "probability": None,
            "samples": samples,
            "source": "COLLECTING",
            "bucket": key,
        }

    prior_strength = max(
        float(getattr(config, "SIGNAL_CALIBRATION_PRIOR_STRENGTH", 10)),
        0,
    )
    wins = float(item.get("wins", 0) or 0)
    probability = (
        (wins + (prior_strength * 0.5)) /
        (samples + prior_strength)
    )

    return {
        "available": True,
        "probability": round(probability, 4),
        "samples": samples,
        "source": "EMPIRICAL",
        "bucket": key,
    }


def record_calibration_outcome(route, raw_score, success, directional_return_pct):
    if not getattr(config, "SIGNAL_CALIBRATION_ENABLED", True):
        return

    key = f"{_route(route)}|{_bucket(raw_score)}"

    try:
        with _lock:
            data = _load_unlocked()
            item = data.setdefault("buckets", {}).setdefault(
                key,
                {
                    "samples": 0,
                    "wins": 0,
                    "return_sum_pct": 0.0,
                },
            )
            item["samples"] = int(item.get("samples", 0) or 0) + 1
            item["wins"] = int(item.get("wins", 0) or 0) + int(bool(success))
            item["return_sum_pct"] = round(
                float(item.get("return_sum_pct", 0) or 0) +
                float(directional_return_pct or 0),
                6,
            )
            _save_unlocked(data)

    except Exception as exc:
        log_error(f"signal calibration update error: {exc}")
