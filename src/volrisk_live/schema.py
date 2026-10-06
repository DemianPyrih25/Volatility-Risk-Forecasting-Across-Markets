"""Forecast payload schema ``volrisk-live/1`` and canonical JSON (docs/LIVE_SPEC.md §4–5)."""

from __future__ import annotations

import hashlib
import json
import math

SCHEMA = "volrisk-live/1"
ZERO_HASH = "0" * 64

PAYLOAD_KEYS = ("schema", "run_id", "run_utc", "frozen", "live_code_sha", "data", "forecasts", "risk", "implied",
                "checks")
FROZEN_KEYS = ("code_sha", "seal_ok", "sealed_utc", "holdout_opened_utc")
DATA_KEYS = ("last_session", "n_sessions", "rows_sha256")
FORECAST_KEYS = ("asset", "horizon", "model", "origin", "window_first", "window_last", "n_t", "F", "vol_ann")
RISK_KEYS = ("asset", "model", "date", "sigma", "var99", "var975", "es975")
IMPLIED_KEYS = ("asset", "origin", "iv", "iv_var_30d", "source")


def canonical_json(obj) -> str:
    """Deterministic JSON: sorted keys, compact separators, UTF-8 text, no NaN/inf (written as null)."""
    return json.dumps(_clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean(x):
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, dict):
        return {str(k): _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):  # numpy scalars
        return _clean(x.item())
    if hasattr(x, "isoformat"):  # date / datetime / Timestamp
        return x.isoformat()
    return x


def validate_payload(p: dict) -> None:
    """Raise ValueError if a payload does not follow ``volrisk-live/1``."""
    missing = [k for k in PAYLOAD_KEYS if k not in p]
    if missing:
        raise ValueError(f"payload misses keys {missing}")
    if p["schema"] != SCHEMA:
        raise ValueError(f"unknown schema {p['schema']!r}")
    for k in FROZEN_KEYS:
        if k not in p["frozen"]:
            raise ValueError(f"payload.frozen misses {k!r}")
    for asset, d in p["data"].items():
        for k in DATA_KEYS:
            if k not in d:
                raise ValueError(f"payload.data[{asset}] misses {k!r}")
    for name, keys in (("forecasts", FORECAST_KEYS), ("risk", RISK_KEYS), ("implied", IMPLIED_KEYS)):
        for i, row in enumerate(p[name]):
            miss = [k for k in keys if k not in row]
            if miss:
                raise ValueError(f"payload.{name}[{i}] misses {miss}")
