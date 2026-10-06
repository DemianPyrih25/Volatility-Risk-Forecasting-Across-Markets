"""Project configuration, paths and constants (see docs/SPEC.md)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "config.yaml"
FROZEN_PATH = ROOT / "config" / "frozen.yaml"

DATA = ROOT / "data"
RAW = DATA / "raw"
BRONZE = DATA / "bronze"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
HOLDOUT = DATA / "holdout"
RESULTS = DATA / "results"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
TABLES = REPORTS / "tables"

ASSETS = ("BTC", "ETH", "EURUSD", "SPX")
CRYPTO = ("BTC", "ETH")
HORIZONS = ("1d", "1w", "1m")


@dataclass(frozen=True)
class AssetCfg:
    name: str
    source: str
    symbol: str
    clock: str  # crypto | fx | xnys
    start: date
    iv: str
    price_scale: float | None = None


@lru_cache(maxsize=1)
def load() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cfg_hash() -> str:
    """Stable hash of the config, used by stages to decide whether cached outputs are current."""
    blob = json.dumps(load(), sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def seed() -> int:
    return int(load()["seed"])


def asset(name: str) -> AssetCfg:
    a = load()["assets"][name]
    return AssetCfg(
        name=name,
        source=a["source"],
        symbol=a["symbol"],
        clock=a["clock"],
        start=_d(a["start"]),
        iv=a["iv"],
        price_scale=float(a["price_scale"]) if "price_scale" in a else None,
    )


def clock(name: str) -> str:
    return asset(name).clock


def data_end() -> date:
    return _d(load()["dates"]["data_end"])


def holdout_start() -> date:
    return _d(load()["dates"]["holdout_start"])


def dev_end() -> date:
    """Last calendar date that belongs to the development period."""
    return date.fromordinal(holdout_start().toordinal() - 1)


def dev_eval_start() -> date:
    return _d(load()["dates"]["dev_eval_start"])


def horizon_days(h: str) -> int:
    return int(load()["horizons"][h]["days"])


def n_max(h: str, asset_name: str) -> int:
    return int(load()["horizons"][h]["n_max"][clock(asset_name)])


def har_lags(asset_name: str) -> tuple[int, int, int]:
    return tuple(load()["har_lags"][clock(asset_name)])  # type: ignore[return-value]


def window() -> int:
    return int(load()["walk_forward"]["window"])


def refit_every(kind: str, asset_name: str) -> int:
    """kind in {'garch', 'ml'}; number of sessions between parameter re-estimations."""
    return int(load()["walk_forward"]["refit"][kind][clock(asset_name)])


def _d(x) -> date:
    return x if isinstance(x, date) else date.fromisoformat(str(x))
