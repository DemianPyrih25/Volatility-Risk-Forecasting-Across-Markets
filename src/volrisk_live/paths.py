"""Paths of the live layer (docs/LIVE_SPEC.md §1). Nothing here points into a sealed location for writing."""

from __future__ import annotations

from volrisk import config as C

ROOT = C.ROOT
LIVE = C.DATA / "live"
LIVE_BRONZE = LIVE / "bronze" / "minute"
LIVE_FLAGS = LIVE / "bronze" / "flags"
LIVE_SILVER = LIVE / "silver"
LIVE_GOLD = LIVE / "gold"
LIVE_HOLDOUT = LIVE / "holdout"
LIVE_DAILY_DEV = LIVE_GOLD / "daily.parquet"
LIVE_DAILY_HOLDOUT = LIVE_HOLDOUT / "daily.parquet"
LIVE_IMPLIED = LIVE / "implied.parquet"
LIVE_RESULTS = LIVE / "forecasts"
LIVE_STATE = LIVE / "state.json"  # last update summary (end date, last sessions, consistency check)

FORECASTS = ROOT / "forecasts"  # meant to be committed by the project owner
RUNS = FORECASTS / "runs"
LEDGER = FORECASTS / "ledger.jsonl"
SCORES = FORECASTS / "scores.csv"
RISK_SCORES = FORECASTS / "risk_scores.csv"

DEMO = ROOT / "demo"  # committed snapshot of the results (scripts/make_demo_bundle.py) for fresh clones
DEMO_RESULTS = DEMO / "results"
DEMO_LIVE = DEMO / "live"

LIVE_REPORTS = C.REPORTS / "live"
TOMORROW_MD = LIVE_REPORTS / "tomorrow.md"
FORWARD_MD = LIVE_REPORTS / "forward_test.md"
VERIFY_MD = LIVE_REPORTS / "verification.md"
VERIFY_JSON = LIVE_REPORTS / "verification.json"

# Sealed artefacts the live layer must never write (asserted by tests).
SEALED_TARGETS = (C.GOLD, C.HOLDOUT, C.RESULTS, C.ROOT / "config", C.ROOT / "SEALED.json", C.ROOT / "holdout_log.jsonl")


def ensure_dirs() -> None:
    for p in (LIVE_BRONZE, LIVE_FLAGS, LIVE_SILVER, LIVE_GOLD, LIVE_HOLDOUT, LIVE_RESULTS, RUNS, LIVE_REPORTS):
        p.mkdir(parents=True, exist_ok=True)
