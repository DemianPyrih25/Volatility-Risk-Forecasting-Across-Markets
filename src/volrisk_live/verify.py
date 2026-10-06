"""Verification report (docs/LIVE_SPEC.md §7) -> ``reports/live/verification.md`` + ``verification.json``.

``uv run python -m volrisk_live verify [--full]`` re-runs every check from the files in the repository and public
sources. Each check is a dict ``{id, title, status, reason, method, evidence, seconds}`` with ``status`` PASS, FAIL
or SKIPPED; ``method`` is one plain-language sentence saying what is compared with what, ``evidence`` holds the
numbers. The groups follow §7:

1. frozen code unchanged; 2. holdout opened exactly once; 3. raw data authenticity (Binance zips vs the CHECKSUM files
Binance serves now, a re-download sample drawn with a fresh seed, every raw file vs its download record);
4. independent sources (Coinbase, FRED, ECB); 5. reproducibility (stored files compared with the sealed ones; with
``--full`` the dev and holdout walk-forwards are re-run in memory and the gold tables rebuilt from the raw files);
6. negative controls (leakage perturbation test on every frozen forecast model and COMBO, at a fixed cut-off and at
each periodically refitted model's refit origin; three cheating toy models, one with a training-label leak; a
pure-noise model in the MCS); 7. no tuning after seeing results; 8. forward-test ledger (hash chain, entries pinned in
Bitcoin, OpenTimestamps proofs of the payloads vs their target windows, recorded forecasts re-derived).

The method text says what the files alone cannot show: comparing a file with another file proves consistency, not
derivation (only ``--full`` re-derives); runs dropped from the end of the ledger together with all their files are
visible only against a head hash published elsewhere (the owner's push of ``forecasts/``).

The pass rules are constants of this module (``CORR_MIN``, ``MIN_DAYS``, ``REPRO_RTOL``, ...) and are printed in the
report. A network failure, or a claim that only the network can check (a Bitcoin attestation), makes a check
SKIPPED, never PASS or FAIL; any other error inside a check is a FAIL. Nothing is written except
``reports/live/verification.*`` (``--full`` also builds a throw-away copy of the gold tables in a temporary directory,
deleted afterwards); the frozen package is only imported.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
import logging
import math
import re
import secrets
import shutil
import sys
import tempfile
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from datetime import time as dtime
from io import BytesIO
from pathlib import Path
from urllib.error import URLError
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import polars as pl
import requests
import yaml

from volrisk import config as C
from volrisk import holdout, io
from volrisk.models.base import align_targets, eligible_end, predict_mask_for, to_forecast_frame
from volrisk.pipeline import FORECAST_MODELS
from volrisk.targets import build_targets
from volrisk_live import context, paths, schema

log = logging.getLogger("volrisk_live.verify")

REPORT_SCHEMA = "volrisk-verify/1"
SEED = 20261002  # project seed (config ``seed``): perturbation and noise controls
STATUSES = ("PASS", "FAIL", "SKIPPED")
USER_AGENT = "Mozilla/5.0 (volrisk verification)"

# ---- pass rules, fixed in code before any comparison was run (printed in the report) --------------------------
CORR_MIN = {"BTC": 0.99, "ETH": 0.99, "SPX": 0.99, "EURUSD": 0.90}  # corr of daily log returns vs the source
MIN_DAYS = 250  # matched days needed for a correlation verdict
REPRO_RTOL = 1e-9  # forecasts: same tolerance as the holdout reproduction (volrisk.pipeline.REPRO_RTOL), atol 0
DIFF_TOL_BP = 25.0  # days with |return difference| above this are counted (same tolerance as the DQ CFD check)
BLOCK_TIME_SLACK = timedelta(hours=2)  # a Bitcoin block time may differ ~2 h from real time (ledger.BLOCK_TIME_SLACK)
ANCHOR_LAG = timedelta(hours=24)  # an entry / payload must be Bitcoin-attested within this (ledger.MAX_ANCHOR_LAG)

# ---- raw data -------------------------------------------------------------------------------------------------
BITCOIN_TIP_URLS = ("https://blockstream.info/api/blocks/tip/hash", "https://mempool.space/api/blocks/tip/hash")
RAW_FILE_RE = re.compile(r"\.(zip|zip\.CHECKSUM|bi5)$")  # raw files the manifest must record
HASH_WORKERS = 8

# ---- independent sources --------------------------------------------------------------------------------------
COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/candles"
COINBASE_MAX = 300  # candles per request (Coinbase limit)
COINBASE_PRODUCTS = {"BTC": "BTC-USD", "ETH": "ETH-USD"}
ECB_URL = "https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A"
ECB_FIX = dtime(14, 15)  # "ECB reference exchange rate, US dollar/Euro, 2.15 pm (C.E.T.)"
ECB_TZ = "Europe/Berlin"
FIX_TOLERANCE = "10m"  # Dukascopy price at the fix = last real minute closing at most 10 minutes before it

# ---- negative controls (dev data only) ------------------------------------------------------------------------
LEAK_T0 = date(2023, 6, 30)  # fixed cut-off of the perturbation test (inside the dev period)
LEAK_CELLS = (("BTC", "1d"), ("BTC", "1m"), ("SPX", "1d"), ("SPX", "1m"))
LEAK_MODELS = (*FORECAST_MODELS, "COMBO")  # every frozen forecast model and the headline combination
LEAK_ONE_CELL = {"MLP": (("BTC", "1d"),)}  # about a minute per MLP walk-forward: one cell only
LEAK_SPLIT_T0 = ("MLP",)  # one process per t0 for the slowest model (it sets the wall time)
LEAK_COST = ("MLP", "COMBO", "LGBM", "GJR", "GARCH")  # slowest first, so the process pool finishes early
PERTURB_LOG_RANGE = 1.5  # every float after t0 is multiplied by exp(U(-1.5, 1.5)), i.e. x0.22 .. x4.5
NOISE_CELL = ("BTC", "1d")
NOISE_WINDOW = (date(2024, 10, 1), date(2025, 9, 30))  # last dev year: short window for speed

# Hyperparameters compared with docs/SPEC.md (§6 walk-forward and lag grid, §7 models).
LGBM_KEYS = ("objective", "n_estimators", "learning_rate", "num_leaves", "min_child_samples", "colsample_bytree",
             "reg_lambda", "n_jobs")
SPEC_PATH = C.ROOT / "docs" / "SPEC.md"
DEVIATIONS_PATH = C.ROOT / "docs" / "DEVIATIONS.md"
SENSITIVITY_LGBM = C.TABLES / "sensitivity_lgbm_dev.csv"

# ---- forward-test ledger --------------------------------------------------------------------------------------
RUN_FILE_RE = re.compile(r"^(\d{8}T\d{6}Z)\.(json|entry)(\.ots)?$")  # runs/ files that belong to a ledger entry
EX_ANTE_PAD = 14  # calendar days searched for the scheduled sessions of a window (as forecast._SCHEDULE_PAD)
FC_KEYS = ["asset", "horizon", "model", "origin"]
RISK_KEYS = ["asset", "model", "date"]
RISK_VALUES = ["sigma", "var99", "var975", "es975"]

RULES = (
    "1 - `verify_seal()` lists no changed artefact, and the code hash now = SEALED.json = the hash logged at the "
    "holdout opening.",
    "2 - exactly one opening in holdout_log.jsonl (`n_previous` 0, no re-run reason), made after the seal, with the "
    "same hashes as SEALED.json, and SEALED.json unchanged since the opening.",
    "3 - every Binance zip matches the CHECKSUM file Binance serves for it now (downloaded again) and its local copy; "
    "at least `sample` raw files drawn with a fresh seed (the newest Bitcoin block hash, printed; `--seed` repeats a "
    "draw) are re-downloaded byte-identical; every raw file still has the SHA-256 recorded at its download and no "
    "raw file is unrecorded (any difference = FAIL; checks the network could not complete = SKIPPED).",
    f"4 - at least {MIN_DAYS} matched days and a correlation of daily log returns >= {CORR_MIN['BTC']} (BTC, ETH vs "
    f"Coinbase), >= {CORR_MIN['SPX']} (S&P 500 vs FRED), >= {CORR_MIN['EURUSD']} (EUR/USD at the ECB fixing time "
    "vs the ECB reference rate).",
    f"5 - compared files are equal: tables exactly (NaN = NaN), forecasts to rtol {REPRO_RTOL:g}, atol 0; the holdout "
    "result files (written at the opening, not in SEALED.json) still have the SHA-256 the first ledger payload "
    "recorded; with --full the re-run walk-forwards match every stored row in both directions and the gold tables "
    "rebuilt from the raw files equal the sealed ones.",
    "6 - for every frozen forecast model and COMBO, no forecast at an origin <= t0 changes when the data after t0 is "
    "perturbed (and forecasts after t0 do change), for t0 = the fixed cut-off and each periodically refitted model's "
    "last refit origin before it; all three cheating models are flagged (the training-label cheat at its refit "
    "origin); the noise model has the worst mean QLIKE and MCS p-value < 0.10.",
    "7 - every compared hyperparameter in config/frozen.yaml and config/config.yaml equals docs/SPEC.md, and the "
    "frozen LightGBM keeps the pre-registered number of leaves.",
    f"8 - the hash chain recomputes and no payload, entry file, proof or score exists outside it (UTC days without a "
    f"run are listed); every entry is pinned in Bitcoin within {ANCHOR_LAG.total_seconds() / 3600:g} h of its run; "
    f"every payload has an .ots proof of its exact bytes, every Bitcoin attestation verifies and its block time + "
    f"{BLOCK_TIME_SLACK.total_seconds() / 3600:g} h precedes the close of the payload's first target session; every "
    "recorded forecast equals the frozen walk-forward at its origin and has the ex-ante window of the session "
    "calendar.",
)

NETWORK_ERRORS = (requests.RequestException, ConnectionError, TimeoutError, URLError)
CheckFn = Callable[[], tuple]


class SkipCheck(Exception):
    """Raised inside a check that cannot run here: status SKIPPED with this reason (and optional evidence)."""

    def __init__(self, reason: str, evidence: dict | None = None):
        super().__init__(reason)
        self.evidence = evidence or {}


# --------------------------------------------------------------------------------------------- helpers
def _sleep(seconds: float) -> None:  # patched in tests
    time.sleep(seconds)


def _http_get(url: str, params: dict | None = None, timeout: float = 60.0, retries: int = 3) -> bytes:
    """GET with a few retries on transient errors; raises ``requests.RequestException`` if it keeps failing."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT})
            if r.status_code == 429 or r.status_code >= 500:
                raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}", response=r)
            if r.status_code != 200:  # 4xx: retrying will not help
                raise requests.HTTPError(f"HTTP {r.status_code} for {r.url}", response=None)
            return r.content
        except requests.HTTPError as e:
            last = e
            if e.response is None:
                raise
        except requests.RequestException as e:
            last = e
        if attempt + 1 < retries:
            _sleep(1.0 * 2**attempt)
    assert last is not None
    raise last


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rel(p: Path | str) -> str:
    try:
        return Path(p).resolve().relative_to(C.ROOT.resolve()).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _utc(text: str) -> datetime:
    t = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _iso(t: datetime | None) -> str | None:
    return None if t is None else t.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _plain(x):
    """JSON-ready copy: NaN/inf -> None, numpy scalars -> Python, dates -> ISO, paths -> repo-relative."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_plain(v) for v in x)
    if isinstance(x, np.ndarray):
        return [_plain(v) for v in x.tolist()]
    if isinstance(x, pd.DataFrame):
        return [_plain(r) for r in x.to_dict("records")]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return _rel(x)
    if isinstance(x, pd.Timestamp):
        return None if pd.isna(x) else (x.date().isoformat() if x == x.normalize() and x.tz is None else x.isoformat())
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        return _plain(x.item())
    if x is None or isinstance(x, (str, int)):
        return x
    return str(x)


def _ok_of(res) -> bool | None:
    """Best-effort verdict of another module's result: bool, ``{"ok": ...}``, a list of problems, or None."""
    if isinstance(res, bool):
        return res
    if isinstance(res, dict):
        for k in ("ok", "passed", "valid", "intact"):
            if k in res and isinstance(res[k], bool):
                return res[k]
        return None
    if isinstance(res, list):
        return not res
    return None


def _live_module(name: str):
    try:
        return importlib.import_module(f"volrisk_live.{name}")
    except ImportError as e:
        raise SkipCheck(f"volrisk_live.{name} is not available ({e})") from None


def _sealed_hashes() -> dict[str, str]:
    try:
        return json.loads(holdout.SEALED.read_text(encoding="utf-8")).get("hashes") or {}
    except (OSError, ValueError):
        return {}


def file_record(path: Path) -> dict:
    """``{file, sha256, bytes, sealed_as}`` of a compared file; ``sealed_as`` names the SEALED.json entry with the
    same hash (None for a file written after the seal, which SEALED.json cannot vouch for)."""
    p = Path(path)
    if not p.exists():
        return {"file": _rel(p), "sha256": None, "bytes": None, "sealed_as": None}
    sha = _sha256_file(p)
    return {"file": _rel(p), "sha256": sha, "bytes": p.stat().st_size,
            "sealed_as": next((k for k, v in _sealed_hashes().items() if v == sha), None)}


def _run_tasks(fn: Callable, tasks: list, workers: int) -> list:
    """``map(fn, tasks)`` in worker processes (``fn`` must be module level; spawned workers see the frozen
    ``data_end``, which every task here uses) or in this process when ``workers <= 1``."""
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as ex:
            return list(ex.map(fn, tasks))
    return [fn(t) for t in tasks]


def run_check(cid: str, title: str, method: str, fn: CheckFn) -> dict:
    """Run one check; ``fn`` returns ``(status, evidence[, reason])`` or raises ``SkipCheck``."""
    t0 = time.perf_counter()
    try:
        out = fn()
        status, evidence = out[0], out[1]
        reason = out[2] if len(out) > 2 else None
    except SkipCheck as e:
        status, evidence, reason = "SKIPPED", e.evidence, str(e)
    except NETWORK_ERRORS as e:
        status, evidence, reason = "SKIPPED", {}, f"network unavailable: {type(e).__name__}: {e}"[:500]
    except Exception as e:  # noqa: BLE001 - a check that crashes is not a pass
        log.exception("check %s failed with an error", cid)
        msg = f"{type(e).__name__}: {e}"
        status, evidence, reason = "FAIL", {"error": msg[:2000]}, msg[:500]
    if status not in STATUSES:
        raise ValueError(f"check {cid} returned unknown status {status!r}")
    secs = round(time.perf_counter() - t0, 1)
    log.info("%-7s %-3s %s (%.1fs)%s", status, cid, title, secs, f" - {reason}" if reason else "")
    return {"id": cid, "title": title, "status": status, "reason": reason, "method": method,
            "evidence": _plain(evidence), "seconds": secs}


# --------------------------------------------------------------------------------------------- 1-2 seal, opening
def check_seal() -> tuple:
    """§7.1: frozen code unchanged since the seal and since the logged holdout opening."""
    bad = holdout.verify_seal()
    sealed = json.loads(holdout.SEALED.read_text(encoding="utf-8"))
    now = holdout.code_sha()
    opened = holdout.openings()
    logged = opened[0].get("hashes", {}).get("code") if opened else None
    ev = {
        "summary": f"{len(holdout.code_files())} files under src/volrisk/, changed artefacts: {bad or 'none'}",
        "verify_seal": bad,
        "n_code_files": len(holdout.code_files()),
        "code_sha256_now": now,
        "code_sha256_sealed": sealed["hashes"].get("code"),
        "code_sha256_logged_at_opening": logged,
        "sealed_utc": sealed.get("created_utc"),
        "changed_code_files": holdout.changed_code_files() if "code" in bad else [],
    }
    ok = not bad and logged is not None and now == ev["code_sha256_sealed"] == logged
    reason = None
    if not ok:
        reason = "no holdout opening logged" if logged is None else f"changed since the seal: {bad or ['code']}"
    return _status(ok), ev, reason


def _earliest_mtime(folder: Path) -> str | None:
    files = [p for p in folder.rglob("*") if p.is_file()] if folder.is_dir() else []
    if not files:
        return None
    return _iso(datetime.fromtimestamp(min(p.stat().st_mtime for p in files), UTC))


def check_opened_once() -> tuple:
    """§7.2: one logged opening whose hashes match SEALED.json."""
    entries = holdout.openings()
    sealed = json.loads(holdout.SEALED.read_text(encoding="utf-8"))
    ev: dict = {"n_openings": len(entries), "sealed_utc": sealed.get("created_utc")}
    if not entries:
        return "FAIL", ev, "holdout_log.jsonl has no opening"
    e = entries[0]
    logged = e.get("hashes", {})
    ev.update(
        opened_utc=e.get("utc"),
        n_previous=e.get("n_previous"),
        rerun_reason=e.get("rerun_reason"),
        opened_after_seal=_utc(e["utc"]) > _utc(sealed["created_utc"]),
        hashes_match_seal=logged == sealed["hashes"],
        seal_file_unchanged_since_opening=e.get("seal_sha") == holdout.file_sha(holdout.SEALED),
        hashes=[{"artefact": k, "sealed": v, "logged_at_opening": logged.get(k), "equal": logged.get(k) == v}
                for k, v in sealed["hashes"].items()],
        # file times are local metadata (not tamper-proof): shown as context only
        earliest_holdout_result_file_utc=_earliest_mtime(holdout.RESULTS_HOLDOUT),
    )
    ok = (len(entries) == 1 and e.get("n_previous") == 0 and e.get("rerun_reason") is None
          and ev["opened_after_seal"] and ev["hashes_match_seal"] and ev["seal_file_unchanged_since_opening"])
    ev["summary"] = f"{len(entries)} opening at {e.get('utc')}, seal created {sealed.get('created_utc')}"
    return _status(ok), ev, None if ok else "opening log does not match the seal"


# --------------------------------------------------------------------------------------------- 3 raw data
def draw_seed(network: bool = True) -> tuple[int, dict]:
    """A sample seed nobody can know before the run: the last 32 bits of the newest Bitcoin block hash (public, so
    anyone can check the seed was not picked), or the operating system's random source when no explorer answers.
    The report prints it; ``--seed`` repeats a draw."""
    drawn = _iso(datetime.now(UTC))
    if network:
        for url in BITCOIN_TIP_URLS:
            try:
                h = _http_get(url, timeout=15.0, retries=1).decode("ascii", "replace").strip().lower()
            except Exception:  # noqa: BLE001 - any failure falls back to the next source
                continue
            if re.fullmatch(r"[0-9a-f]{64}", h):
                return int(h[-8:], 16), {"source": "newest Bitcoin block hash (last 32 bits)", "block_hash": h,
                                         "url": url, "drawn_utc": drawn}
    return secrets.randbits(32), {"source": "operating-system random number (no block explorer reachable)",
                                  "drawn_utc": drawn}


def binance_zip_url(zip_path: Path) -> str:
    """Public URL of a local Binance zip ``{raw}/{SYMBOL}/{monthly|daily}/{file}`` (inverse of ``binance.raw_path``)."""
    from volrisk.data import binance

    p = Path(zip_path)
    return f"{binance.BASE_URL}/{p.parent.name}/klines/{p.parent.parent.name}/1m/{p.name}"


def _remote_checksum(zip_path: Path) -> tuple[str | None, str | None]:
    """(sha256 Binance serves in ``<zip url>.CHECKSUM``, error)."""
    from volrisk.data import binance

    try:
        text = _http_get(binance.checksum_url(binance_zip_url(zip_path)), timeout=30.0).decode("ascii", "replace")
    except NETWORK_ERRORS as e:
        return None, f"{type(e).__name__}: {e}"[:300]
    token = text.strip().split()[0].lower() if text.strip() else ""
    return (token, None) if re.fullmatch(r"[0-9a-f]{64}", token) else (None, f"unexpected answer {text[:80]!r}")


def check_binance_checksums(raw_dir: Path | None = None, network: bool = True, workers: int = HASH_WORKERS) -> tuple:
    """§7.3: every local Binance zip against the ``.CHECKSUM`` (SHA-256) Binance serves for it now, downloaded again
    (the local copy of a CHECKSUM could have been rewritten together with its zip), and against the local copy.
    Without network only the local comparison runs, and a clean result is SKIPPED, not PASS."""
    from volrisk.data import binance

    root = binance.RAW_DIR if raw_dir is None else Path(raw_dir)
    zips = sorted(root.rglob("*.zip")) if root.exists() else []
    if not zips:
        raise SkipCheck(f"no Binance zips under {_rel(root)}")
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        shas = list(ex.map(_sha256_file, zips))
        remote = list(ex.map(_remote_checksum, zips)) if network else [(None, "network disabled")] * len(zips)
    bad_local, missing, bad_remote, copy_differs, errors = [], [], [], [], []
    for z, sha, (rsha, err) in zip(zips, shas, remote, strict=True):
        cs = binance.checksum_path(z)
        local = binance.read_checksum(cs) if cs.exists() else None
        if local is None:
            missing.append(_rel(z))
        elif local != sha:
            bad_local.append(_rel(z))
        if network and err is not None:
            errors.append({"zip": _rel(z), "error": err})
        elif rsha is not None:
            if rsha != sha:
                bad_remote.append(_rel(z))
            if local is not None and local != rsha:
                copy_differs.append(_rel(z))
    n_remote = sum(r is not None for r, _ in remote)
    ev = {
        "summary": (f"{n_remote - len(bad_remote)} of {len(zips)} zips match the CHECKSUM Binance serves now, "
                    f"{len(zips) - len(bad_local) - len(missing)} match their local copy"),
        "n_zips": len(zips),
        "n_monthly": sum("monthly" in z.parts for z in zips),
        "n_daily": sum("daily" in z.parts for z in zips),
        "megabytes": round(sum(z.stat().st_size for z in zips) / 1e6, 1),
        "n_checksums_downloaded": n_remote,
        "n_match_binance": n_remote - len(bad_remote),
        "n_mismatch_binance": len(bad_remote),
        "n_match_local_copy": len(zips) - len(bad_local) - len(missing),
        "n_mismatch_local_copy": len(bad_local),
        "n_without_local_checksum": len(missing),
        "n_local_copy_differs_from_binance": len(copy_differs),
        "n_download_errors": len(errors),
        "mismatch_binance": bad_remote[:20],
        "mismatch_local_copy": bad_local[:20],
        "without_local_checksum": missing[:20],
        "local_copy_differs_from_binance": copy_differs[:20],
        "errors": errors[:10],
    }
    if bad_remote or bad_local or missing or copy_differs:
        return "FAIL", ev, (f"{len(bad_remote)} zips differ from Binance's CHECKSUM, {len(bad_local)} from the local "
                            f"copy, {len(missing)} without CHECKSUM, {len(copy_differs)} local CHECKSUM copies differ "
                            "from Binance's")
    if not network:
        raise SkipCheck("network disabled: the zips were compared with the local CHECKSUM copies only "
                        f"({ev['n_match_local_copy']} of {len(zips)} match)", ev)
    if errors:
        raise SkipCheck(f"{len(errors)} of {len(zips)} CHECKSUM files could not be downloaded from Binance "
                        f"({errors[0]['error']})", ev)
    return "PASS", ev, None


def _local_raw(path_str: str) -> Path:
    """Local file of a manifest row (absolute path recorded at download; re-rooted if the project moved)."""
    p = Path(path_str)
    if p.exists():
        return p
    s = str(path_str).replace("\\", "/")
    i = s.find("data/raw/")
    return C.RAW / s[i + len("data/raw/"):] if i >= 0 else p


def _refetch(row: dict) -> dict:
    local = _local_raw(row["local_path"])
    rec = {"source": row["source"], "url": row["url"], "bytes": int(row["bytes"]),
           "sha256_manifest": row["sha256"], "sha256_local": _sha256_file(local) if local.exists() else None}
    try:
        body = _http_get(row["url"])
    except NETWORK_ERRORS as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:300]
        return rec
    rec["sha256_remote"] = hashlib.sha256(body).hexdigest()
    rec["identical"] = rec["sha256_local"] is not None and rec["sha256_remote"] == rec["sha256_local"]
    rec["local_matches_manifest"] = rec["sha256_local"] == rec["sha256_manifest"]
    return rec


def resample_raw(sample: int = 30, seed: int = SEED, manifest: pd.DataFrame | None = None, workers: int = 4,
                 seed_info: dict | None = None) -> tuple:
    """§7.3: a random sample of raw files (half Binance zips, half Dukascopy ``.bi5``) re-downloaded from the URLs
    recorded in ``data/raw/manifest.parquet`` and compared byte for byte (SHA-256) with the local copy.

    ``seed`` should be drawn at run time (``draw_seed``), so the sample cannot be known in advance; ``seed_info``
    records where it came from. A file that cannot be downloaded is replaced by the next file of the same seeded order
    (at most twice the quota per source is tried), so a flaky connection does not shrink the sample silently.
    """
    from volrisk.data import http

    if manifest is None:
        if not http.MANIFEST.exists():
            raise SkipCheck("no data/raw/manifest.parquet")
        manifest = pd.read_parquet(http.MANIFEST)
    m = manifest
    kind = ((m["source"] == "binance") & m["url"].str.endswith(".zip")) | (
        (m["source"] == "dukascopy") & m["url"].str.endswith(".bi5"))
    m = m[(m["status"] == "ok") & (m["bytes"] > 0) & kind].sort_values("url", kind="stable").reset_index(drop=True)
    groups = {s: g.reset_index(drop=True) for s, g in m.groupby("source", sort=True)}
    if not groups or sample <= 0:
        raise SkipCheck("the manifest lists no downloaded Binance zip or Dukascopy file")
    rng = np.random.default_rng(seed)
    order = {s: rng.permutation(len(g)) for s, g in groups.items()}
    names = list(groups)
    quota = {s: min(len(groups[s]), sample // len(names) + (1 if i < sample % len(names) else 0))
             for i, s in enumerate(names)}
    nxt = dict.fromkeys(names, 0)
    got = dict.fromkeys(names, 0)
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        while True:
            batch = []
            for s in names:
                take = min(quota[s] - got[s], 2 * quota[s] - nxt[s], len(groups[s]) - nxt[s])
                for _ in range(max(take, 0)):
                    batch.append(groups[s].iloc[int(order[s][nxt[s]])].to_dict())
                    nxt[s] += 1
            if not batch:
                break
            for rec in ex.map(_refetch, batch):
                results.append(rec)
                if "error" not in rec:
                    got[rec["source"]] += 1
    compared = [r for r in results if "error" not in r]
    differ = [r for r in compared if not r["identical"]]
    errors = [r for r in results if "error" in r]
    target = sum(quota.values())
    ev = {
        "summary": f"{len(compared) - len(differ)} of {len(compared)} re-downloaded files identical "
                   f"(seed {seed}, {len(errors)} download errors)",
        "seed": seed,
        "seed_source": seed_info or {"source": "given"},
        "requested": target,
        "population": {s: len(g) for s, g in groups.items()},
        "n_compared": len(compared),
        "n_identical": len(compared) - len(differ),
        "n_different": len(differ),
        "n_download_errors": len(errors),
        "by_source": {s: got[s] for s in names},
        "files": [{"source": r["source"], "url": r["url"], "bytes": r["bytes"], "sha256": r["sha256_remote"],
                   "identical": r["identical"]} for r in compared],
        "different": [{"url": r["url"], "local": r["sha256_local"], "remote": r["sha256_remote"]} for r in differ],
        "errors": [{"url": r["url"], "error": r["error"]} for r in errors[:10]],
    }
    if differ:
        return "FAIL", ev, f"{len(differ)} re-downloaded file(s) differ from the local copy"
    if len(compared) < target:
        raise SkipCheck(f"only {len(compared)} of {target} files could be re-downloaded "
                        f"({errors[0]['error'] if errors else 'no candidates left'})", ev)
    return "PASS", ev, None


def _hash_row(item: tuple[str, str, str]) -> tuple[str, str, str | None]:
    url, path, recorded = item
    p = _local_raw(path) if path else None
    return url, recorded, (_sha256_file(p) if p is not None and p.is_file() else None)


def check_raw_manifest(manifest: pd.DataFrame | None = None, raw_root: Path | None = None,
                       workers: int = HASH_WORKERS) -> tuple:
    """§7.3: every raw file the download manifest records as fetched still has the SHA-256 recorded when it was
    downloaded, and every raw file on disk (Binance zips and CHECKSUMs, Dukascopy day files) is recorded. The manifest
    is a local file, so this shows the raw store is consistent with its own record; 3a/3b compare with the sources."""
    from volrisk.data import http

    if manifest is None:
        if not http.MANIFEST.exists():
            raise SkipCheck("no data/raw/manifest.parquet")
        manifest = pd.read_parquet(http.MANIFEST)
    root = C.RAW if raw_root is None else Path(raw_root)
    m = manifest[(manifest["status"] == "ok") & (manifest["bytes"] > 0)]
    items = list(zip(m["url"], m["local_path"], m["sha256"], strict=True))
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        hashed = list(ex.map(_hash_row, items))
    missing = [u for u, _, sha in hashed if sha is None]
    differ = [u for u, rec, sha in hashed if sha is not None and sha != rec]
    recorded = {_local_raw(p).resolve() for p in m["local_path"] if p}
    on_disk = [p for src in ("binance", "dukascopy") if (root / src).exists()
               for p in (root / src).rglob("*") if p.is_file() and RAW_FILE_RE.search(p.name)]
    unrecorded = sorted(_rel(p) for p in on_disk if p.resolve() not in recorded)
    by_source = m.groupby("source").size().to_dict()
    ev = {"summary": f"{len(hashed) - len(missing) - len(differ)} of {len(hashed)} recorded raw files unchanged since "
                     f"their download, {len(unrecorded)} unrecorded raw files on disk",
          "n_recorded": len(hashed), "by_source": by_source, "n_unchanged": len(hashed) - len(missing) - len(differ),
          "n_changed": len(differ), "n_missing": len(missing), "n_files_on_disk": len(on_disk),
          "n_unrecorded": len(unrecorded), "changed": differ[:20], "missing": missing[:20],
          "unrecorded": unrecorded[:20]}
    ok = not (missing or differ or unrecorded)
    return _status(ok), ev, None if ok else (f"{len(differ)} changed, {len(missing)} missing, {len(unrecorded)} "
                                             "unrecorded raw files")


# --------------------------------------------------------------------------------------------- 4 independent sources
def load_history() -> tuple[pd.DataFrame, str]:
    """Gold rows of every asset: sealed dev + holdout (read after the logged opening) plus live-rebuild sessions
    after the frozen data end, if any. ``session_date`` as datetime64[ns]."""
    context.require_opened_holdout()
    with context.history_access():
        d = io.load_daily(include_holdout=True)
    d = d.assign(session_date=pd.to_datetime(d["session_date"]).astype("datetime64[ns]"))
    src = "sealed gold, dev + holdout"
    if paths.LIVE_DAILY_HOLDOUT.exists():
        live = pd.read_parquet(paths.LIVE_DAILY_HOLDOUT)
        live["session_date"] = pd.to_datetime(live["session_date"]).astype("datetime64[ns]")
        live = live[live["session_date"] > pd.Timestamp(context.frozen_data_end())]
        if len(live):
            d = pd.concat([d, live[[c for c in d.columns if c in live.columns]]], ignore_index=True)
            src += f" + live rebuild to {live['session_date'].max().date()}"
    return d.sort_values(["asset", "session_date"], kind="stable").reset_index(drop=True), src


def session_returns(daily: pd.DataFrame, asset: str) -> pd.DataFrame:
    """``date, r, prev``: close-to-close % log return of every session and the session it starts from."""
    g = daily[daily["asset"] == asset].sort_values("session_date")
    sd = pd.to_datetime(g["session_date"]).astype("datetime64[ns]")
    return pd.DataFrame({"date": sd.to_numpy(), "r": g["r_cc"].to_numpy(dtype=float),
                         "prev": sd.shift(1).to_numpy()})


def _return_stats(a: pd.Series, b: pd.Series, dates: pd.Series) -> dict:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = len(a)
    if n == 0:
        return {"n": 0, "corr": None}
    diff = 100.0 * np.abs(a - b)  # % -> bp
    i = int(np.argmax(diff))
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    return {
        "n": n,
        "corr": float(np.corrcoef(a, b)[0, 1]) if n >= 3 else None,
        "mean_abs_diff_bp": float(diff.mean()),
        "median_abs_diff_bp": float(np.median(diff)),
        "p99_abs_diff_bp": float(np.quantile(diff, 0.99)),
        "max_abs_diff_bp": float(diff[i]),
        "max_diff_date": d.iloc[i].date().isoformat(),
        f"days_over_{int(DIFF_TOL_BP)}bp": int((diff > DIFF_TOL_BP).sum()),
        "first": d.min().date().isoformat(),
        "last": d.max().date().isoformat(),
    }


def compare_returns(ours: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Our returns (``date, r, prev``) vs a reference level series (``date, level``).

    The reference return on date ``d`` runs from its previous observation; a day is compared only when both
    returns start on the same date (a missing day on either side would make them span different periods).
    """
    r = ref.dropna().sort_values("date").drop_duplicates("date", keep="last")
    r = r[r["level"] > 0]
    r = r.assign(r_ref=100.0 * np.log(r["level"].astype(float)).diff(), prev_ref=r["date"].shift(1))
    m = ours.merge(r, on="date", how="inner")
    m = m[(m["prev"] == m["prev_ref"]) & np.isfinite(m["r"]) & np.isfinite(m["r_ref"])]
    return _return_stats(m["r"], m["r_ref"], m["date"])


def compare_levels(ours: pd.DataFrame, ref: pd.DataFrame) -> dict:
    """Two level series (``date, level``) observed at the same time of day: returns over consecutive common dates
    plus the level gap (bp)."""
    m = ours.merge(ref, on="date", suffixes=("", "_ref")).sort_values("date")
    m = m[(m["level"] > 0) & (m["level_ref"] > 0)].reset_index(drop=True)
    ra = 100.0 * np.log(m["level"].astype(float)).diff()
    rb = 100.0 * np.log(m["level_ref"].astype(float)).diff()
    ok = ra.notna() & rb.notna()
    out = _return_stats(ra[ok], rb[ok], m["date"][ok])
    gap = 1e4 * np.abs(np.log(m["level"].astype(float) / m["level_ref"].astype(float)))
    out.update(n_levels=len(m), level_mean_abs_diff_bp=float(gap.mean()) if len(m) else None,
               level_median_abs_diff_bp=float(gap.median()) if len(m) else None)
    return out


def fetch_coinbase_closes(product: str, start: date, end: date) -> pd.DataFrame:
    """Daily closes (UTC days) of a Coinbase product, paginated in windows of at most 300 candles.

    A candle is ``[time, low, high, open, close, volume]`` with ``time`` = UTC day start.
    """
    url = COINBASE_URL.format(product=product)
    closes: dict[date, float] = {}
    n_req = 0
    s = start
    while s <= end:
        e = min(s + timedelta(days=COINBASE_MAX - 1), end)
        params = {"granularity": 86400, "start": f"{s.isoformat()}T00:00:00Z", "end": f"{e.isoformat()}T00:00:00Z"}
        body = _http_get(url, params=params)
        n_req += 1
        try:
            data = json.loads(body)
        except ValueError:
            raise requests.RequestException(f"Coinbase answered with non-JSON: {body[:200]!r}") from None
        if not isinstance(data, list):
            raise requests.RequestException(f"unexpected Coinbase answer: {str(data)[:200]}")
        for c in data:
            d = datetime.fromtimestamp(int(c[0]), UTC).date()
            if s <= d <= e:
                closes[d] = float(c[4])
        s = e + timedelta(days=1)
        _sleep(0.15)  # public endpoint: stay well below its rate limit
    days = sorted(closes)
    out = pd.DataFrame({"date": pd.to_datetime(days).astype("datetime64[ns]"), "level": [closes[d] for d in days]})
    out.attrs["requests"] = n_req
    return out


def parse_ecb_csv(content: bytes) -> pd.DataFrame:
    """ECB SDMX ``csvdata`` -> ``date, level`` (USD per EUR = EUR/USD)."""
    raw = pd.read_csv(BytesIO(content), usecols=["TIME_PERIOD", "OBS_VALUE"])
    out = pd.DataFrame({"date": pd.to_datetime(raw["TIME_PERIOD"]).astype("datetime64[ns]"),
                        "level": pd.to_numeric(raw["OBS_VALUE"], errors="coerce")})
    return out.dropna().sort_values("date").reset_index(drop=True)


def fetch_ecb_eurusd() -> pd.DataFrame:
    body = _http_get(ECB_URL, params={"format": "csvdata"}, timeout=90.0)
    try:
        return parse_ecb_csv(body)
    except (ValueError, KeyError) as e:  # the source changed its format: not evidence about our data
        raise requests.RequestException(f"unexpected ECB answer ({e}): {body[:200]!r}") from None


def fix_times_utc(days: list[date]) -> list[datetime]:
    """ECB fixing time 14:15 Frankfurt local time (CET/CEST) of each day, in UTC."""
    tz = ZoneInfo(ECB_TZ)
    return [datetime.combine(d, ECB_FIX, tzinfo=tz).astimezone(UTC) for d in days]


def dukascopy_fix_prices(days: list[date], bronze_dir: Path) -> pd.DataFrame:
    """EUR/USD Dukascopy mid at the ECB fixing time: close of the last real minute ending at most
    ``FIX_TOLERANCE`` before 14:15 Frankfurt time. Returns ``date, level``."""
    if not days:
        return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"), "level": pd.Series(dtype=float)})
    years = {d.year for d in days}
    files = [p for p in sorted((bronze_dir / "asset=EURUSD").glob("year=*/part.parquet"))
             if int(p.parent.name.split("=")[1]) in years]
    if not files:
        raise SkipCheck(f"no EUR/USD bronze minutes under {_rel(bronze_dir)}")
    ts_type = pl.Datetime("us", "UTC")
    # 14:15 Frankfurt is 12:15 (summer) or 13:15 (winter) UTC: only minutes opening 11:00-13:59 UTC are needed
    minutes = (
        pl.concat([pl.scan_parquet(f).select("ts", "close", "is_real") for f in files])
        .filter(pl.col("is_real") & pl.col("ts").dt.hour().is_between(11, 13))
        .select(end=(pl.col("ts").cast(ts_type) + pl.duration(minutes=1)), level=pl.col("close"))
        .collect()
        .sort("end")
    )
    fixes = pl.DataFrame({"date": days, "fix": fix_times_utc(days)},
                         schema={"date": pl.Date, "fix": ts_type}).sort("fix")
    j = fixes.join_asof(minutes, left_on="fix", right_on="end", strategy="backward", tolerance=FIX_TOLERANCE)
    j = j.drop_nulls("level").to_pandas()
    return pd.DataFrame({"date": pd.to_datetime(j["date"]).astype("datetime64[ns]"), "level": j["level"]})


def _num_txt(x, digits: int) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{x:.{digits}f}"


def _corr_summary(st: dict) -> str:
    if not st.get("n") or st.get("corr") is None:
        return f"{st.get('n', 0)} matched days, no correlation"
    return f"corr {st['corr']:.4f} over {st['n']} days, mean |diff| {st['mean_abs_diff_bp']:.1f} bp"


def _corr_verdict(asset: str, st: dict) -> tuple[str, str | None]:
    if st["n"] < MIN_DAYS or st["corr"] is None:
        return "FAIL", f"only {st['n']} matched days (need {MIN_DAYS})"
    if st["corr"] < CORR_MIN[asset]:
        return "FAIL", f"correlation {st['corr']:.4f} < {CORR_MIN[asset]}"
    return "PASS", None


def check_crypto(asset: str, daily: pd.DataFrame, src: str, network: bool = True) -> tuple:
    """§7.4: BTC/ETH close-to-close returns (Binance, USDT, UTC days) vs Coinbase USD daily candles."""
    if not network:
        raise SkipCheck("network disabled")
    ours = session_returns(daily, asset)
    if ours.empty:
        raise SkipCheck(f"no {asset} sessions")
    first = ours["date"].min().date()
    end = min(ours["date"].max().date(), datetime.now(UTC).date() - timedelta(days=1))
    product = COINBASE_PRODUCTS[asset]
    ref = fetch_coinbase_closes(product, first - timedelta(days=1), end)
    st = compare_returns(ours, ref)
    status, reason = _corr_verdict(asset, st)
    ev = {"summary": _corr_summary(st),
          "ours": f"{asset} gold r_cc ({src}; Binance {asset}USDT, close of the UTC day)",
          "reference": f"Coinbase {product} daily candles (close of the UTC day)",
          "threshold_corr": CORR_MIN[asset], **st, "reference_days": len(ref),
          "reference_requests": ref.attrs.get("requests")}
    return status, ev, reason


def check_spx(daily: pd.DataFrame, src: str) -> tuple:
    """§7.4: S&P 500 CFD close-to-close returns vs FRED SP500 (official closes; newest local snapshot)."""
    from volrisk import quality

    f = quality.newest_fred()
    if f is None:
        raise SkipCheck("no FRED SP500 snapshot under data/raw/implied/")
    fred = quality.read_fred_sp500(f).to_pandas()
    ref = pd.DataFrame({"date": pd.to_datetime(fred["date"]).astype("datetime64[ns]"), "level": fred["close"]})
    st = compare_returns(session_returns(daily, "SPX"), ref)
    status, reason = _corr_verdict("SPX", st)
    ev = {"summary": _corr_summary(st),
          "ours": f"SPX gold r_cc ({src}; Dukascopy USA500 CFD mid at the 16:00 New York close)",
          "reference": f"FRED SP500 official closes ({_rel(f)})", "threshold_corr": CORR_MIN["SPX"], **st}
    return status, ev, reason


def check_eurusd(daily: pd.DataFrame, src: str, network: bool = True, bronze_dir: Path | None = None) -> tuple:
    """§7.4: EUR/USD vs the ECB reference rate. The verdict compares like with like: our Dukascopy mid at the
    ECB fixing time (14:15 Frankfurt) vs the fix. Our 17:00 New York close-to-close returns vs the fix are reported
    alongside; they are ~9 hours apart, so their correlation is lower by construction."""
    if not network:
        raise SkipCheck("network disabled")
    ecb = fetch_ecb_eurusd()
    ours = session_returns(daily, "EURUSD")
    if ours.empty:
        raise SkipCheck("no EURUSD sessions")
    lo, hi = ours["date"].min() - pd.Timedelta(days=7), ours["date"].max()
    ecb = ecb[(ecb["date"] >= lo) & (ecb["date"] <= hi)].reset_index(drop=True)
    context.require_opened_holdout()  # the bronze minutes cover the holdout period too
    if bronze_dir is None:
        bronze_dir = C.BRONZE / "minute"
        if not (bronze_dir / "asset=EURUSD").exists():
            bronze_dir = paths.LIVE_BRONZE
    fix = dukascopy_fix_prices([d.date() for d in ecb["date"]], bronze_dir)
    matched = compare_levels(fix, ecb)
    close = compare_returns(ours, ecb)
    status, reason = _corr_verdict("EURUSD", matched)
    ev = {"summary": f"at the fixing time: {_corr_summary(matched)}, median level gap "
                     f"{_num_txt(matched.get('level_median_abs_diff_bp'), 2)} bp; 17:00 NY close vs fix "
                     f"(timing differs): corr {_num_txt(close.get('corr'), 3)}",
          "reference": "ECB euro reference rate USD/EUR (14:15 CET), data-api.ecb.europa.eu",
          "threshold_corr": CORR_MIN["EURUSD"],
          "at_fixing_time": {"ours": f"Dukascopy EUR/USD mid at 14:15 Frankfurt ({_rel(bronze_dir)})", **matched},
          "session_close_vs_fix": {"ours": f"EURUSD gold r_cc ({src}; 17:00 New York close)",
                                   "note": "informational: the two prices are taken ~9 hours apart", **close}}
    return status, ev, reason


# --------------------------------------------------------------------------------------------- 5 reproducibility
def _live_state() -> dict:
    try:
        return json.loads(paths.LIVE_STATE.read_text(encoding="utf-8")) if paths.LIVE_STATE.exists() else {}
    except (OSError, ValueError):
        return {}


def check_live_gold() -> tuple:
    """§7.5: the live tables last built by ``update`` (data/live/) equal the sealed gold
    (``update.check_against_sealed``; raises ``LiveDataMismatch``). A file comparison: the rebuild itself is re-done
    from the raw files only by ``--full`` (5g)."""
    if not (paths.LIVE_DAILY_DEV.exists() and paths.LIVE_DAILY_HOLDOUT.exists()):
        raise SkipCheck("no live rebuild under data/live/ yet (run `python -m volrisk_live update`)")
    context.require_opened_holdout()  # the compared files hold holdout-period rows
    res = _live_module("update").check_against_sealed()
    ok = _ok_of(res)
    state = _live_state()
    files = [file_record(p) for p in (paths.LIVE_DAILY_DEV, paths.LIVE_DAILY_HOLDOUT, paths.LIVE_IMPLIED)]
    ev = {"summary": "live dev {d} rows and holdout {h} rows identical to the sealed gold (live files built by "
                     "update at {t})".format(d=res.get("dev", {}).get("rows_compared", "?"),
                                             h=res.get("holdout", {}).get("rows_compared", "?"),
                                             t=state.get("started_utc", "?"))
          if isinstance(res, dict) else str(res),
          "built_utc": state.get("started_utc"), "finished_utc": state.get("finished_utc"),
          "live_end": state.get("end"), "live_files": files,
          "sealed_files": [file_record(p) for p in (io.DAILY_DEV, io.DAILY_HOLDOUT, io.IMPLIED)],
          "note": "compares the files under data/live/ with the sealed ones; a deterministic rebuild of the dev table "
                  "gives the same bytes as the sealed file, so the files alone cannot show it was rebuilt rather than "
                  "copied - verify --full (5g) rebuilds the gold tables from the raw files",
          "result": res}
    return _status(ok is not False), ev, None if ok is not False else "live rebuild differs from the sealed data"


def check_live_forecasts() -> tuple:
    """§7.5: the live walk-forward written by the last forecast run reproduces every stored dev/holdout-run forecast
    (``forecast.check_reproduces_holdout``). A file comparison; ``--full`` (5f) re-runs the walk-forward."""
    f = paths.LIVE_RESULTS / "forecasts.parquet"
    if not f.exists():
        raise SkipCheck("no live forecasts under data/live/forecasts/ yet (run `python -m volrisk_live forecast`)")
    mod = _live_module("forecast")
    fc = pd.read_parquet(f)
    n = int(mod.check_reproduces_holdout(fc))  # raises ReproductionError on any difference
    if n == 0:
        raise SkipCheck("no sealed forecast overlaps the live run")
    ev: dict = {"file": file_record(f), "n_sealed_forecasts_reproduced": n, "rtol": REPRO_RTOL, "atol": 0.0}
    if hasattr(mod, "reproduction_check"):
        st = mod.reproduction_check(fc)
        ev.update(max_abs_diff=st.get("max_abs_diff"), max_rel_diff=st.get("max_rel_diff"),
                  per_file=st.get("files"), new_live_forecasts=st.get("new_rows"))
    risk = paths.LIVE_RESULTS / "risk.parquet"
    if risk.exists() and hasattr(mod, "risk_reproduction_check"):
        ev["risk"] = mod.risk_reproduction_check(pd.read_parquet(risk))
        ev["risk_file"] = file_record(risk)
    ev["summary"] = (f"{n} stored forecasts equal in the live walk-forward file (rtol {REPRO_RTOL:g}, atol 0), "
                     f"written {_iso(datetime.fromtimestamp(f.stat().st_mtime, UTC))}")
    return "PASS", ev, None


def compare_tables(tables: dict[str, pd.DataFrame], folder: Path, prefix: str = "eval",
                   rtol: float | None = None) -> list[dict]:
    """Recomputed tables vs ``{folder}/{prefix}_{name}.parquet``: exact equality (NaN = NaN, dtypes ignored), or
    floats to ``rtol`` with atol 0 when given (tables from re-run forecasts)."""
    rows = []
    kw = {"check_exact": True} if rtol is None else {"check_exact": False, "rtol": rtol, "atol": 0.0}
    for k, v in tables.items():
        p = folder / f"{prefix}_{k}.parquet"
        if not p.exists():
            empty = v.shape[1] == 0  # pipeline._write_tables never writes a zero-column table
            rows.append({"table": k, "rows": len(v), "equal": empty,
                         "note": "empty table, never written" if empty else "stored file missing"})
            continue
        stored = pd.read_parquet(p)
        try:
            pd.testing.assert_frame_equal(v.reset_index(drop=True), stored.reset_index(drop=True),
                                          check_dtype=False, **kw)
            eq, note = True, "identical" if rtol is None else f"equal (rtol {rtol:g})"
        except AssertionError as e:
            eq, note = False, " ".join(str(e).split())[:300]
        rows.append({"table": k, "rows": len(v), "equal": eq, "note": note})
    for p in sorted(folder.glob(f"{prefix}_*.parquet")):
        name = p.stem[len(prefix) + 1:]
        if name not in tables:
            rows.append({"table": name, "rows": None, "equal": False, "note": "stored table was not recomputed"})
    return rows


def pinned_result_files(ledger_path: Path | None = None, runs_dir: Path | None = None) -> dict[str, dict]:
    """SHA-256 of the stored result files as first recorded by a ledger payload (``checks.reproduction.files``,
    ``checks.risk_reproduction.files``; the payload is OpenTimestamps-stamped): ``{file: {sha256, run_id}}``."""
    ledger_path = paths.LEDGER if ledger_path is None else Path(ledger_path)
    runs_dir = paths.RUNS if runs_dir is None else Path(runs_dir)
    out: dict[str, dict] = {}
    if not ledger_path.exists():
        return out
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        try:
            run_id = json.loads(line)["run_id"]
            payload = json.loads((runs_dir / f"{run_id}.json").read_text(encoding="utf-8"))
        except (ValueError, KeyError, TypeError, OSError):
            continue
        checks = payload.get("checks") or {}
        for part in ("reproduction", "risk_reproduction"):
            for name, meta in ((checks.get(part) or {}).get("files") or {}).items():
                if name not in out and isinstance(meta, dict) and meta.get("sha256"):
                    out[name] = {"sha256": meta["sha256"], "run_id": run_id}
    return out


def check_eval_tables(mode: str) -> tuple:
    """§7.5: every evaluation table recomputed in memory from the stored forecasts/targets equals the stored one.
    Internal consistency of the stored files; the holdout result files are not in SEALED.json, so their SHA-256 is
    compared with the one the first ledger payload recorded, and ``--full`` (5f) re-derives them."""
    from volrisk.evaluation.suite import evaluate

    if mode == "holdout":
        context.require_opened_holdout()
        folder = holdout.RESULTS_HOLDOUT
    else:
        folder = C.RESULTS
    fc_path, tg_path = folder / "forecasts.parquet", folder / "targets.parquet"
    if not (fc_path.exists() and tg_path.exists()):
        raise SkipCheck(f"no stored forecasts/targets under {_rel(folder)}")
    files = [file_record(folder / n) for n in ("forecasts.parquet", "targets.parquet", "risk.parquet")
             if (folder / n).exists()]
    pinned = pinned_result_files() if mode == "holdout" else {}
    for rec in files:
        pin = pinned.get(rec["file"])
        rec["pinned_by_ledger_run"] = pin["run_id"] if pin else None
        rec["unchanged_since_pinned"] = (pin["sha256"] == rec["sha256"]) if pin else None
    changed = [r["file"] for r in files if r["unchanged_since_pinned"] is False]
    tables = evaluate(pd.read_parquet(fc_path), pd.read_parquet(tg_path), mode=mode)
    rows = compare_tables(tables, folder)
    lb = tables["leaderboard"]
    combo = lb[lb["model"] == "COMBO"][["asset", "horizon", "qlike_ratio", "n"]].copy()
    combo["qlike_ratio"] = combo["qlike_ratio"].round(3)
    n_eq = sum(r["equal"] for r in rows)
    if mode == "holdout":
        pins = sorted({r["pinned_by_ledger_run"] for r in files if r["pinned_by_ledger_run"]})
        anchor = (f"result files changed since ledger run {pins[0]} pinned them: {changed}" if changed else
                  f"result files unchanged since ledger run {pins[0]} pinned them" if pins else
                  "result files not pinned yet (no ledger payload)")
        note = ("internal consistency of the stored holdout files; they were written at the opening, after the seal, "
                "so SEALED.json does not hash them - the first ledger payload pins their SHA-256 and verify --full "
                "(5f) re-derives them from the sealed data")
    else:
        anchor = "stored dev forecasts are sealed" if any(r["sealed_as"] for r in files) else "not sealed"
        note = "internal consistency of the stored dev files (data/results/forecasts.parquet is hashed in SEALED.json)"
    ev = {"summary": f"{n_eq} of {len(rows)} tables identical; {anchor}", "folder": _rel(folder), "files": files,
          "note": note, "tables": rows, "combo_qlike_ratio_vs_har": combo}
    ok = n_eq == len(rows) and len(rows) > 0 and not changed
    reason = None if ok else (anchor if changed else "recomputed tables differ from the stored ones")
    return _status(ok), ev, reason


def check_full_dev(workers: int = 10) -> tuple:
    """§7.5 ``--full``: the complete dev walk-forward in memory reproduces data/results/forecasts.parquet."""
    from volrisk import pipeline

    fc, _ = pipeline.compute_forecasts(include_holdout=False, workers=workers)
    n = pipeline.assert_reproduces_dev(fc)  # raises RuntimeError on any difference
    return "PASS", {"summary": f"{n} dev forecasts reproduced", "n_compared": n, "rtol": pipeline.REPRO_RTOL,
                    "atol": 0.0, "workers": workers}, None


def compare_keyed(stored: pd.DataFrame, rerun: pd.DataFrame, keys: list[str], values: list[str],
                  exact: tuple[str, ...] = (), rtol: float = REPRO_RTOL) -> dict:
    """Two tables on ``keys`` in both directions: every row on both sides, ``exact`` columns equal, the other
    ``values`` to ``rtol`` (atol 0, NaN = NaN). Returns counts, the largest differences and a few examples."""
    def norm(df: pd.DataFrame) -> pd.DataFrame:
        d = df[[*keys, *values]].copy()
        for k in keys:
            if pd.api.types.is_datetime64_any_dtype(d[k]) or k in ("origin", "date", "session_date"):
                d[k] = pd.to_datetime(d[k]).astype("datetime64[ns]")
        return d

    a, b = norm(stored), norm(rerun)
    dup = int(a.duplicated(keys).sum() + b.duplicated(keys).sum())
    m = a.merge(b, on=keys, how="outer", suffixes=("_stored", "_rerun"), indicator=True)
    both = (m["_merge"] == "both").to_numpy()
    bad = ~both
    max_abs = max_rel = 0.0
    differ: dict[str, int] = {}
    for v in values:
        x, y = m[f"{v}_stored"], m[f"{v}_rerun"]
        if v in exact or not (pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y)):
            same = (x.astype(str) == y.astype(str)).to_numpy() | (x.isna() & y.isna()).to_numpy()
        else:
            xa, ya = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
            same = np.isclose(xa, ya, rtol=rtol, atol=0.0) | (np.isnan(xa) & np.isnan(ya))
            fin = both & np.isfinite(xa) & np.isfinite(ya)
            if fin.any():
                d = np.abs(xa[fin] - ya[fin])
                max_abs = max(max_abs, float(d.max()))
                max_rel = max(max_rel, float((d / np.maximum(np.abs(xa[fin]), np.finfo(float).tiny)).max()))
        differ[v] = int((both & ~same).sum())
        bad |= both & ~same
    return {"rows_stored": len(a), "rows_rerun": len(b), "only_stored": int((m["_merge"] == "left_only").sum()),
            "only_rerun": int((m["_merge"] == "right_only").sum()), "differ": differ, "duplicates": dup,
            "max_abs_diff": max_abs, "max_rel_diff": max_rel, "equal": not bad.any() and dup == 0,
            "examples": m.loc[bad].drop(columns="_merge").head(5)}


def check_full_holdout(workers: int = 10) -> tuple:
    """§7.5 ``--full``: the one-time holdout run re-done in memory from the sealed data with the frozen code
    (``pipeline.compute_forecasts(include_holdout=True)`` inside ``context.history_access``): forecasts, targets and
    VaR/ES equal the stored data/results/holdout files in both directions, its dev rows equal the sealed dev
    forecasts, and the holdout evaluation and risk tables recomputed from the re-run equal the stored ones."""
    from volrisk import pipeline
    from volrisk.evaluation.suite import evaluate
    from volrisk.risk.suite import build_risk, evaluate_risk, time_in_green

    context.require_opened_holdout()
    folder = holdout.RESULTS_HOLDOUT
    if not (folder / "forecasts.parquet").exists():
        raise SkipCheck(f"no stored holdout results under {_rel(folder)}")
    files = [file_record(folder / n) for n in ("forecasts.parquet", "targets.parquet", "risk.parquet")]
    with context.history_access():  # parent process only; spawned workers get ready-made frames
        fc, tg = pipeline.compute_forecasts(include_holdout=True, workers=workers)
        daily_all = io.load_daily(include_holdout=True)
    n_dev = pipeline.assert_reproduces_dev(fc)  # raises RuntimeError on any difference
    fc_cmp = compare_keyed(pd.read_parquet(folder / "forecasts.parquet"), fc, FC_KEYS, ["F", "n_t", "split"],
                           exact=("n_t", "split"))
    stored_tg = pd.read_parquet(folder / "targets.parquet")
    tg_keys = ["asset", "horizon", "origin"]
    tg_cmp = compare_keyed(stored_tg, tg, tg_keys, [c for c in stored_tg.columns if c not in tg_keys], rtol=0.0)
    star = dict(holdout.frozen()["har_star"])
    risk = build_risk(daily_all, fc, star)
    stored_risk = pd.read_parquet(folder / "risk.parquet")
    risk_cmp = compare_keyed(stored_risk, risk, RISK_KEYS, [c for c in stored_risk.columns if c not in RISK_KEYS])
    eval_rows = compare_tables(evaluate(fc, tg, mode="holdout"), folder, rtol=REPRO_RTOL)
    rt = evaluate_risk(risk, daily_all, fc, mode="holdout")
    rt["time_in_zone"] = time_in_green(rt["rolling_zones"], "holdout")
    risk_rows = compare_tables(rt, folder, prefix="risk", rtol=REPRO_RTOL)
    n_tab = sum(r["equal"] for r in eval_rows + risk_rows)
    ok = fc_cmp["equal"] and tg_cmp["equal"] and risk_cmp["equal"] and n_tab == len(eval_rows) + len(risk_rows)
    ev = {"summary": (f"{len(fc)} forecasts re-run: {'equal' if fc_cmp['equal'] else 'DIFFERENT'} to the stored "
                      f"{fc_cmp['rows_stored']} (both directions), targets {'equal' if tg_cmp['equal'] else 'DIFFER'}, "
                      f"VaR/ES {'equal' if risk_cmp['equal'] else 'DIFFER'}, {n_tab} of "
                      f"{len(eval_rows) + len(risk_rows)} tables equal; dev rows = sealed dev forecasts ({n_dev})"),
          "stored_files": files, "forecasts": fc_cmp, "targets": tg_cmp, "risk": risk_cmp,
          "n_dev_rows_equal_sealed": n_dev, "eval_tables": eval_rows, "risk_tables": risk_rows,
          "rtol": REPRO_RTOL, "atol": 0.0, "workers": workers}
    return _status(ok), ev, None if ok else "the re-run holdout walk-forward differs from the stored holdout results"


def _frames_equal(a: Path, b: Path) -> tuple[bool, str]:
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(a), pd.read_parquet(b), check_exact=True, check_dtype=True)
        return True, "identical values"
    except AssertionError as e:
        return False, " ".join(str(e).split())[:300]


def check_rebuild_gold(tmp_root: Path | None = None) -> tuple:
    """§7.5 ``--full``: the gold and implied-vol tables rebuilt from the raw files with the frozen builders
    (bronze -> 5-minute bars -> gold, frozen data end) in a temporary directory equal the sealed files."""
    from volrisk import bars, measures
    from volrisk.data import binance, dukascopy, implied

    context.require_opened_holdout()  # the rebuild reads holdout-period raw files
    tmp = Path(tempfile.mkdtemp(prefix="volrisk_verify_rebuild_", dir=tmp_root))
    try:
        t0 = time.perf_counter()
        for a in C.ASSETS:
            src = binance if C.asset(a).source == binance.SOURCE else dukascopy
            kw = {"end": C.data_end()} if src is binance else {}
            src.build_bronze(a, out_dir=tmp / "bronze" / "minute", flags_dir=tmp / "bronze" / "flags", **kw)
            bars.build_bars(a, bronze_dir=tmp / "bronze", out_dir=tmp / "silver", end=C.data_end())
        measures.build_gold(C.ASSETS, silver_dir=tmp / "silver", gold_dir=tmp / "gold", holdout_dir=tmp / "holdout")
        implied.build_implied(out_path=tmp / "implied.parquet")
        secs = round(time.perf_counter() - t0, 1)
        sealed = _sealed_hashes()
        rows = []
        for key, built, ref in (("dev_data", tmp / "gold" / "daily.parquet", io.DAILY_DEV),
                                ("holdout_data", tmp / "holdout" / "daily.parquet", io.DAILY_HOLDOUT),
                                ("implied", tmp / "implied.parquet", io.IMPLIED)):
            sha = _sha256_file(built) if built.exists() else None
            same_bytes = sha is not None and sha == sealed.get(key)
            if same_bytes:
                same, note = True, "same bytes"
            else:
                same, note = _frames_equal(built, ref) if sha else (False, "not built")
            rows.append({"table": _rel(ref), "sealed_as": key, "sha256_rebuilt": sha, "sha256_sealed": sealed.get(key),
                         "same_bytes": same_bytes, "equal": same, "note": note})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    ok = all(r["equal"] for r in rows)
    ev = {"summary": f"{sum(r['equal'] for r in rows)} of {len(rows)} tables rebuilt from data/raw equal the sealed "
                     f"ones ({sum(r['same_bytes'] for r in rows)} byte for byte) in {secs} s",
          "tables": rows, "data_end": C.data_end(), "note": "built in a temporary directory, deleted afterwards"}
    return _status(ok), ev, None if ok else "a table rebuilt from the raw files differs from the sealed one"


# --------------------------------------------------------------------------------------------- 6 negative controls
class OracleModel:
    """Cheater: ``F = n_t · ybar`` with the *realised* target mean (must be flagged by ``detect_leak``)."""

    name = "ORACLE"

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        F = np.where(predict_mask_for(a), a["n_t"].to_numpy(dtype=float) * a["ybar"].to_numpy(dtype=float), np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


class PeekModel:
    """Cheater: ``F = n_t · tv`` of the *next* session (must be flagged by ``detect_leak``)."""

    name = "PEEK"

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        tv_next = daily["tv"].shift(-1).to_numpy(dtype=float)
        F = np.where(predict_mask_for(a), a["n_t"].to_numpy(dtype=float) * tv_next, np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


class LabelLeakModel:
    """Cheater with a training-label leak, the realistic kind: a log-linear direct model (constant, ``log tv``, its
    5- and 22-session means) re-estimated every ``refit_every`` session rows on the last ``window`` purged rows,
    except that its purge is off by ``extra`` row(s): it also trains on the first row(s) whose target window has not
    ended at the refit origin. Its forecasts at an origin only use data up to that origin; the leak sits in the
    training labels, so only a cut-off within one target window after a refit origin can expose it."""

    name = "LABEL-LEAK"

    def __init__(self, refit_every: int = 90, window: int = 500, extra: int = 1):
        self.refit_every, self.window, self.extra = int(refit_every), int(window), int(extra)

    def _start(self) -> int:
        return self.window + 60  # enough rows for the 22-session mean and a 30-day window

    def refit_schedule(self, daily: pd.DataFrame, asset: str) -> dict[str, list[pd.Timestamp]]:
        dates = pd.to_datetime(daily["session_date"]).reset_index(drop=True)
        return {self.name: list(dates.iloc[list(range(self._start(), len(dates), self.refit_every))])}

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        lv = pd.Series(np.log(daily["tv"].to_numpy(dtype=float)))
        X = np.column_stack([np.ones(len(lv)), lv, lv.rolling(5).mean(), lv.rolling(22).mean()])
        with np.errstate(divide="ignore", invalid="ignore"):
            y = np.log(a["ybar"].to_numpy(dtype=float))
        we = pd.to_datetime(a["window_end"]).to_numpy().astype("datetime64[D]")
        org = pd.to_datetime(daily["session_date"]).to_numpy().astype("datetime64[D]")
        x_ok = np.all(np.isfinite(X), axis=1)
        obs = np.flatnonzero(np.isfinite(y) & x_ok & ~np.isnat(we))
        k_end = np.minimum(eligible_end(we, org) + self.extra, len(org))  # the bug: one row too many
        n_elig = np.searchsorted(obs, k_end, side="left")
        want = predict_mask_for(a) & x_ok
        out = np.full(len(org), np.nan)
        for j in range(self._start(), len(org), self.refit_every):
            rows = np.flatnonzero(want[j : j + self.refit_every]) + j
            if rows.size == 0 or n_elig[j] < self.window:
                continue
            train = obs[n_elig[j] - self.window : n_elig[j]]
            beta = np.linalg.lstsq(X[train], y[train], rcond=None)[0]
            out[rows] = np.exp(X[rows] @ beta)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"],
                                 a["n_t"].to_numpy(dtype=float) * out)


class NoiseModel:
    """Pure noise: ``F = n_t · c · exp(σ ε)``, ``ε ~ N(0, 1)`` i.i.d. (seeded), ``c`` = median ``tv`` of the first
    ``n_level`` sessions. It carries no information about the target and must rank last."""

    name = "NOISE"

    def __init__(self, seed: int = SEED, sigma: float = 1.0, n_level: int = 250):
        self.seed, self.sigma, self.n_level = int(seed), float(sigma), int(n_level)

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        a = align_targets(daily, targets)
        c = float(np.nanmedian(daily["tv"].to_numpy(dtype=float)[: self.n_level]))
        eps = np.random.default_rng(self.seed).standard_normal(len(daily))
        F = np.where(predict_mask_for(a), a["n_t"].to_numpy(dtype=float) * c * np.exp(self.sigma * eps), np.nan)
        return to_forecast_frame(asset, horizon, self.name, daily["session_date"], a["n_t"], F)


class ComboForecaster:
    """COMBO through the leakage detector: the frozen members' forecasts (``models.combo_members``) combined by the
    frozen ``combine``, exactly as the walk-forward forms the headline model."""

    name = "COMBO"

    def __init__(self, members: list | None = None):
        from volrisk import pipeline
        from volrisk.models.combo import combo_members

        self.members = members if members is not None else [pipeline._model(m) for m in combo_members()]

    def refit_schedule(self, daily: pd.DataFrame, asset: str) -> dict[str, list[pd.Timestamp]]:
        out: dict[str, list[pd.Timestamp]] = {}
        for m in self.members:
            out.update(refit_schedule(m, daily, asset))
        return out

    def forecast(self, daily: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str) -> pd.DataFrame:
        from volrisk.models.combo import combine

        parts = [m.forecast(daily, targets, asset, horizon) for m in self.members]
        return combine(pd.concat(parts, ignore_index=True), members=[m.name for m in self.members], name=self.name)


def refit_schedule(model, daily: pd.DataFrame, asset: str) -> dict[str, list[pd.Timestamp]]:
    """Origins of the periodic re-estimations of ``model`` on ``daily`` by component (empty: re-estimated at every
    origin). Frozen GARCH/GJR refit at the return positions ``W-1 + m·R``, LightGBM/MLP at the session rows
    ``oos_start + m·R`` (SPEC §6); COMBO and the toy models describe their own schedule."""
    from volrisk.models import garch, ml

    if hasattr(model, "refit_schedule"):
        return model.refit_schedule(daily, asset)
    dates = pd.to_datetime(daily["session_date"]).reset_index(drop=True)
    if isinstance(model, garch._ArchForecaster):
        W = C.window() if model.window is None else int(model.window)
        R = C.refit_every("garch", asset) if model.refit_every is None else int(model.refit_every)
        rows = np.flatnonzero(np.isfinite(daily["r_cc"].to_numpy(dtype=float)))[W - 1 :: R]
        return {model.name: list(dates.iloc[rows])}
    if isinstance(model, ml._DirectML):
        W = C.window() if model.window is None else int(model.window)
        R = C.refit_every("ml", asset) if model.refit_every is None else int(model.refit_every)
        X = model.design(daily, asset).to_numpy(dtype=float)
        start = ml.oos_start_row(daily, asset, np.all(np.isfinite(X), axis=1), W)
        return {model.name: [] if start is None else list(dates.iloc[list(range(start, len(dates), R))])}
    return {}


def leak_t0s(model, daily: pd.DataFrame, asset: str, t0: date = LEAK_T0) -> list[tuple[date, str, bool]]:
    """Cut-offs of the perturbation test for ``model``: ``(t0, why, is_refit_origin)``, sorted. The fixed ``t0`` plus,
    for every periodically refitted component, its last refit origin at or before ``t0``: a training-label leak at a
    refit origin ``r`` (a label whose window ends after ``r``) changes the forecast at ``r`` only when the data after
    ``r`` is perturbed, so a single fixed cut-off would miss it unless it fell inside that window."""
    daily = daily.sort_values("session_date", kind="stable").reset_index(drop=True)
    why: dict[date, list[str]] = {t0: ["fixed cut-off"]}
    refit: set[date] = set()
    for label, origins in refit_schedule(model, daily, asset).items():
        before = [pd.Timestamp(d).date() for d in origins if pd.Timestamp(d).date() <= t0]
        if before:
            why.setdefault(before[-1], []).append(f"last {label} refit origin <= {t0}")
            refit.add(before[-1])
    return [(d, " = ".join(why[d]), d in refit) for d in sorted(why)]


def perturb_after(daily: pd.DataFrame, t0: date, seed: int = SEED) -> pd.DataFrame:
    """Copy of ``daily`` with every float column of the sessions after ``t0`` multiplied by an independent random
    factor ``exp(U(-1.5, 1.5))`` (signs and positivity kept); rows up to ``t0`` are untouched."""
    out = daily.copy()
    after = (pd.to_datetime(out["session_date"]) > pd.Timestamp(t0)).to_numpy()
    rng = np.random.default_rng(seed)
    for col in out.columns:
        if col == "session_date" or not pd.api.types.is_float_dtype(out[col]):
            continue
        vals = out[col].to_numpy(dtype=float, copy=True)
        vals[after] *= np.exp(rng.uniform(-PERTURB_LOG_RANGE, PERTURB_LOG_RANGE, int(after.sum())))
        out[col] = vals
    return out


def _leak_result(name: str, asset: str, horizon: str, t0: date, f0: pd.DataFrame, f1: pd.DataFrame) -> dict:
    def frame(f: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"origin": pd.to_datetime(f["origin"]).astype("datetime64[ns]"),
                             "F": f["F"].to_numpy(dtype=float)})

    m = frame(f0).merge(frame(f1), on="origin", how="outer", suffixes=("_orig", "_pert"))
    a, b = m["F_orig"].to_numpy(), m["F_pert"].to_numpy()
    same = np.isclose(a, b, rtol=REPRO_RTOL, atol=0.0)  # NaN (missing on one side) counts as changed
    before = (m["origin"] <= pd.Timestamp(t0)).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.abs(b - a) / np.abs(a)
    changed = before & ~same
    rel_before = rel[before & np.isfinite(rel)]
    return {
        "model": name, "asset": asset, "horizon": horizon, "t0": pd.Timestamp(t0).date().isoformat(),
        "n_before": int(before.sum()),
        "n_changed_before": int(changed.sum()),
        "max_rel_change_before": float(rel_before.max()) if rel_before.size else 0.0,
        "first_changed_origin": m.loc[changed, "origin"].min().date().isoformat() if changed.any() else None,
        "n_after": int((~before).sum()),
        "n_changed_after": int((~before & ~same).sum()),
        "leak": bool(changed.any()),
    }


def detect_leak_multi(model, daily: pd.DataFrame, asset: str, horizon: str, t0s: list[date],
                      last_date: date | None = None, seed: int = SEED) -> list[dict]:
    """Leakage perturbation test (LIVE_SPEC §7.6) at several cut-offs, sharing the unperturbed run.

    Forecasts with the original data vs with every value after ``t0`` randomly perturbed (targets rebuilt from the
    perturbed data with ``volrisk.targets.build_targets(last_date=dev_end)``). A model that uses only information
    available at its origin gives the same forecast (rtol 1e-9, atol 0) at every origin ``<= t0``; any change, or a
    forecast present on one side only, flags a leak. ``n_changed_after`` > 0 shows the perturbation took effect.
    """
    last = C.dev_end() if last_date is None else last_date
    daily = daily.sort_values("session_date", kind="stable").reset_index(drop=True)
    f0 = model.forecast(daily, build_targets(daily, asset, horizon, last), asset, horizon)
    out = []
    for t0 in t0s:
        pert = perturb_after(daily, t0, seed)
        f1 = model.forecast(pert, build_targets(pert, asset, horizon, last), asset, horizon)
        out.append(_leak_result(model.name, asset, horizon, t0, f0, f1))
    return out


def detect_leak(model, daily: pd.DataFrame, asset: str, horizon: str, t0: date, last_date: date | None = None,
                seed: int = SEED) -> dict:
    """``detect_leak_multi`` at one cut-off ``t0``."""
    return detect_leak_multi(model, daily, asset, horizon, [t0], last_date, seed)[0]


def _leak_model(name: str):
    """A frozen forecast model by name (``volrisk.pipeline._model``), COMBO, or one of the cheating toy models."""
    toys = {"ORACLE": OracleModel, "PEEK": PeekModel, "LABEL-LEAK": LabelLeakModel, "COMBO": ComboForecaster}
    if name in toys:
        return toys[name]()
    from volrisk import pipeline

    return pipeline._model(name)


def _leak_task(task: tuple) -> list[dict]:
    """One ``(model, asset, horizon, t0 indices or None)`` detector run on dev data; module level, so worker
    processes can run it (nothing here depends on an overridden data end)."""
    name, asset, horizon, only = task
    model = _leak_model(name)
    daily = io.load_daily(asset).sort_values("session_date", kind="stable").reset_index(drop=True)  # dev loader
    t0s = leak_t0s(model, daily, asset)
    if only is not None:
        t0s = [t0s[i] for i in only if i < len(t0s)]
    if not t0s:
        return []
    rows = detect_leak_multi(model, daily, asset, horizon, [t for t, _, _ in t0s])
    for r, (_, why, is_refit) in zip(rows, t0s, strict=True):
        r.update(t0_from=why, t0_is_refit_origin=is_refit)
    return rows


def leak_tasks(models: tuple[str, ...] = LEAK_MODELS) -> list[tuple]:
    """Detector tasks of 6a, slowest first: every model on ``LEAK_CELLS`` (MLP on one cell, one task per cut-off)."""
    order = {m: i for i, m in enumerate(LEAK_COST)}
    tasks: list[tuple] = []
    for name in sorted(models, key=lambda m: order.get(m, len(order))):
        for asset, horizon in LEAK_ONE_CELL.get(name, LEAK_CELLS):
            if name in LEAK_SPLIT_T0:
                tasks += [(name, asset, horizon, (i,)) for i in range(2)]  # fixed cut-off + one refit origin
            else:
                tasks.append((name, asset, horizon, None))
    return tasks


def check_leak_frozen(workers: int = 10, tasks: list[tuple] | None = None) -> tuple:
    """§7.6: every frozen forecast model and COMBO passes the leakage perturbation test at the fixed cut-off and at
    the refit origins of its periodically re-estimated components."""
    tasks = leak_tasks() if tasks is None else tasks
    rows = [r for part in _run_tasks(_leak_task, tasks, workers) for r in part]
    clean = [r for r in rows if not r["leak"] and r["n_before"] > 0 and r["n_changed_after"] > 0]
    ok = len(clean) == len(rows) and len(rows) > 0
    models = list(dict.fromkeys(r["model"] for r in rows))
    ev = {"summary": f"{len(clean)} of {len(rows)} model x cell x cut-off runs clean, {len(models)} models "
                     f"({sum(r['n_before'] for r in rows)} forecasts at origins <= t0 compared)",
          "models": models, "fixed_t0": LEAK_T0,
          "t0_rule": "the fixed cut-off and, for GARCH/GJR/LightGBM/MLP (and COMBO through GJR and LightGBM), the last "
                     "refit origin at or before it",
          "perturbation": f"every float after t0 x exp(U(-{PERTURB_LOG_RANGE}, {PERTURB_LOG_RANGE}))",
          "runs": rows}
    bad = [f"{r['model']} {r['asset']} {r['horizon']} t0 {r['t0']}" for r in rows if r not in clean]
    return _status(ok), ev, None if ok else f"leak or vacuous test: {', '.join(bad) or 'no runs'}"


CHEAT_MODELS = ("ORACLE", "PEEK", "LABEL-LEAK")


def check_leak_cheaters(workers: int = 1) -> tuple:
    """§7.6: the detector flags the three cheating toy models: the oracle and the peek at the fixed cut-off, the
    training-label leak at its refit origin. Its run at the fixed cut-off alone is shown for comparison."""
    tasks = [(m, a, h, None) for m in CHEAT_MODELS for a, h in LEAK_CELLS]
    rows = [r for part in _run_tasks(_leak_task, tasks, workers) for r in part]
    for r in rows:
        r["must_flag"] = r["model"] != "LABEL-LEAK" or r["t0_is_refit_origin"]
    must = [r for r in rows if r["must_flag"]]
    caught = [r for r in must if r["leak"]]
    fixed_only = [r for r in rows if r["model"] == "LABEL-LEAK" and not r["t0_is_refit_origin"]]
    # one flagged run per cheater and cell: the label cheat must have been cut at a refit origin in every cell
    ok = len(caught) == len(must) == len(CHEAT_MODELS) * len(LEAK_CELLS)
    ev = {"summary": f"{len(caught)} of {len(must)} cheating runs flagged (training-label leak: "
                     f"{sum(r['leak'] for r in must if r['model'] == 'LABEL-LEAK')} of "
                     f"{sum(r['model'] == 'LABEL-LEAK' for r in must)} at its refit origin, "
                     f"{sum(r['leak'] for r in fixed_only)} of {len(fixed_only)} at the fixed cut-off alone)",
          "fixed_t0": LEAK_T0, "runs": rows,
          "note": "the training-label cheat refits every 90 rows and trains on one row whose target window has not "
                  "ended; it is visible only when the cut-off falls within one target window after a refit origin, "
                  "which is why 6a also cuts at every refitted model's refit origin"}
    return _status(ok), ev, None if ok else "a cheating model was not flagged"


def noise_control(daily: pd.DataFrame, forecasts: pd.DataFrame, targets: pd.DataFrame, asset: str, horizon: str,
                  start: date, end: date, seed: int = SEED, reps: int | None = None) -> dict:
    """§7.6: ``NoiseModel`` next to the stored forecasts of one dev cell; mean QLIKE ranking and the frozen MCS
    (``volrisk.evaluation.mcs.run_mcs``, SPEC §8 settings) on the common origins in ``[start, end]``."""
    from volrisk.evaluation.iv import IV_MODELS
    from volrisk.evaluation.leaderboard import common_dates, losses_frame
    from volrisk.evaluation.mcs import run_mcs

    cfg = C.load()["evaluation"]
    tg = targets[(targets["asset"] == asset) & (targets["horizon"] == horizon)].reset_index(drop=True)
    tg = tg.assign(origin=pd.to_datetime(tg["origin"]).astype(daily["session_date"].dtype))
    fc = forecasts[(forecasts["asset"] == asset) & (forecasts["horizon"] == horizon)
                   & ~forecasts["model"].isin(IV_MODELS)]
    noise = NoiseModel(seed).forecast(daily, tg, asset, horizon)
    cols = ["asset", "horizon", "model", "origin", "n_t", "F"]
    losses = losses_frame(pd.concat([fc[cols], noise[cols]], ignore_index=True), tg)
    losses = losses[(losses["origin"] >= pd.Timestamp(start)) & (losses["origin"] <= pd.Timestamp(end))]
    models = list(dict.fromkeys(losses["model"]))
    wide = common_dates(losses, models, "qlike")
    mcs = run_mcs(wide, C.n_max(horizon, asset), size=float(cfg["mcs_size"]),
                  reps=int(cfg["mcs_reps"]) if reps is None else int(reps), seed=C.seed()).set_index("model")
    mean = wide.mean().sort_values()
    ref = mean.get("HAR", np.nan)
    ranking = [{"rank": i + 1, "model": m, "mean_qlike": float(v), "ratio_vs_HAR": float(v / ref),
                "mcs_pvalue": float(mcs.loc[m, "pvalue"]), "in_90_mcs": bool(mcs.loc[m, "in_90"])}
               for i, (m, v) in enumerate(mean.items())]
    noise_row = next(r for r in ranking if r["model"] == NoiseModel.name)
    return {"asset": asset, "horizon": horizon, "start": wide.index.min(), "end": wide.index.max(), "T": len(wide),
            "n_models": len(models), "noise_rank": noise_row["rank"], "noise_last": noise_row["rank"] == len(ranking),
            "noise_mcs_pvalue": noise_row["mcs_pvalue"], "noise_in_90_mcs": noise_row["in_90_mcs"],
            "ranking": ranking}


def check_noise(daily: pd.DataFrame | None = None, forecasts: pd.DataFrame | None = None,
                targets: pd.DataFrame | None = None) -> tuple:
    """§7.6: the noise model ranks last and is outside the 90% MCS on the stored dev forecasts of ``NOISE_CELL``."""
    asset, horizon = NOISE_CELL
    fc = pd.read_parquet(io.FORECASTS) if forecasts is None else forecasts
    tg = pd.read_parquet(C.RESULTS / "targets.parquet") if targets is None else targets
    daily = io.load_daily(asset) if daily is None else daily
    res = noise_control(daily, fc, tg, asset, horizon, *NOISE_WINDOW)
    ok = res["noise_last"] and not res["noise_in_90_mcs"]
    res["summary"] = (f"noise ranks {res['noise_rank']} of {len(res['ranking'])} on {asset} {horizon} "
                      f"({res['T']} dev origins), MCS p = {res['noise_mcs_pvalue']:.4f}")
    return _status(ok), res, None if ok else "the noise model was not ranked last / was inside the 90% MCS"


# --------------------------------------------------------------------------------------------- 7 no tuning
def _num(text: str) -> float | int:
    v = ast.literal_eval(text)
    if not isinstance(v, (int, float)):
        raise ValueError(text)
    return v


def spec_hyperparameters(text: str) -> dict:
    """The hyperparameters written in docs/SPEC.md (§6 walk-forward and HAR lag grid, §7 LightGBM and MLP) as a
    flat ``{dotted name: value}``; a value that cannot be found is absent."""
    text = " ".join(text.split())  # sentences wrap across lines in the markdown source
    out: dict = {}
    m = re.search(r"LGBMRegressor\((.*?)\)", text, re.S)
    if m:
        for part in m.group(1).split(","):
            k, _, v = part.partition("=")
            if k.strip() in LGBM_KEYS:
                out[f"lgbm.{k.strip()}"] = ast.literal_eval(v.strip())
    pats = {
        "mlp.hidden": r"Architecture `in\s*→\s*(\d+)\s*→\s*(\d+)\s*→\s*1`",
        "mlp.lr": r"Adam lr ([0-9.eE+-]+)",
        "mlp.weight_decay": r"weight decay ([0-9.eE+-]+)",
        "mlp.epochs": r"(\d+) epochs",
        "mlp.n_seeds": r"(\d+)-seed average",
        "walk_forward.window": r"\*\*W = (\d+)\*\*",
        "garch": r"GARCH/GJR every (\d+) sessions \(SPX/EURUSD\) or (\d+) days \(crypto\)",
        "ml": r"LightGBM & MLP every (\d+) sessions \(SPX/EURUSD\) or (\d+) days \(crypto\)",
        "har_lags": r"SPX & EURUSD \((\d+), (\d+), (\d+)\); crypto \((\d+), (\d+), (\d+)\)",
    }
    for key, pat in pats.items():
        m = re.search(pat, text)
        if not m:
            continue
        g = [_num(x) for x in m.groups()]
        if key == "mlp.hidden":
            out[key] = g
        elif key in ("garch", "ml"):
            for clock, v in (("fx", g[0]), ("xnys", g[0]), ("crypto", g[1])):
                out[f"walk_forward.refit.{key}.{clock}"] = v
        elif key == "har_lags":
            out.update({"har_lags.fx": g[:3], "har_lags.xnys": g[:3], "har_lags.crypto": g[3:]})
        else:
            out[key] = g[0]
    return out


def _lookup(d: dict, dotted: str):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _equal(a, b) -> bool:
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) and len(a) == len(b) and all(
            _equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        return float(a) == float(b)
    return a == b


def check_no_tuning(spec_path: Path = SPEC_PATH, frozen_path: Path = C.FROZEN_PATH,
                    config_path: Path = C.CONFIG_PATH, sensitivity_path: Path = SENSITIVITY_LGBM,
                    deviations_path: Path = DEVIATIONS_PATH) -> tuple:
    """§7.7: frozen hyperparameters = the values written in SPEC.md; the better dev LightGBM setting found in the
    post-holdout sensitivity analysis (7 leaves) was not adopted; post-start changes are listed in DEVIATIONS.md."""
    if not spec_path.exists():  # e.g. a clone without the docs/ folder: nothing to compare against
        raise SkipCheck(f"{spec_path.name} is not in this checkout, so the pre-registered values cannot be compared")
    spec = spec_hyperparameters(spec_path.read_text(encoding="utf-8"))
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    names = ([f"lgbm.{k}" for k in LGBM_KEYS] + [f"mlp.{k}" for k in ("hidden", "epochs", "lr", "weight_decay",
             "n_seeds")] + ["walk_forward.window"]
             + [f"walk_forward.refit.{k}.{c}" for k in ("garch", "ml") for c in ("crypto", "fx", "xnys")]
             + [f"har_lags.{c}" for c in ("crypto", "fx", "xnys")])

    def where(src: dict, n: str, models_key: str):
        return _lookup(src, f"{models_key}.{n}") if n.startswith(("lgbm.", "mlp.")) else _lookup(src, n)

    rows = []
    for n in names:
        s, f, c = spec.get(n), where(frozen, n, "hyperparameters"), where(cfg, n, "models")
        rows.append({"parameter": n, "SPEC.md": s, "frozen.yaml": f, "config.yaml": c,
                     "equal": s is not None and _equal(s, f) and _equal(s, c)})
    n_eq = sum(r["equal"] for r in rows)
    ev: dict = {"summary": f"{n_eq} of {len(rows)} hyperparameters equal in SPEC.md, frozen.yaml and config.yaml",
                "spec": _rel(spec_path), "parameters": rows}
    ok = n_eq == len(rows)
    leaves_frozen = _lookup(frozen, "hyperparameters.lgbm.num_leaves")
    ok &= leaves_frozen == spec.get("lgbm.num_leaves")
    if sensitivity_path.exists():
        sens = pd.read_csv(sensitivity_path)
        alt = next((c for c in sens.columns if "7 leaves" in c), None)
        base = next((c for c in sens.columns if "15 leaves" in c), None)
        if alt and base:
            better = sens[alt] < sens[base]
            ev["sensitivity"] = {
                "file": _rel(sensitivity_path),
                "cells_where_7_leaves_better": int(better.sum()), "cells": len(sens),
                "mean_qlike_ratio_15_leaves": float(sens[base].mean()),
                "mean_qlike_ratio_7_leaves": float(sens[alt].mean()),
                "frozen_num_leaves": leaves_frozen,
                "written_utc": _iso(datetime.fromtimestamp(sensitivity_path.stat().st_mtime, UTC)),
                "note": "the 7-leaf setting was better on dev but the sealed config keeps the pre-registered 15",
            }
    if deviations_path.exists():
        text = deviations_path.read_text(encoding="utf-8")
        ev["deviations"] = {
            "file": _rel(deviations_path),
            "sections": re.findall(r"^## (.+)$", text, re.M),
            "n_entries": len(re.findall(r"^- ", text, re.M)),
            # entries appended by volrisk.holdout (re-opening / re-freeze after an opening) start with a UTC time
            "n_automatic_reopen_or_refreeze_entries": len(re.findall(r"^- \d{4}-\d{2}-\d{2}T", text, re.M)),
        }
    return _status(ok), ev, None if ok else "a frozen hyperparameter differs from SPEC.md"


# --------------------------------------------------------------------------------------------- 8 ledger, timestamps
def _ledger_lines(ledger_path: Path) -> list[str]:
    if not ledger_path.exists():
        return []
    return [ln for ln in ledger_path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def run_files(runs_dir: Path | None = None) -> list[Path]:
    """Files under ``forecasts/runs/`` that belong to a ledger entry: payloads, entry files and their proofs."""
    runs_dir = paths.RUNS if runs_dir is None else Path(runs_dir)
    if not runs_dir.exists():
        return []
    return sorted(p for p in runs_dir.iterdir() if p.is_file() and RUN_FILE_RE.match(p.name))


def days_without_run(run_ids: list[str]) -> list[str]:
    """UTC dates between the first and the last run on which no run was recorded."""
    days = sorted({date(int(r[:4]), int(r[4:6]), int(r[6:8])) for r in run_ids if r and re.match(r"^\d{8}T", r)})
    if len(days) < 2:
        return []
    have = set(days)
    return [d.isoformat() for d in (days[0] + timedelta(n) for n in range((days[-1] - days[0]).days + 1))
            if d not in have]


def verify_ledger_file(ledger_path: Path | None = None, runs_dir: Path | None = None) -> dict:
    """Independent re-implementation of the §4 chain rule: ``entry_sha256 = sha256(canonical_json(entry without
    entry_sha256))``, ``prev_entry_sha256`` = previous entry hash (``"0"*64`` first), consecutive ``seq``, the
    payload file's SHA-256 = ``payload_sha256``, the entry file ``runs/<run_id>.entry`` hashes to ``entry_sha256``,
    and no payload, entry file or proof under ``runs/`` belongs to a run the ledger does not record."""
    ledger_path = paths.LEDGER if ledger_path is None else Path(ledger_path)
    runs_dir = paths.RUNS if runs_dir is None else Path(runs_dir)
    lines = _ledger_lines(ledger_path)
    prev, seq0, problems, n_payload_ok, n_entry_ok, run_ids = schema.ZERO_HASH, None, [], 0, 0, []
    for i, line in enumerate(lines):
        try:
            e = json.loads(line)
        except ValueError:
            problems.append(f"entry {i}: not valid JSON")
            prev = None
            continue
        run_ids.append(e.get("run_id"))
        body = {k: v for k, v in e.items() if k != "entry_sha256"}
        if schema.sha256_text(schema.canonical_json(body)) != e.get("entry_sha256"):
            problems.append(f"entry {i} ({e.get('run_id')}): entry_sha256 does not match the entry content")
        if e.get("prev_entry_sha256") != prev:
            problems.append(f"entry {i} ({e.get('run_id')}): does not link to the previous entry")
        seq0 = e.get("seq") if i == 0 else seq0
        if isinstance(seq0, int) and e.get("seq") != seq0 + i:
            problems.append(f"entry {i}: seq {e.get('seq')} is not consecutive")
        payload = runs_dir / f"{e.get('run_id')}.json"
        if not payload.exists():
            problems.append(f"entry {i}: payload {payload.name} is missing")
        elif _sha256_file(payload) != e.get("payload_sha256"):
            problems.append(f"entry {i}: payload {payload.name} does not match payload_sha256 (edited)")
        else:
            n_payload_ok += 1
        entry_file = runs_dir / f"{e.get('run_id')}.entry"
        if not entry_file.exists():
            problems.append(f"entry {i}: entry file {entry_file.name} is missing")
        elif _sha256_file(entry_file) != e.get("entry_sha256"):
            problems.append(f"entry {i}: entry file {entry_file.name} does not hash to entry_sha256 (rewritten)")
        else:
            n_entry_ok += 1
        prev = e.get("entry_sha256")
    recorded = set(run_ids)
    orphans = [p.name for p in run_files(runs_dir) if RUN_FILE_RE.match(p.name).group(1) not in recorded]
    if orphans:
        problems.append(f"file(s) of a run without a ledger entry: {', '.join(orphans[:5])}")
    return {"ok": not problems, "n_entries": len(lines), "n_payloads_ok": n_payload_ok, "n_entry_files_ok": n_entry_ok,
            "head_sha256": prev, "first_run": run_ids[0] if run_ids else None,
            "last_run": run_ids[-1] if run_ids else None, "run_ids": run_ids,
            "days_without_run": days_without_run([r for r in run_ids if isinstance(r, str)]),
            "orphans": orphans[:20], "first_broken": problems[0] if problems else None, "problems": problems[:20]}


def scored_run_ids() -> dict[str, list[str]]:
    """run_ids that ``forecasts/scores.csv`` / ``risk_scores.csv`` hold scores for (by file)."""
    out: dict[str, list[str]] = {}
    for p in (paths.SCORES, paths.RISK_SCORES):
        if not p.exists():
            continue
        try:
            ids = pd.read_csv(p, usecols=["run_id"], dtype=str)["run_id"].dropna().unique().tolist()
        except (ValueError, KeyError, pd.errors.EmptyDataError):
            ids = []
        out[_rel(p)] = sorted(ids)
    return out


def check_ledger() -> tuple:
    """§7.8: the forward-test hash chain recomputes (independently and with ``ledger.verify_chain``), nothing under
    ``forecasts/`` belongs to a run outside it, and the UTC days without a run are listed."""
    files = run_files()
    scored = scored_run_ids()
    n_scored = sum(len(v) for v in scored.values())
    if not _ledger_lines(paths.LEDGER):
        if files or n_scored:
            ev = {"summary": f"no ledger, but {len(files)} run file(s) and scores of {n_scored} run(s) exist",
                  "run_files": [p.name for p in files[:20]], "scored_run_ids": scored}
            return "FAIL", ev, ("forecast files or scores exist but forecasts/ledger.jsonl is missing or empty "
                                "(ledger deleted or truncated?)")
        raise SkipCheck("no forecast run recorded yet (no ledger, no run files, no scores)")
    own = verify_ledger_file()
    ev: dict = {"independent": own}
    ok = own["ok"]
    problems = list(own["problems"])
    try:
        mod = importlib.import_module("volrisk_live.ledger")
    except ImportError as e:
        ev["ledger_module"] = f"not available ({e})"
    else:
        res = mod.verify_chain()
        ev["ledger_module"] = res
        if _ok_of(res) is False:
            ok = False
            problems.append(f"ledger.verify_chain: {res.get('reason') if isinstance(res, dict) else res}")
    recorded = set(own["run_ids"])
    unknown = {f: [r for r in ids if r not in recorded] for f, ids in scored.items()}
    unknown = {f: ids for f, ids in unknown.items() if ids}
    if unknown:
        ok = False
        problems.append(f"scores of run(s) the ledger does not record: {unknown}")
    gaps = own["days_without_run"]
    ev["scored_runs_not_in_ledger"] = unknown
    ev["days_without_run"] = gaps
    ev["head_sha256"] = own["head_sha256"]
    ev["note"] = ("runs dropped from the end of the chain together with all their files leave no trace here: compare "
                  "head_sha256 with a copy published elsewhere (e.g. the last pushed forecasts/ledger.jsonl); a "
                  "dropped run in the middle of the chain, with the later entries rewritten, is caught by 8c")
    ev["summary"] = (f"{own['n_entries']} entries ({own['first_run']} .. {own['last_run']}), {own['n_payloads_ok']} "
                     f"payloads and {own['n_entry_files_ok']} entry files match, head {str(own['head_sha256'])[:16]}; "
                     f"UTC days without a run: {', '.join(gaps) if gaps else 'none'}") if ok else \
        f"broken: {problems[0]}"
    if not ok:
        return "FAIL", ev, problems[0]
    reason = None
    if gaps:
        reason = (f"{len(gaps)} UTC day(s) between the first and the last run have no run ({', '.join(gaps[:10])}): "
                  "a run that was dropped cannot be told from a day without a run")
    return "PASS", ev, reason


def check_anchors(network: bool = True) -> tuple:
    """§7.8: every ledger entry file is pinned in a Bitcoin block soon after its run (``ledger.verify_anchors``): a
    chain rewritten after a removed or edited run must be stamped again, so its pins come late."""
    if not _ledger_lines(paths.LEDGER):
        raise SkipCheck("no ledger entries (see 8a)")
    if not network:
        raise SkipCheck("network disabled (the pins are checked against Bitcoin block headers)")
    res = _live_module("ledger").verify_anchors()
    rows = res.get("entries") or []
    failed = [r for r in rows if r.get("status") == "failed"]
    late = [r for r in rows if r.get("status") == "late"]
    # a late entry without any pin whose own proof is only waiting (not upgraded) or unverifiable (no explorer) is
    # not evidence either way; a late pin, or no proof at all, is
    late_open = [r for r in late if not r.get("pinned_utc") and r.get("proof") in ("pending", "partial", "upgraded",
                                                                                     "unverified", "error")]
    late_bad = [r for r in late if r not in late_open]
    counts = res.get("counts") or {}
    ev = {"summary": f"{len(rows)} entries: " + (", ".join(f"{v} {k}" for k, v in sorted(counts.items())) or "none"),
          "counts": counts, "max_lag_hours": res.get("max_lag_hours"), "chain_ok": res.get("chain_ok"),
          "entries": rows, "note": res.get("note")}
    if failed or late_bad:
        r = (failed or late_bad)[0]
        return "FAIL", ev, f"entry {r.get('seq')} ({r.get('run_id')}): {r.get('status')} - {r.get('reason')}"
    if late_open:
        raise SkipCheck(f"{len(late_open)} entries older than {res.get('max_lag_hours')} h are not pinned yet: their "
                        "proofs are not upgraded or no block explorer answered - run `python -m volrisk_live stamp`, "
                        "then verify again", ev)
    n_pending = counts.get("pending", 0)
    return "PASS", ev, (f"{n_pending} recent entries wait for their Bitcoin attestation" if n_pending else None)


def inspect_ots(path: Path) -> dict:
    """Offline look at ``path.ots`` (opentimestamps core): does it commit to the file's SHA-256, and which
    attestations does it carry (calendar = pending, Bitcoin block heights)."""
    from opentimestamps.core.notary import BitcoinBlockHeaderAttestation, PendingAttestation
    from opentimestamps.core.op import OpSHA256
    from opentimestamps.core.serialize import StreamDeserializationContext
    from opentimestamps.core.timestamp import DetachedTimestampFile

    with open(Path(str(path) + ".ots"), "rb") as f:
        dtf = DetachedTimestampFile.deserialize(StreamDeserializationContext(f))
    digest = hashlib.sha256(Path(path).read_bytes()).digest()
    atts = list(dtf.timestamp.all_attestations())
    return {"digest_ok": isinstance(dtf.file_hash_op, OpSHA256) and dtf.file_digest == digest,
            "pending_calendars": sorted({a.uri for _, a in atts if isinstance(a, PendingAttestation)}),
            "bitcoin_heights": sorted({a.height for _, a in atts if isinstance(a, BitcoinBlockHeaderAttestation)})}


def _session(asset: str, day: date) -> tuple[datetime, datetime] | None:
    """(open_utc, close_utc) of a scheduled session of the frozen calendar, or None."""
    from volrisk import sessions

    s = sessions.session_schedule(asset, day, day)
    if not s.height:
        return None
    o, c = s["open_utc"][0], s["close_utc"][0]
    return (o if o.tzinfo else o.replace(tzinfo=UTC)), (c if c.tzinfo else c.replace(tzinfo=UTC))


def payload_windows(payload: dict) -> list[dict]:
    """Target windows of a payload, one per (asset, horizon) forecast window and per VaR/ES date: the open of the
    first session, the close of the first session (its first outcome) and the close of the last session, in UTC."""
    items = [(r.get("asset"), r.get("horizon"), r.get("window_first"), r.get("window_last") or r.get("window_first"))
             for r in payload.get("forecasts", [])]
    items += [(r.get("asset"), "VaR/ES", r.get("date"), r.get("date")) for r in payload.get("risk", [])]
    seen, out = set(), []
    for asset, horizon, first, last in items:
        if asset not in C.ASSETS or not first:
            continue
        key = (asset, horizon, str(first)[:10], str(last)[:10])
        if key in seen:
            continue
        seen.add(key)
        f, lst = _session(asset, date.fromisoformat(key[2])), _session(asset, date.fromisoformat(key[3]))
        out.append({"asset": asset, "horizon": horizon, "window_first": key[2], "window_last": key[3],
                    "open_utc": f[0] if f else None, "first_close_utc": f[1] if f else None,
                    "close_utc": lst[1] if lst else None})
    return out


def first_outcome_utc(payload: dict) -> datetime | None:
    """Earliest time at which any outcome of a payload is complete: the close of the first session of every
    forecast window (``window_first``) and of every risk date, from the frozen session calendars."""
    closes = [w["first_close_utc"] for w in payload_windows(payload) if w["first_close_utc"] is not None]
    return min(closes) if closes else None


def _window_timing(w: dict, proven: datetime) -> dict:
    """Share of a target window elapsed at the Bitcoin-proven time (0 = before it opened, 1 = complete)."""
    o, c = w["open_utc"], w["close_utc"]
    frac = None
    if o is not None and c is not None and c > o:
        frac = min(max((proven - o).total_seconds() / (c - o).total_seconds(), 0.0), 1.0)
    return {"asset": w["asset"], "horizon": w["horizon"], "window_first": w["window_first"],
            "window_last": w["window_last"], "open_utc": _iso(o), "first_close_utc": _iso(w["first_close_utc"]),
            "close_utc": _iso(c), "before_open": o is not None and proven < o,
            "before_first_close": w["first_close_utc"] is not None and proven < w["first_close_utc"],
            "elapsed_at_proof": None if frac is None else round(frac, 4)}


def _payload_run_utc(payload: dict, path: Path) -> datetime | None:
    try:
        return _utc(payload["run_utc"])
    except (KeyError, TypeError, ValueError):
        m = re.match(r"^(\d{8}T\d{6}Z)", path.name)
        return datetime.strptime(m.group(1), "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC) if m else None


def check_timestamps(network: bool = True, now_utc: datetime | None = None) -> tuple:
    """§7.8: every payload carries an OpenTimestamps proof of its exact bytes; Bitcoin attestations are verified
    (``ots.verify``: block header, proof of work, merkle root) and their block time + ``BLOCK_TIME_SLACK`` precedes
    the payload's first outcome. Per target window the share already elapsed at that proven time is reported."""
    payloads = sorted(paths.RUNS.glob("*.json")) if paths.RUNS.exists() else []
    if not payloads:
        raise SkipCheck("no forecast payloads under forecasts/runs/ yet")
    now = (now_utc or datetime.now(UTC)).astimezone(UTC)
    try:
        ots = importlib.import_module("volrisk_live.ots")
    except ImportError:
        ots = None
    rows, problems, unchecked, waiting, n_verified, elapsed_1d = [], [], [], [], 0, []
    for p in payloads:
        rec: dict = {"payload": p.name, "ots": Path(str(p) + ".ots").exists()}
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
        except ValueError as e:
            problems.append(f"{p.name}: not valid JSON ({e})")
            rows.append(rec)
            continue
        run_utc = _payload_run_utc(payload, p)
        rec["run_utc"] = _iso(run_utc)
        if not rec["ots"]:
            problems.append(f"{p.name}: no .ots proof")
            rows.append(rec)
            continue
        try:
            info = inspect_ots(p)
        except Exception as e:  # noqa: BLE001 - an unreadable proof is a failed proof
            problems.append(f"{p.name}: unreadable proof ({type(e).__name__}: {e})")
            rows.append(rec)
            continue
        rec.update(digest_ok=info["digest_ok"], n_calendars=len(info["pending_calendars"]),
                   bitcoin_heights=info["bitcoin_heights"])
        if not info["digest_ok"]:
            problems.append(f"{p.name}: the proof does not commit to the current file (edited after stamping)")
        windows = payload_windows(payload)
        outcome = first_outcome_utc(payload)
        rec["first_outcome_utc"] = _iso(outcome)
        if info["bitcoin_heights"]:
            if not network or ots is None:
                rec["bitcoin"] = "not checked (" + ("network disabled" if not network else "ots module missing") + ")"
                unchecked.append(f"{p.name} ({rec['bitcoin']})")
            else:
                v = ots.verify(p)
                rec["bitcoin"] = v.get("status")
                rec["attested_utc"] = v.get("attested_utc")
                if v.get("status") in ("failed", "invalid"):
                    problems.append(f"{p.name}: Bitcoin verification {v.get('status')} ({v.get('reason')})")
                elif v.get("status") == "verified":
                    n_verified += 1
                    proven = _utc(v["attested_utc"]) + BLOCK_TIME_SLACK
                    rec["proven_by_utc"] = _iso(proven)
                    rec["windows"] = [_window_timing(w, proven) for w in windows]
                    elapsed_1d += [w["elapsed_at_proof"] for w in rec["windows"]
                                   if w["horizon"] in ("1d", "VaR/ES") and w["elapsed_at_proof"] is not None]
                    rec["before_first_outcome"] = outcome is None or proven < outcome
                    if outcome is not None:
                        rec["hours_before_first_outcome"] = round((outcome - proven).total_seconds() / 3600, 2)
                        if proven >= outcome:
                            problems.append(f"{p.name}: Bitcoin block time {v['attested_utc']} + "
                                            f"{BLOCK_TIME_SLACK.total_seconds() / 3600:g} h is not before the first "
                                            f"outcome {_iso(outcome)}")
                else:
                    unchecked.append(f"{p.name} ({v.get('status')}: {v.get('reason')})")
        elif run_utc is not None and now - run_utc > ANCHOR_LAG:
            waiting.append(p.name)
        rows.append(rec)
    n_pending = sum(1 for r in rows if r.get("ots") and not r.get("bitcoin_heights"))
    timing = (f"; at the proven time the next sessions were up to {max(elapsed_1d):.0%} elapsed"
              if elapsed_1d else "")
    ev: dict = {"summary": f"{len(payloads)} payloads, {sum(r['ots'] for r in rows)} with a proof, "
                           f"{n_verified} Bitcoin-verified, {n_pending} waiting for Bitcoin{timing}",
                "n_payloads": len(payloads), "n_bitcoin_verified": n_verified, "n_pending": n_pending,
                "block_time_slack_hours": BLOCK_TIME_SLACK.total_seconds() / 3600,
                "max_elapsed_next_session": max(elapsed_1d) if elapsed_1d else None,
                "payloads": rows, "problems": problems[:20], "not_checked": unchecked[:20],
                "older_than_lag_without_bitcoin": waiting[:20],
                "note": "a forecast of a 24-hour market is recorded during its target session (the previous day's "
                        "data are complete only when the next session has begun); its value depends on data up to "
                        "its origin only, which 8d re-derives"}
    if ots is not None:
        try:
            ev["ots_status_all"] = ots.status_all().get("counts")
        except Exception as e:  # noqa: BLE001 - evidence only
            ev["ots_status_all"] = f"{type(e).__name__}: {e}"
    if problems:
        return "FAIL", ev, problems[0]
    if unchecked:
        raise SkipCheck(f"{len(unchecked)} payload proof(s) claim a Bitcoin attestation that was not checked here: "
                        f"{'; '.join(unchecked[:3])}", ev)
    if waiting:
        raise SkipCheck(f"{len(waiting)} payload(s) older than {ANCHOR_LAG.total_seconds() / 3600:g} h have no Bitcoin "
                        "attestation in their proof yet - run `python -m volrisk_live stamp`, then verify again", ev)
    reason = None if n_verified == len(payloads) else (f"{n_pending} recent payload(s) wait for their Bitcoin "
                                                       "attestation (proofs are upgraded a few hours after stamping)")
    return "PASS", ev, reason


def ex_ante_window(asset: str, horizon: str, origin: date) -> tuple[date, date, int]:
    """LIVE_SPEC §3 ex-ante window of a live origin from the frozen session calendar alone: ``(first, last, n_t)``
    with ``n_t`` = scheduled sessions in ``(t, t+days]`` capped at ``n_max`` (1d: the next scheduled session)."""
    from volrisk import sessions

    days = C.horizon_days(horizon)
    sched = sessions.session_schedule(asset, origin + timedelta(days=1),
                                      origin + timedelta(days=days + EX_ANTE_PAD))["session_date"].to_list()
    if not sched:
        raise ValueError(f"{asset}: no scheduled session within {days + EX_ANTE_PAD} days after {origin}")
    n = 1 if horizon == "1d" else sum(d <= origin + timedelta(days=days) for d in sched)
    n = min(n, C.n_max(horizon, asset))
    if n < 1:
        raise ValueError(f"{asset} {horizon}: empty scheduled window after {origin}")
    return sched[0], sched[n - 1], n


def _payloads() -> list[tuple[Path, dict | None]]:
    out = []
    for p in sorted(paths.RUNS.glob("*.json")) if paths.RUNS.exists() else []:
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except ValueError:
            out.append((p, None))
    return out


def payload_inputs(items: list[tuple[Path, dict | None]]) -> dict[tuple[str, str], bool | None]:
    """Per (payload file, asset): do the live gold rows up to the payload's last session still hash to the
    ``data.<asset>.rows_sha256`` the payload recorded (``forecast.rows_sha256``)? A forecast whose input rows were
    revised since its run (a re-published source day) cannot be re-derived from today's files. Empty when the live
    tables cannot be read."""
    try:
        update, forecast = _live_module("update"), _live_module("forecast")
        daily = forecast.normalise_daily(update.live_daily())
    except Exception as e:  # noqa: BLE001 - a diagnostic only: without it no row is marked as revised
        log.warning("payload input hashes not checked: %s: %s", type(e).__name__, e)
        return {}
    sd = pd.to_datetime(daily["session_date"])
    out: dict[tuple[str, str], bool | None] = {}
    for p, payload in items:
        for asset, d in ((payload or {}).get("data") or {}).items():
            try:
                rows = daily[(daily["asset"] == asset) & (sd <= pd.Timestamp(str(d["last_session"])[:10]))]
                out[(p.name, asset)] = forecast.rows_sha256(rows) == d.get("rows_sha256")
            except (KeyError, TypeError, ValueError, AttributeError):
                out[(p.name, asset)] = None
    return out


def _live_walk_forward(workers: int) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """``--full``: the live walk-forward re-run in memory (``forecast.compute`` on the live tables, which re-checks
    the stored dev/holdout forecasts on the way)."""
    update, forecast = _live_module("update"), _live_module("forecast")
    context.require_opened_holdout()
    end = date.fromisoformat(str(update.read_state()["end"])[:10])
    run = forecast.compute(end, update.live_daily(), update.live_implied(), workers)
    return run.fc, run.risk, f"in-memory live walk-forward to {end} (--full)"


def check_payload_forecasts(full: bool = False, workers: int = 10) -> tuple:
    """§7.8: every forecast and VaR/ES row recorded in a ledger payload is what the frozen walk-forward gives at its
    origin (rtol 1e-9, atol 0), and every forecast window / ``n_t`` is the ex-ante one of the session calendar. The
    walk-forward is causal, so a later run reproduces the forecasts of earlier live origins."""
    items = _payloads()
    if not items:
        raise SkipCheck("no forecast payloads under forecasts/runs/ yet")
    if full:
        fc, risk, source = _live_walk_forward(workers)
        files: list[dict] = []
    else:
        f, rf = paths.LIVE_RESULTS / "forecasts.parquet", paths.LIVE_RESULTS / "risk.parquet"
        if not f.exists():
            raise SkipCheck("no live walk-forward under data/live/forecasts/ (run `python -m volrisk_live forecast`, "
                            "or verify --full to re-run it in memory)")
        fc = pd.read_parquet(f)
        risk = pd.read_parquet(rf) if rf.exists() else pd.DataFrame(columns=[*RISK_KEYS, *RISK_VALUES])
        files = [file_record(f), file_record(rf)]
        source = f"{_rel(f)}, written by the last forecast run (its own reproduction of the stored results: 5b)"
    ref = fc[[*FC_KEYS, "n_t", "F"]].assign(origin=pd.to_datetime(fc["origin"]).dt.strftime("%Y-%m-%d"))
    ref = ref.drop_duplicates(FC_KEYS, keep="last").set_index(FC_KEYS)
    rref = risk[[*RISK_KEYS, *RISK_VALUES]].assign(date=pd.to_datetime(risk["date"]).dt.strftime("%Y-%m-%d"))
    rref = rref.drop_duplicates(RISK_KEYS, keep="last").set_index(RISK_KEYS)
    inputs = payload_inputs(items)
    out, problems, revised = [], [], []
    counts = {"equal": 0, "different": 0, "missing": 0, "n_t_realised_differs": 0, "inputs_revised": 0,
              "window_mismatch": 0, "risk_equal": 0, "risk_different": 0, "risk_missing": 0, "risk_inputs_revised": 0}
    for p, payload in items:
        if payload is None:
            problems.append(f"{p.name}: not valid JSON")
            continue
        for r in payload.get("forecasts", []):
            key = (r.get("asset"), r.get("horizon"), r.get("model"), str(r.get("origin"))[:10])
            row = {"payload": p.name, "asset": key[0], "horizon": key[1], "model": key[2], "origin": key[3],
                   "n_t": r.get("n_t"), "F": r.get("F"), "inputs_unchanged": inputs.get((p.name, key[0]))}
            first, last, n = ex_ante_window(key[0], key[1], date.fromisoformat(key[3]))
            if (str(r.get("window_first"))[:10], str(r.get("window_last"))[:10], int(r.get("n_t"))) != \
                    (first.isoformat(), last.isoformat(), n):
                counts["window_mismatch"] += 1
                problems.append(f"{p.name} {' '.join(key)}: window {r.get('window_first')}..{r.get('window_last')} "
                                f"n_t {r.get('n_t')} != ex-ante {first}..{last} n_t {n}")
            if key not in ref.index:
                row["status"] = "missing"
            else:
                wf = ref.loc[key]
                row.update(F_walk_forward=float(wf["F"]), n_t_walk_forward=int(wf["n_t"]))
                if int(wf["n_t"]) != int(r.get("n_t")):
                    row["status"] = "n_t_realised_differs"  # realised sessions differ from the schedule
                else:
                    row["status"] = "equal" if np.isclose(float(r.get("F")), float(wf["F"]), rtol=REPRO_RTOL,
                                                          atol=0.0) else "different"
            if row["status"] in ("missing", "different") and row["inputs_unchanged"] is False:
                row["status"] = "inputs_revised"  # the source rows it used changed since the run: not re-derivable
                revised.append(f"{p.name} {' '.join(key)}")
            counts[row["status"]] += 1
            if row["status"] in ("missing", "different"):
                problems.append(f"{p.name} {' '.join(key)}: {row['status']} (payload F {r.get('F')}, walk-forward "
                                f"{row.get('F_walk_forward')})")
            out.append(row)
        for r in payload.get("risk", []):
            key = (r.get("asset"), r.get("model"), str(r.get("date"))[:10])
            if key not in rref.index:
                st = "risk_missing"
            else:
                wf = rref.loc[key]
                st = "risk_equal" if all(np.isclose(float(r.get(v)), float(wf[v]), rtol=REPRO_RTOL, atol=0.0)
                                         for v in RISK_VALUES) else "risk_different"
            if st != "risk_equal" and inputs.get((p.name, key[0])) is False:
                st = "risk_inputs_revised"
                revised.append(f"{p.name} VaR/ES {' '.join(key)}")
            counts[st] += 1
            if st in ("risk_missing", "risk_different"):
                problems.append(f"{p.name} VaR/ES {' '.join(key)}: {st.removeprefix('risk_')}")
    n_rows = len(out)
    checked = [v for v in inputs.values() if v is not None]
    ev = {"summary": (f"{counts['equal']} of {n_rows} recorded forecasts and {counts['risk_equal']} VaR/ES rows "
                      f"re-derived exactly from {len(items)} payloads ({counts['n_t_realised_differs']} not "
                      "comparable: realised sessions differ from the ex-ante schedule); windows = session calendar; "
                      f"recorded input data unchanged for {sum(checked)} of {len(checked)} payload x asset"),
          "reference": source, "files": files, "counts": counts, "rtol": REPRO_RTOL, "atol": 0.0,
          "inputs_changed_since_run": sorted(f"{k[0]} {k[1]}" for k, v in inputs.items() if v is False)[:20],
          "rows": [r for r in out if r["status"] != "equal"][:60], "problems": problems[:20]}
    if problems:
        return "FAIL", ev, problems[0]
    if revised:
        raise SkipCheck(f"{len(revised)} recorded value(s) cannot be re-derived: the source rows they used were "
                        f"revised after the run (rows_sha256 differs), e.g. {revised[0]}", ev)
    return "PASS", ev, None


# --------------------------------------------------------------------------------------------- plan, report
def plan(full: bool = False, sample: int = 30, seed: int | None = None, network: bool = True, workers: int = 10,
         seed_info: dict | None = None) -> list[tuple[str, str, str, CheckFn]]:
    """The checks of §7 in report order: ``(id, title, method, fn)``. ``seed=None`` draws the re-download seed when
    3b runs (``draw_seed``)."""
    cache: dict = {}

    def hist() -> tuple[pd.DataFrame, str]:
        if "h" not in cache:
            cache["h"] = load_history()
        return cache["h"]

    def resample() -> tuple:
        if not network:
            raise SkipCheck("network disabled")
        s, info = (seed, seed_info or {"source": "given (--seed)"}) if seed is not None else draw_seed(network)
        return resample_raw(sample, s, seed_info=info)

    net = "" if network else " (network disabled)"
    seed_txt = f"seed {seed}" if seed is not None else "a seed drawn at run time from the newest Bitcoin block hash"
    return [
        ("1", "Frozen code unchanged",
         "Recompute the SHA-256 of every frozen code, config, data and result file and compare with SEALED.json and "
         "with the code hash written into holdout_log.jsonl when the holdout was opened.", check_seal),
        ("2", "Holdout opened exactly once",
         "Read holdout_log.jsonl and check it holds a single opening, made after the seal, that recorded exactly the "
         "sealed hashes, and that SEALED.json has not changed since.", check_opened_once),
        ("3a", "Binance zips match Binance CHECKSUM files",
         "Recompute the SHA-256 of every downloaded Binance zip and compare it with the checksum file Binance serves "
         f"for it now (downloaded again) and with the local copy of that file{net}.",
         lambda: check_binance_checksums(network=network)),
        ("3b", "Random raw files re-downloaded byte-identical",
         f"Draw {sample} raw files at random ({seed_txt}, printed below; half Binance, half Dukascopy), download them "
         f"again from the original public URLs and check the bytes are identical (same SHA-256) to the local "
         f"copies{net}.", resample),
        ("3c", "Raw files unchanged since download",
         "Recompute the SHA-256 of every raw file and compare it with the hash recorded in data/raw/manifest.parquet "
         "when it was downloaded, and check no raw file on disk is missing from that record (a local consistency "
         "check: 3a and 3b compare with the sources).", check_raw_manifest),
        ("4a", "BTC returns vs Coinbase BTC-USD",
         f"Download Coinbase BTC-USD daily candles (another exchange) and correlate their day-to-day returns with "
         f"our Binance-based BTC returns; PASS if correlation >= {CORR_MIN['BTC']}.",
         lambda: check_crypto("BTC", *hist(), network=network)),
        ("4b", "ETH returns vs Coinbase ETH-USD",
         f"Same comparison for ETH with Coinbase ETH-USD; PASS if correlation >= {CORR_MIN['ETH']}.",
         lambda: check_crypto("ETH", *hist(), network=network)),
        ("4c", "S&P 500 returns vs FRED SP500",
         f"Correlate our S&P 500 (Dukascopy CFD) close-to-close returns with the official S&P 500 closes published "
         f"by FRED; PASS if correlation >= {CORR_MIN['SPX']}.", lambda: check_spx(*hist())),
        ("4d", "EUR/USD vs ECB reference rate",
         f"Take our EUR/USD price at 14:15 Frankfurt time, when the ECB fixes its reference rate, and correlate its "
         f"daily returns with the ECB rate (PASS if >= {CORR_MIN['EURUSD']}); our 17:00 New York close is compared "
         "too, for information.", lambda: check_eurusd(*hist(), network=network)),
        ("5a", "Live data tables equal sealed gold",
         "Compare the daily tables last built from the raw files by `update` (data/live/; build time and file hashes "
         "below) with the sealed gold tables, value by value; this compares files - the rebuild from the raw files is "
         "re-done only by --full (5g).", check_live_gold),
        ("5b", "Live forecasts reproduce sealed forecasts",
         "Compare the walk-forward file written by the last live forecast run (data/live/forecasts/) with every "
         f"forecast stored by the development and holdout runs (relative tolerance {REPRO_RTOL:g}); this compares "
         "files - the walk-forward is re-run only by --full (5f).", check_live_forecasts),
        ("5c", "Dev evaluation tables recomputed exactly",
         "Recompute every development evaluation table (leaderboard, Diebold-Mariano, MCS, ...) from the stored "
         "development forecasts (sealed) and check it equals the stored table exactly.",
         lambda: check_eval_tables("dev")),
        ("5d", "Holdout evaluation tables recomputed exactly",
         "Recompute every holdout evaluation table (including the bootstrap confidence intervals) from the stored "
         "holdout forecasts and check it equals the stored table exactly - internal consistency only: those files "
         "are not in SEALED.json, so their hashes are compared with the ones the first ledger payload recorded, and "
         "only --full (5f) re-derives them.", lambda: check_eval_tables("holdout")),
        ("5e", "Full dev walk-forward reproduced",
         "Re-run the complete development walk-forward of all models from scratch, without holdout data, and compare "
         "every forecast with data/results/forecasts.parquet (only with --full; about 7 minutes).",
         (lambda: check_full_dev(workers)) if full else (lambda: _skip("only with --full"))),
        ("5f", "Full holdout run reproduced",
         "Re-do the one-time holdout run in memory from the sealed data with the frozen code and compare every "
         "forecast, target, VaR/ES value and evaluation table with data/results/holdout/, row by row in both "
         "directions (only with --full; about 10 minutes).",
         (lambda: check_full_holdout(workers)) if full else (lambda: _skip("only with --full"))),
        ("5g", "Gold tables rebuilt from raw files",
         "Rebuild the daily gold and implied-volatility tables from data/raw with the frozen builders in a temporary "
         "folder and compare them with the sealed tables (only with --full).",
         check_rebuild_gold if full else (lambda: _skip("only with --full"))),
        ("6a", "Leakage test: every frozen model and COMBO clean",
         f"Randomly change all data after a cut-off ({LEAK_T0}, and the last refit date before it of every model that "
         "is re-estimated periodically) and re-run all ten frozen forecast models and COMBO: a model that does not "
         "look into the future gives the same forecasts at every date up to the cut-off.",
         lambda: check_leak_frozen(workers)),
        ("6b", "Leakage test catches cheating models",
         "Run the same test on three deliberately cheating models (one copies the realised answer, one peeks at the "
         "next day's variance, one trains on a label whose window has not ended): the test must catch all three.",
         check_leak_cheaters),
        ("6c", "Pure-noise model ranks last, outside 90% MCS",
         "Add a model that outputs random noise to the BTC 1-day development comparison: it must have the worst "
         "loss and be excluded from the 90% Model Confidence Set.", check_noise),
        ("7", "No tuning after seeing results",
         "Compare the model settings sealed in config/frozen.yaml (and config/config.yaml, which the code reads) with "
         "the values written in docs/SPEC.md, and show the better LightGBM setting found later was not adopted.",
         check_no_tuning),
        ("8a", "Forward-test ledger hash chain intact",
         "Recompute the hash chain of forecasts/ledger.jsonl (each entry holds the hash of its forecast file and of "
         "the previous entry) and check no forecast file, proof or score exists outside it; an edited, inserted or "
         "reordered entry breaks it, but runs dropped from the end with all their files show only against a head "
         "hash published elsewhere (the owner's push of forecasts/), and UTC days without a run are listed.",
         check_ledger),
        ("8b", "Forecast payloads timestamped (OpenTimestamps)",
         "Check every forecast file has an OpenTimestamps proof of its exact bytes and that confirmed proofs sit in "
         "a Bitcoin block mined (allowing 2 hours of block-time error) before the first target session closed, and "
         "show how much of each target window had passed by then (the .ots files can also be checked at "
         f"opentimestamps.org){net}.", lambda: check_timestamps(network)),
        ("8c", "Ledger entries pinned in Bitcoin",
         "Verify each ledger entry's own timestamp proof against the Bitcoin block chain: an entry pinned long after "
         "its run means the chain may have been rewritten from there on (for example to drop a run)"
         f"{net}.", lambda: check_anchors(network)),
        ("8d", "Recorded forecasts re-derived",
         "Look up every forecast and VaR/ES value recorded in the ledger in the frozen walk-forward at the same "
         "origin (" + ("re-run in memory" if full else "the last live forecast run's file; re-run in memory with "
                                                         "--full") + ") and check its target window against the "
         "session calendar, so a recorded number cannot have been changed or chosen later.",
         lambda: check_payload_forecasts(full, workers)),
    ]


def _skip(reason: str) -> tuple:
    raise SkipCheck(reason)


def run_all(full: bool = False, sample: int = 30, seed: int | None = None, network: bool = True, workers: int = 10,
            now_utc: datetime | None = None, write: bool = True) -> dict:
    """Run every §7 check, write ``reports/live/verification.md`` + ``.json`` (unless ``write`` is False) and return
    ``{schema, generated_utc, args, overall, counts, rules, seconds, checks}``; ``overall`` is FAIL iff any check
    failed (SKIPPED does not fail). ``seed=None`` draws the re-download seed at run time (recorded in ``args``)."""
    t0 = time.perf_counter()
    generated = _iso(now_utc.astimezone(UTC) if now_utc else datetime.now(UTC))
    seed_info = {"source": "given (--seed)"} if seed is not None else None
    if seed is None and network:
        seed, seed_info = draw_seed(network)
    checks = [run_check(*spec) for spec in plan(full, sample, seed, network, workers, seed_info)]
    counts = {s: sum(c["status"] == s for c in checks) for s in STATUSES}
    result = {
        "schema": REPORT_SCHEMA,
        "generated_utc": generated,
        "args": {"full": full, "sample": sample, "seed": seed, "seed_source": seed_info, "network": network,
                 "workers": workers},
        "overall": "FAIL" if counts["FAIL"] else "PASS",
        "counts": counts,
        "rules": list(RULES),
        "seconds": round(time.perf_counter() - t0, 1),
        "checks": checks,
    }
    if write:
        write_report(result)
    return result


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, list) and all(not isinstance(x, (dict, list)) for x in v):
        return ", ".join(_fmt(x) for x in v) if v else "(none)"
    if isinstance(v, (dict, list)):
        return json.dumps(v, ensure_ascii=False, default=str)
    return str(v).replace("|", "\\|").replace("\n", " ")


def _table(rows: list[dict], max_rows: int = 60) -> list[str]:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |" for r in rows[:max_rows]]
    if len(rows) > max_rows:
        out.append(f"\n({len(rows) - max_rows} more rows in verification.json)")
    return out


def _evidence_md(ev: dict, depth: int = 0) -> list[str]:
    pad = "  " * depth
    lines: list[str] = []
    for k, v in ev.items():
        if k == "summary":
            continue
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            lines += [f"{pad}- **{k}** ({len(v)} rows):", "", *_table(v), ""]
        elif isinstance(v, dict) and v:
            lines.append(f"{pad}- **{k}**:")
            lines += _evidence_md(v, depth + 1)
        else:
            lines.append(f"{pad}- {k}: {_fmt(v)}")
    return lines


def render_markdown(result: dict) -> str:
    c = result["counts"]
    args = result.get("args") or {}
    seed_line = []
    if args.get("seed") is not None:
        src = (args.get("seed_source") or {}).get("source", "given")
        seed_line = [f"Re-download sample seed: {args['seed']} ({src}); `--seed {args['seed']}` repeats this draw.", ""]
    lines = [
        "# Verification report",
        "",
        f"Generated {result['generated_utc']} by `uv run python -m volrisk_live verify"
        f"{' --full' if args.get('full') else ''}` (docs/LIVE_SPEC.md §7, `src/volrisk_live/verify.py`). "
        "Every number below is recomputed from the files in this repository and from public sources; run the "
        "command again to reproduce it. Machine-readable copy: `verification.json`.",
        "",
        f"**Overall: {result['overall']}** - {c['PASS']} PASS, {c['FAIL']} FAIL, {c['SKIPPED']} SKIPPED "
        f"({result['seconds']} s).",
        "",
        *seed_line,
        "## Pass rules (fixed in the code before any comparison was run)",
        "",
        *[f"- {r}" for r in result["rules"]],
        "",
        "## Summary",
        "",
        "| # | check | status | result |",
        "|---|---|---|---|",
    ]
    for ch in result["checks"]:
        note = (ch["evidence"] or {}).get("summary") if isinstance(ch["evidence"], dict) else None
        note = note or ch["reason"] or ""
        if ch["reason"] and ch["reason"] not in note:
            note = f"{note} ({ch['reason']})" if note else ch["reason"]
        lines.append(f"| {ch['id']} | {ch['title']} | **{ch['status']}** | {_fmt(note)} |")
    lines += ["", "## Details", ""]
    for ch in result["checks"]:
        lines += [f"### {ch['id']}. {ch['title']} - {ch['status']}", "", f"*How:* {ch['method']}", ""]
        if ch["reason"]:
            lines += [f"*{'Reason' if ch['status'] != 'PASS' else 'Note'}:* {ch['reason']}", ""]
        if isinstance(ch["evidence"], dict) and ch["evidence"].get("summary"):
            lines += [f"*Result:* {ch['evidence']['summary']}", ""]
        if isinstance(ch["evidence"], dict) and ch["evidence"]:
            lines += _evidence_md(ch["evidence"])
        lines += ["", f"_{ch['seconds']} s_", ""]
    return "\n".join(lines).rstrip() + "\n"


def write_report(result: dict, md_path: Path | None = None, json_path: Path | None = None) -> tuple[Path, Path]:
    md_path = paths.VERIFY_MD if md_path is None else Path(md_path)
    json_path = paths.VERIFY_JSON if json_path is None else Path(json_path)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    clean = _plain(result)
    json_path.write_text(json.dumps(clean, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    md_path.write_text(render_markdown(clean), encoding="utf-8")
    return md_path, json_path


# --------------------------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m volrisk_live.verify", description="LIVE_SPEC §7 verification report")
    ap.add_argument("--full", action="store_true",
                    help="also re-run the dev and holdout walk-forwards and rebuild the gold tables (~25 min)")
    ap.add_argument("--sample", type=int, default=30, help="raw files to re-download (default 30)")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed of the re-download sample (default: drawn from the newest Bitcoin block hash)")
    ap.add_argument("--no-network", action="store_true", help="skip every check that needs the internet")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    res = run_all(full=a.full, sample=a.sample, seed=a.seed, network=not a.no_network, workers=a.workers)
    for ch in res["checks"]:
        line = f"{ch['status']:<8}{ch['id']:<4}{ch['title']}" + (f"  ({ch['reason']})" if ch["reason"] else "")
        print(line.encode("ascii", "replace").decode("ascii"))
    print(f"overall: {res['overall']} -> {_rel(paths.VERIFY_MD)}")
    return 1 if res["overall"] == "FAIL" else 0


if __name__ == "__main__":
    sys.exit(main())
