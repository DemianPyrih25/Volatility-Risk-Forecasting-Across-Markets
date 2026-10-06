"""Live data update and the consistency proof (docs/LIVE_SPEC.md §2).

``update`` moves the frozen data end to the last complete UTC day (``context.live_end``), lets the frozen
downloaders **add** the new raw files to the shared ``data/raw`` store, and rebuilds bronze -> 5-minute bars ->
gold -> implied vol with the frozen builders into ``data/live/`` (their ``out_dir``-style parameters; nothing is
written to a sealed location). ``check_against_sealed`` then proves that the rebuild reproduces the sealed dev gold
and every sealed holdout row bit for bit (plus the sealed implied-vol rows), so the live forecasts start from
exactly the history that was evaluated; any difference raises :class:`LiveDataMismatch`.

Notes on the frozen builders (all called with defaults except the directories and the end date):

- ``binance.build_bronze`` caps at ``end`` (default ``C.data_end()``); ``dukascopy.build_bronze`` has no date
  window and mirrors every raw day file on disk — later days only reach bronze, the bars/gold builders cut at
  ``end`` through the session schedule and ``C.data_end()``.
- ``bars.build_bars`` keeps its build marker under ``{out_dir}/_build/``, so the live silver has its own markers
  (keyed on the end date, schedule and bronze hashes): the first live build is complete, a re-run without new
  data is skipped.
- The downloaders never request the current UTC day. Binance zips already on disk are never fetched again and a
  Binance 404 is asked again on every run. The frozen Dukascopy downloader (``http.fetch_many(skip_done=True)``)
  never asks again for a URL the manifest records as ``ok``, ``missing`` (404) or ``empty``, and Dukascopy
  answers 404 for a day it has not published yet. So :func:`refetch_live_window` first requests again the day
  files after the frozen end whose recorded answer may predate their publication (never a sealed day). Implied
  snapshots are reused when one from the same UTC day exists.
- :func:`completeness` lists, per asset, the scheduled sessions after the frozen end that have no live gold row
  and the raw day files they need that are not ``ok`` (state ``missing_sessions`` / ``incomplete_assets``);
  ``stale_assets`` only looks at the last session.
- ``live_end`` is an in-memory override, so everything that depends on the data end runs in this process.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections import Counter
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from volrisk import bars, holdout, io, measures, sessions
from volrisk import config as C
from volrisk.data import binance, dukascopy, http, implied
from volrisk_live import paths as P
from volrisk_live.context import frozen_data_end, live_end, require_opened_holdout

log = logging.getLogger(__name__)

STATE_SCHEMA = "volrisk-live-state/1"
FUTURE_SLACK = timedelta(minutes=5)  # tolerated clock skew for an explicit ``now_utc``
# A Dukascopy day file fetched less than this long after its UTC day ended is requested again (it may have been
# fetched before the day was published or complete); so is one recorded as ``missing`` (404) or ``empty``.
SETTLE = timedelta(hours=2)
REFETCH_STATUSES = ("missing", "empty")  # recorded answers the frozen downloader never asks again (skip_done)
GOLD_KEYS = ("asset", "session_date")
IMPLIED_KEYS = ("asset", "origin")
N_EXAMPLES = 5  # differing cells / rows shown per column in a mismatch report
# Sealed files the live rebuild is compared with, by their SEALED.json hash key (read at call time).
_SEALED_BASE = {"dev_data": "DAILY_DEV", "holdout_data": "DAILY_HOLDOUT", "implied": "IMPLIED"}


class LiveDataMismatch(RuntimeError):
    """The live rebuild differs from the sealed data (data drift or a changed source). ``report`` has the details."""

    def __init__(self, message: str, report: dict):
        super().__init__(message)
        self.report = report


# --------------------------------------------------------------------------------------------- dates


def _wall_clock() -> datetime:
    """Current UTC time (one place, so tests can move the clock)."""
    return datetime.now(timezone.utc)


def _as_utc(now: datetime | str | None) -> datetime:
    if now is None:
        return _wall_clock()
    if isinstance(now, str):
        now = datetime.fromisoformat(now.strip().replace("Z", "+00:00"))
    if now.tzinfo is None:
        raise ValueError("now_utc must be timezone-aware (UTC)")
    return now.astimezone(timezone.utc)


def live_end_date(now_utc: datetime | str) -> date:
    """Last complete UTC day before ``now_utc``: ``now_utc.date() - 1 day`` (LIVE_SPEC §2).

    Binance daily zips and Dukascopy day files are published after the UTC day ends; the FX (17:00 New York)
    and XNYS sessions of that day close before 24:00 UTC.
    """
    return _as_utc(now_utc).date() - timedelta(days=1)


# --------------------------------------------------------------------------------------------- update


def _assets(assets: Sequence[str]) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(assets))
    bad = [a for a in out if a not in C.ASSETS]
    if bad or not out:
        raise ValueError(f"assets must be a non-empty subset of {C.ASSETS}, got {list(assets)}")
    return out


def _source(asset: str):
    return binance if C.asset(asset).source == binance.SOURCE else dukascopy


def update(
    now_utc: datetime | str | None = None,
    assets: Sequence[str] = C.ASSETS,
    download: bool = True,
    *,
    check: bool = True,
) -> dict:
    """Download new raw data and rebuild the live tables up to the last complete UTC day (LIVE_SPEC §2).

    Under ``live_end(end)``: the frozen downloaders, each Dukascopy one preceded by :func:`refetch_live_window`
    (``download=False`` skips both), then ``build_bronze -> build_bars -> build_gold`` into ``data/live/`` and
    ``build_implied`` into ``data/live/implied.parquet``. With ``check`` the rebuild is compared with the sealed
    data (:func:`check_against_sealed`). The summary is written to ``data/live/state.json`` and returned; on a
    failed check the state records the failure and :class:`LiveDataMismatch` is re-raised. A failed implied-vol
    download is recorded and the newest existing snapshots are used. A failed Binance/Dukascopy download shows up
    in ``completeness`` / ``missing_sessions`` / ``incomplete_assets`` (any scheduled session after the frozen end
    without a gold row, or a raw file it needs that is not ``ok``) and, when it is the last session, as an older
    ``last_sessions`` entry (``stale_assets``).
    """
    now = _as_utc(now_utc)
    if now > _wall_clock() + FUTURE_SLACK:
        raise ValueError(f"now_utc {now.isoformat()} is in the future")
    end = live_end_date(now)
    assets = _assets(assets)
    opened = require_opened_holdout()  # the rebuild holds holdout-period rows
    for p in (P.LIVE_BRONZE, P.LIVE_FLAGS, P.LIVE_SILVER, P.LIVE_GOLD, P.LIVE_HOLDOUT):
        p.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}

    def timed(name: str, fn, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            timings[name] = round(time.perf_counter() - t0, 2)
            log.info("live update: %s done in %.1fs", name, timings[name])

    state: dict = {
        "schema": STATE_SCHEMA,
        "started_utc": _wall_clock(),
        "now_utc": now,
        "end": end,
        "frozen_data_end": frozen_data_end(),
        "holdout_start": C.holdout_start(),
        "holdout_opened_utc": opened.get("utc"),
        "assets": list(assets),
    }
    log.info("live update: end %s (now %s), assets %s", end, now.isoformat(), list(assets))
    with live_end(end):
        if download:
            dl: dict = {"refetch": {}}
            for a in assets:
                if _source(a) is dukascopy:  # before the frozen download, which then skips these URLs
                    dl["refetch"][a] = timed(f"refetch_{a}", refetch_live_window, a, end)
                dl[a] = timed(f"download_{a}", _source(a).download, a)
            try:
                snaps = timed("download_implied", implied.download_all)
                dl["implied"] = {k: (p.name if p is not None else None) for k, p in snaps.items()}
            except Exception as exc:  # noqa: BLE001 — network/API failure: keep the newest existing snapshots
                log.warning("live update: implied download failed (%r); using the newest existing snapshots", exc)
                dl["implied"] = {"error": repr(exc)}
            state["download"] = dl
        else:
            state["download"] = None

        bronze_sum, bars_sum = {}, {}
        for a in assets:
            src = _source(a)
            kw = {"end": end} if src is binance else {}
            bronze_sum[a] = timed(f"bronze_{a}", src.build_bronze, a, out_dir=P.LIVE_BRONZE,
                                  flags_dir=P.LIVE_FLAGS, **kw)
            bars_sum[a] = timed(f"bars_{a}", bars.build_bars, a, bronze_dir=P.LIVE_BRONZE.parent,
                                out_dir=P.LIVE_SILVER, end=end)
        gold_sum = timed("gold", measures.build_gold, assets, silver_dir=P.LIVE_SILVER, gold_dir=P.LIVE_GOLD,
                         holdout_dir=P.LIVE_HOLDOUT)
        iv = timed("implied", implied.build_implied, out_path=P.LIVE_IMPLIED)

    state["bronze"] = {a: {k: v for k, v in s.items() if not k.endswith("_by_month")} for a, s in bronze_sum.items()}
    state["bars"] = bars_sum
    state["gold"] = gold_sum
    state["implied"] = {
        a: {"rows": int(g.height), "first_origin": g["origin"].min(), "last_origin": g["origin"].max()}
        for (a,), g in sorted(iv.partition_by("asset", as_dict=True).items())
    }
    last = last_sessions()
    expected = {a: _expected_last_session(a, end) for a in last}
    state["last_sessions"] = last
    state["expected_last_sessions"] = expected
    state["stale_assets"] = [a for a in assets if a not in last or (expected[a] is not None and last[a] < expected[a])]
    comp = completeness(assets, end)
    state["completeness"] = comp
    state["missing_sessions"] = {a: c["missing_sessions"] for a, c in comp.items() if c["missing_sessions"]}
    state["incomplete_assets"] = [a for a in assets if comp[a]["missing_sessions"] or comp[a]["raw_not_ok"]]
    for a in state["incomplete_assets"]:
        c = comp[a]
        log.warning("live update: %s incomplete after the frozen end — %d of %d scheduled sessions missing %s; "
                    "raw files not ok: %s", a, len(c["missing_sessions"]), c["scheduled"],
                    [d.isoformat() for d in c["missing_sessions"][:10]],
                    [f"{r['file']} ({r['status']})" for r in c["raw_not_ok"][:10]])
    state["files"] = {name: {"path": p, "sha256": holdout.file_sha(p)} for name, p in (
        ("live_dev_gold", P.LIVE_DAILY_DEV), ("live_holdout_gold", P.LIVE_DAILY_HOLDOUT),
        ("live_implied", P.LIVE_IMPLIED))}
    state["timings_s"] = timings

    error: LiveDataMismatch | None = None
    if check:
        t0 = time.perf_counter()
        try:
            state["check"] = check_against_sealed(assets)
        except LiveDataMismatch as exc:
            error = exc
            state["check"] = {**exc.report, "ok": False, "error": str(exc)}
        timings["check"] = round(time.perf_counter() - t0, 2)
    else:
        state["check"] = None
    state["finished_utc"] = _wall_clock()
    state["total_s"] = round(sum(timings.values()), 2)
    _write_state(state)
    log.info("live update: end %s, last sessions %s, incomplete %s, check %s (%.0fs)", end,
             {a: d.isoformat() for a, d in last.items()}, state["incomplete_assets"] or "none",
             "skipped" if not check else ("FAILED" if error else "passed"), state["total_s"])
    if error is not None:
        raise error
    return state


def _expected_last_session(asset: str, end: date) -> date | None:
    """Last *scheduled* session of ``asset`` on or before ``end`` (calendar only, never the data)."""
    sched = sessions.session_schedule(asset, end - timedelta(days=14), end)
    return sched["session_date"].max() if sched.height else None


# --------------------------------------------------------------------------------------------- live window raw files


def _day_end(day: date) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc) + timedelta(days=1)


def _manifest_records(urls: Sequence[str]) -> dict[str, tuple[str, datetime | None]]:
    """``{url: (status, fetched_at)}`` from the shared download manifest, for ``urls`` only."""
    m = http.read_manifest().filter(pl.col("url").is_in(list(urls)))
    return {u: (s, t) for u, s, t in m.select("url", "status", "fetched_at").iter_rows()}


def refetch_live_window(asset: str, end: date) -> dict:
    """Request again the Dukascopy day files after the frozen end whose recorded answer may predate publication.

    The frozen ``dukascopy.download`` goes through ``http.fetch_many(skip_done=True)``, which never asks again for
    a URL the manifest records as ``missing`` (404) or ``empty``. Dukascopy answers 404 for a day it has not
    published yet, so one early run would lose that day for good. For the days in ``(frozen data_end, end]``
    (never a sealed day), and with the frozen job list, this fetches again with ``skip_done=False`` every BID/ASK
    file recorded as ``missing``/``empty`` or as ``ok`` but fetched less than :data:`SETTLE` after its UTC day
    ended. Files never requested are left to the frozen download, which runs next; so is ``error``, which it
    retries itself. Returns the counts and the files that are still not ``ok``.
    """
    if _source(asset) is not dukascopy:
        raise ValueError(f"{asset} is not a Dukascopy asset")
    first = frozen_data_end() + timedelta(days=1)
    with live_end(end):  # dukascopy.jobs caps at C.data_end() and at yesterday (UTC)
        js = dukascopy.jobs(asset, first, end)
    rec = _manifest_records([j.url for j in js])

    def unsettled(j: http.Job) -> bool:
        status, fetched = rec.get(j.url, (None, None))
        if status in REFETCH_STATUSES:
            return True
        day = date.fromisoformat(j.local_path.name[:10])  # {YYYY-MM-DD}_{SIDE}.bi5
        return status == "ok" and fetched is not None and fetched < _day_end(day) + SETTLE

    retry = [j for j in js if unsettled(j)]
    rows = http.fetch_many(retry, skip_done=False) if retry else []
    counts = Counter(r.status for r in rows)
    out = {"asset": asset, "first": first, "end": end, "jobs": len(js), "requested": len(retry),
           **{s: counts.get(s, 0) for s in dukascopy.STATUSES},
           "not_ok": sorted(f"{Path(r.local_path).name} ({r.status})" for r in rows if r.status != "ok")}
    if retry:
        log.info("live update: re-requested %d %s day file(s) after the frozen end: %s", len(retry), asset, out)
    return out


def _utc_days(open_utc: datetime, close_utc: datetime) -> list[date]:
    """UTC days holding the minutes of a session ``[open_utc, close_utc)`` (minute open times)."""
    lo, hi = open_utc.date(), (close_utc - timedelta(minutes=1)).date()
    return [lo + timedelta(days=k) for k in range((hi - lo).days + 1)]


def _raw_status(asset: str, day: date, rec: dict[str, tuple[str, datetime | None]]) -> list[tuple[str, str]]:
    """``(file, status)`` per raw file of ``day``: ``ok``, the manifest status, ``not_requested`` or ``absent``."""
    sym = C.asset(asset).symbol
    if _source(asset) is binance:  # local zips are the cache: the monthly zip if present, else the daily one
        zips = [binance.raw_path(binance.monthly_url(sym, day.year, day.month)),
                binance.raw_path(binance.daily_url(sym, day))]
        ok = any(z.exists() and binance.checksum_path(z).exists() for z in zips)
        return [(zips[1].name, "ok" if ok else "absent")]
    out = []
    for s in dukascopy.SIDES:
        p, u = dukascopy.raw_path(sym, day, s), dukascopy.url(sym, day, s)
        status = rec.get(u, ("not_requested", None))[0]
        out.append((p.name, "absent" if status == "ok" and not p.exists() else status))
    return out


def completeness(assets: Sequence[str] = C.ASSETS, end: date | None = None) -> dict[str, dict]:
    """Completeness of the live data after the frozen end, per asset (calendar and raw files, never the models).

    For the scheduled sessions in ``(frozen data_end, end]`` (``end`` defaults to the last update's end):
    ``missing_sessions`` = those without a live gold row (a raw file not published yet, a failed download, or a
    session dropped by the frozen validity rules, as for 13 SPX dev sessions); ``raw_not_ok`` = the raw files
    these sessions need (Dukascopy BID+ASK of each UTC day the session spans, Binance the zip of its day) that are
    not ``ok``, with the status (manifest status, ``not_requested`` or ``absent``) and the sessions that need them.
    ``stale_assets`` only sees a gap at the end; this also sees one in the middle, which no later run fills
    unless its raw file arrives.
    """
    require_opened_holdout()
    assets = _assets(assets)
    if end is None:
        last_end = read_state().get("end")
        if not last_end:
            raise FileNotFoundError(f"{P.LIVE_STATE} has no end date; pass end or run update() first")
        end = date.fromisoformat(last_end)
    first = frozen_data_end() + timedelta(days=1)
    keys = pl.concat([_read(P.LIVE_DAILY_DEV).select(GOLD_KEYS), _read(P.LIVE_DAILY_HOLDOUT).select(GOLD_KEYS)])
    have = keys.filter(pl.col("session_date") >= first)
    scheds = {a: sessions.session_schedule(a, first, end) for a in assets}
    days = {a: {s: _utc_days(o, c) for s, o, c in sch.select("session_date", "open_utc", "close_utc").iter_rows()}
            for a, sch in scheds.items()}
    rec = _manifest_records([dukascopy.url(C.asset(a).symbol, d, side) for a in assets if _source(a) is dukascopy
                             for ds in days[a].values() for d in ds for side in dukascopy.SIDES])
    out = {}
    for a in assets:
        present = set(have.filter(pl.col("asset") == a)["session_date"].to_list())
        sched = scheds[a]["session_date"].to_list()
        bad: dict[str, dict] = {}
        for s, ds in days[a].items():
            for d in ds:
                for name, status in _raw_status(a, d, rec):
                    if status != "ok":
                        bad.setdefault(name, {"file": name, "day": d, "status": status, "sessions": []})
                        bad[name]["sessions"].append(s)
        out[a] = {"first": first, "end": end, "scheduled": len(sched),
                  "missing_sessions": [s for s in sched if s not in present],
                  "raw_not_ok": sorted(bad.values(), key=lambda r: (r["day"], r["file"]))}
    return out


def read_state() -> dict:
    """The last ``update`` summary (``data/live/state.json``), ``{}`` if there is none yet."""
    try:
        return json.loads(P.LIVE_STATE.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}


def _write_state(state: dict) -> None:
    P.LIVE_STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = P.LIVE_STATE.with_suffix(P.LIVE_STATE.suffix + ".part")
    tmp.write_text(json.dumps(_jsonable(state), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, P.LIVE_STATE)


def _rel(p: Path) -> str:
    try:
        return Path(p).resolve().relative_to(C.ROOT.resolve()).as_posix()
    except ValueError:
        return str(p)


def _jsonable(x):
    """Plain JSON values: dates as ISO strings, paths relative to the project root, NaN/inf as null."""
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return _rel(x)
    if isinstance(x, (datetime, date)):
        return x.isoformat()
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):  # numpy scalars
        return _jsonable(x.item())
    return x


# --------------------------------------------------------------------------------------------- consistency proof


def check_against_sealed(assets: Sequence[str] = C.ASSETS, *, implied_vol: bool = True) -> dict:
    """Prove that the live rebuild reproduces the sealed data exactly (LIVE_SPEC §2); raise otherwise.

    Compared, for ``assets``, on the key ``(asset, session_date)`` and every column of the gold schema:

    - ``data/live/gold/daily.parquet`` vs the sealed dev gold ``data/gold/daily.parquet`` (all rows);
    - ``data/live/holdout/daily.parquet`` rows ``<= data_end`` (frozen, 2026-09-30) vs the sealed holdout gold;
    - with ``implied_vol``: ``data/live/implied.parquet`` rows ``origin <= data_end`` vs ``data/gold/implied.parquet``.

    "Exactly" means: same columns with the schema dtypes on both sides, the same keys (no missing, extra or
    duplicate rows) and every cell identical — floats bit for bit (so ``-0.0 != 0.0``), any NaN equals any NaN,
    null equals null only. The sealed files are first checked against their SEALED.json hashes. Returns the
    evidence (row/cell counts, content SHA-256 of both sides, live rows after the frozen end with the scheduled
    sessions among them that have no row); raises :class:`LiveDataMismatch` with the same report on any
    difference. A missing live session after the frozen end is reported, not raised (nothing sealed to compare).
    """
    require_opened_holdout()
    assets = _assets(assets)
    frozen_end, h0 = frozen_data_end(), C.holdout_start()
    in_assets = pl.col("asset").is_in(list(assets))
    report: dict = {"frozen_data_end": frozen_end, "holdout_start": h0, "assets": list(assets)}
    base = _sealed_base(implied_vol)
    report["sealed_base"] = base

    sealed_dev = pl.read_parquet(io.DAILY_DEV).filter(in_assets)
    live_dev = _read(P.LIVE_DAILY_DEV).filter(in_assets)
    report["dev"] = _compare(sealed_dev, live_dev, measures.GOLD_SCHEMA, GOLD_KEYS)

    sealed_ho = pl.read_parquet(io.DAILY_HOLDOUT).filter(in_assets)
    live_ho_all = _read(P.LIVE_DAILY_HOLDOUT).filter(in_assets)
    upto = pl.col("session_date") <= frozen_end
    report["holdout"] = _compare(sealed_ho.filter(upto), live_ho_all.filter(upto), measures.GOLD_SCHEMA, GOLD_KEYS)
    report["holdout"]["sealed_rows_after_frozen_end"] = sealed_ho.filter(~upto).height  # 0 by construction
    # Not part of the proof (no sealed counterpart), but recorded with it: rows after the frozen end and the
    # scheduled sessions up to each asset's last one that have no row (a gap in the middle; see completeness()).
    report["live_after_frozen_end"] = {}
    for (a,), g in sorted(live_ho_all.filter(~upto).partition_by("asset", as_dict=True).items()):
        last = g["session_date"].max()
        sched = sessions.session_schedule(a, frozen_end + timedelta(days=1), last)["session_date"].to_list()
        have = set(g["session_date"].to_list())
        report["live_after_frozen_end"][a] = {"rows": g.height, "first": g["session_date"].min(), "last": last,
                                              "scheduled": len(sched), "missing": [d for d in sched if d not in have]}
    live_all = pl.concat([df.select(GOLD_KEYS) for df in (live_dev, live_ho_all) if set(GOLD_KEYS) <= set(df.columns)])
    report["live_per_asset"] = {
        a: {"dev_rows": int((g["session_date"] < h0).sum()), "holdout_rows": int((g["session_date"] >= h0).sum()),
            "first": g["session_date"].min(), "last": g["session_date"].max()}
        for (a,), g in sorted(live_all.partition_by("asset", as_dict=True).items())
    }

    if implied_vol:
        sealed_iv = pl.read_parquet(io.IMPLIED).filter(in_assets)
        live_iv = _read(P.LIVE_IMPLIED).filter(in_assets)
        up = pl.col("origin") <= frozen_end
        report["implied"] = _compare(sealed_iv.filter(up), live_iv.filter(up), implied.IMPLIED_SCHEMA, IMPLIED_KEYS)
        report["implied"]["live_rows_after_frozen_end"] = live_iv.filter(~up).height

    problems = [f"sealed {k} ({b['path']}) does not match its SEALED.json hash" for k, b in base.items() if not b["ok"]]
    for part in ("dev", "holdout", "implied"):
        if part in report:
            problems += [f"{part}: {p}" for p in _describe(report[part])]
    report["ok"] = not problems
    report["problems"] = problems
    if problems:
        raise LiveDataMismatch("live rebuild differs from the sealed data — " + "; ".join(problems), report)
    log.info("consistency check passed: dev %d rows, holdout %d rows (<= %s)%s identical to the sealed data",
             report["dev"]["rows_compared"], report["holdout"]["rows_compared"], frozen_end,
             f", implied {report['implied']['rows_compared']} rows" if implied_vol else "")
    return report


def _read(path: Path) -> pl.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} does not exist; run volrisk_live.update.update() first")
    return pl.read_parquet(path)


def _sealed_base(implied_vol: bool) -> dict[str, dict]:
    """SHA-256 of each sealed comparison file vs its hash in SEALED.json (the base of the proof must be sealed)."""
    sealed = json.loads(holdout.SEALED.read_text(encoding="utf-8"))["hashes"]
    out = {}
    for key, attr in _SEALED_BASE.items():
        if key == "implied" and not implied_vol:
            continue
        p = getattr(io, attr)
        sha = holdout.file_sha(p) if p.exists() else "MISSING"
        out[key] = {"path": _rel(p), "sha256": sha, "ok": sha == sealed.get(key)}
    return out


def _frame_sha256(df: pl.DataFrame) -> str:
    """Content hash of a normalised, key-sorted table (CSV text: floats in shortest round-trip form)."""
    return hashlib.sha256(df.write_csv().encode("utf-8")).hexdigest()


def _normalise(df: pl.DataFrame, schema: dict, side: str, problems: list[str]) -> pl.DataFrame:
    """Columns in schema order, cast to the schema; column-set and dtype differences are recorded as problems."""
    missing = [c for c in schema if c not in df.columns]
    extra = [c for c in df.columns if c not in schema]
    if missing or extra:
        problems.append(f"{side} columns differ from the schema (missing {missing}, extra {extra})")
    wrong = {c: str(df.schema[c]) for c in schema if c in df.columns and df.schema[c] != schema[c]}
    if wrong:
        problems.append(f"{side} dtypes differ from the schema: {wrong}")
    cols = []
    for c in schema:
        if c not in df.columns:
            continue
        try:
            cols.append(df[c].cast(schema[c], strict=True))
        except (pl.exceptions.PolarsError, TypeError, ValueError):
            problems.append(f"{side} column {c!r} ({df.schema[c]}) cannot be cast to {schema[c]}")
    return pl.DataFrame(cols)


def _equal(a: pl.Series, b: pl.Series) -> np.ndarray:
    """Cell-wise exact equality: floats bit for bit or both NaN; null equals null only."""
    na, nb = a.is_null().to_numpy(), b.is_null().to_numpy()
    if a.dtype.is_float() and b.dtype.is_float():
        x = np.ascontiguousarray(a.cast(pl.Float64).to_numpy(), dtype=np.float64)
        y = np.ascontiguousarray(b.cast(pl.Float64).to_numpy(), dtype=np.float64)
        same = (x.view(np.int64) == y.view(np.int64)) | (np.isnan(x) & np.isnan(y))
    else:
        same = a.eq_missing(b).to_numpy()
    return np.where(na | nb, na & nb, same)


def _cell(v):
    return repr(v) if isinstance(v, float) else v


def _compare(sealed: pl.DataFrame, live: pl.DataFrame, schema: dict, keys: tuple[str, ...]) -> dict:
    """Exact comparison of two tables on ``keys`` (see :func:`check_against_sealed`); returns the evidence."""
    problems: list[str] = []
    s = _normalise(sealed, schema, "sealed", problems)
    lv = _normalise(live, schema, "live", problems)
    out: dict = {"rows_sealed": sealed.height, "rows_live": live.height, "rows_compared": 0, "cells_compared": 0,
                 "only_sealed": {"n": 0, "examples": []}, "only_live": {"n": 0, "examples": []},
                 "mismatches": {}, "problems": problems}
    if any(k not in s.columns or k not in lv.columns for k in keys):
        problems.append(f"key columns {list(keys)} missing")
        out["equal"] = False
        return out
    for side, df in (("sealed", s), ("live", lv)):
        n_dup = int(df.select(keys).is_duplicated().sum())
        if n_dup:
            problems.append(f"{side} has {n_dup} rows with duplicate keys {list(keys)}")
    s, lv = s.sort(list(keys)), lv.sort(list(keys))
    out["sha256_sealed"], out["sha256_live"] = _frame_sha256(s), _frame_sha256(lv)
    for name, a, b in (("only_sealed", s, lv), ("only_live", lv, s)):
        rows = a.join(b.select(keys), on=list(keys), how="anti")
        out[name] = {"n": rows.height, "examples": rows.select(keys).head(N_EXAMPLES).rows()}
    both = s.join(lv, on=list(keys), how="inner", suffix="_live").sort(list(keys))
    common = [c for c in s.columns if c in lv.columns and c not in keys]
    out["rows_compared"] = both.height
    out["cells_compared"] = both.height * (len(common) + len(keys))
    for c in common:
        a, b = both[c], both[f"{c}_live"]
        eq = _equal(a, b)
        if eq.all():
            continue
        bad = both.filter(pl.Series(~eq))
        info: dict = {"n": int((~eq).sum()), "examples": [
            {**{k: r[k] for k in keys}, "sealed": _cell(r[c]), "live": _cell(r[f"{c}_live"])}
            for r in bad.head(N_EXAMPLES).iter_rows(named=True)]}
        if a.dtype.is_float():
            d = (bad[c] - bad[f"{c}_live"]).abs()
            info["max_abs_diff"] = d.max() if d.drop_nans().drop_nulls().len() else None
        out["mismatches"][c] = info
    out["equal"] = not (problems or out["mismatches"] or out["only_sealed"]["n"] or out["only_live"]["n"])
    return out


def _describe(res: dict) -> list[str]:
    msgs = list(res["problems"])
    for name, what in (("only_sealed", "missing in live"), ("only_live", "only in live")):
        if res[name]["n"]:
            msgs.append(f"{res[name]['n']} row(s) {what}, e.g. {res[name]['examples'][:3]}")
    for c, info in res["mismatches"].items():
        ex = info["examples"][0]
        msgs.append(f"column {c!r} differs in {info['n']} row(s), e.g. "
                    f"{ {k: v for k, v in ex.items() if k not in ('sealed', 'live')} }: "
                    f"sealed {ex['sealed']} vs live {ex['live']}")
    return msgs


# --------------------------------------------------------------------------------------------- live loaders


def live_daily(asset: str | None = None) -> pd.DataFrame:
    """Live gold, dev + holdout + live sessions (gold schema), as pandas sorted by ``(asset, session_date)``.

    Converted exactly like ``volrisk.io.load_daily`` (polars concat -> sort -> ``to_pandas``), so the frozen
    models see the same dtypes as in the sealed runs.
    """
    require_opened_holdout()
    dev = _read(P.LIVE_DAILY_DEV)
    if dev.height and dev["session_date"].max() >= C.holdout_start():
        raise RuntimeError(f"{P.LIVE_DAILY_DEV} contains holdout sessions — the split is broken")
    df = pl.concat([dev, _read(P.LIVE_DAILY_HOLDOUT)], how="vertical_relaxed")
    if asset is not None:
        df = df.filter(pl.col("asset") == asset)
    return df.sort(["asset", "session_date"]).to_pandas()


def live_implied(asset: str | None = None) -> pd.DataFrame:
    """Live implied-vol table up to the live end, as pandas sorted by ``(asset, origin)`` (as ``io.load_implied``)."""
    require_opened_holdout()
    df = _read(P.LIVE_IMPLIED)
    if asset is not None:
        df = df.filter(pl.col("asset") == asset)
    return df.sort(["asset", "origin"]).to_pandas()


def last_sessions() -> dict[str, date]:
    """Last session present in the live gold per asset (``C.ASSETS`` order; assets without rows are omitted)."""
    require_opened_holdout()
    df = pl.concat([_read(P.LIVE_DAILY_DEV).select(GOLD_KEYS), _read(P.LIVE_DAILY_HOLDOUT).select(GOLD_KEYS)])
    last = dict(df.group_by("asset").agg(pl.col("session_date").max()).iter_rows())
    return {a: last[a] for a in C.ASSETS if a in last}
