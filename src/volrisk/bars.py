"""Session-aligned 5-minute bars from bronze minutes, in Polars (SPEC §4.2).

- A bronze minute with open time ``ts`` closes at ``ts + 60s``. The bar labelled ``T`` (its end) holds the
  minutes whose close time lies in ``(T - 5m, T]`` — right-closed and right-labelled.
- Only real minutes count (``is_real``; zero-volume candles are flat fills, not prices). A bar's ``price`` is
  the close of its last real minute, ``null`` if it has none — no forward fill, no interpolation.
- Bars live inside the session window ``(open_utc, close_utc]``; every scheduled bar ``bar_idx = 1..n_sched``
  is a row (``ts_end = open_utc + 5m·bar_idx``), so a null price marks a bar without real minutes.
- Per session: ``p_open`` = open of the first real minute with open time in ``[open_utc, close_utc)``,
  ``p_close`` = close of the last real minute with close time in ``(open_utc, close_utc]``,
  ``coverage = n_real_bars / n_sched``.

Minutes are assigned to sessions with one as-of join on the (sorted, non-overlapping) schedule and to bars
with integer arithmetic, so the whole history of an asset is processed in a few vectorised passes.

SPEC §12 skip rule: :func:`build_bars` records a build marker ``{silver}/_build/bars_{ASSET}.json`` holding
the config hash and SHA-256 fingerprints of everything the outputs depend on (schedule, bronze files, this
code) and of the two outputs; it skips the work while all of them still match (``force=True`` rebuilds).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl

from volrisk import config as C
from volrisk.io import write_parquet
from volrisk.sessions import SCHEDULE_SCHEMA, UTC_US, bar_minutes, session_schedule, utc_us_expr

log = logging.getLogger(__name__)

BARS_SCHEMA = {
    "asset": pl.Utf8,
    "session_date": pl.Date,
    "bar_idx": pl.Int32,
    "ts_end": UTC_US,
    "price": pl.Float64,
    "n_real_min": pl.Int32,
}
SESSIONS_SCHEMA = {
    "asset": pl.Utf8,
    "session_date": pl.Date,
    "open_utc": UTC_US,
    "close_utc": UTC_US,
    "n_sched": pl.Int32,
    "n_real_bars": pl.Int32,
    "coverage": pl.Float64,
    "p_open": pl.Float64,
    "p_close": pl.Float64,
}
_MINUTE = pl.duration(minutes=1)


def minute_dir(asset: str, bronze_dir: Path | str | None = None) -> Path:
    """``{bronze}/minute/asset={ASSET}`` holding ``year=YYYY/part.parquet`` (SPEC §3)."""
    return Path(bronze_dir or C.BRONZE) / "minute" / f"asset={asset}"


def bars_path(asset: str, silver_dir: Path | str | None = None) -> Path:
    return Path(silver_dir or C.SILVER) / "bars5m" / f"asset={asset}" / "part.parquet"


def sessions_path(asset: str, silver_dir: Path | str | None = None) -> Path:
    return Path(silver_dir or C.SILVER) / "sessions" / f"asset={asset}" / "part.parquet"


def marker_path(asset: str, silver_dir: Path | str | None = None) -> Path:
    """Build marker of the asset's silver outputs (kept outside the ``bars5m``/``sessions`` tables)."""
    return Path(silver_dir or C.SILVER) / "_build" / f"bars_{asset}.json"


def bars_from_minutes(minutes: pl.DataFrame, schedule: pl.DataFrame, asset: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """5-minute bars and per-session summary for one asset (pure; SPEC §4.2).

    ``minutes`` needs ``ts`` (minute open, UTC), ``open``, ``close``, ``is_real``; ``schedule`` is a
    :func:`volrisk.sessions.session_schedule` frame. Returns ``(bars5m, sessions)`` with every scheduled
    session and bar present, sorted by ``(session_date, bar_idx)`` / ``session_date``.
    """
    sched = _check_schedule(schedule)
    bar_us = bar_minutes() * 60_000_000
    ts = pl.col("ts")
    if minutes.schema["ts"] != UTC_US:
        minutes = minutes.with_columns(utc_us_expr("ts", minutes.schema["ts"]))
    if minutes.height and minutes["ts"].is_duplicated().any():
        raise ValueError(f"{asset}: duplicate minute timestamps in bronze input (ts must be unique)")

    real = (
        minutes.lazy()
        .filter(pl.col("is_real").fill_null(False) & pl.col("close").is_not_null())
        .select(ts, pl.col("open").cast(pl.Float64), pl.col("close").cast(pl.Float64))
        .sort(ts)
    )
    # Backward as-of: the latest session with open_utc <= ts; the minute is in it iff ts < close_utc.
    joined = (
        real.join_asof(
            sched.lazy().select("open_utc", "close_utc", "session_date"),
            left_on="ts",
            right_on="open_utc",
            strategy="backward",
            coalesce=False,
        )
        .filter(pl.col("session_date").is_not_null() & (ts < pl.col("close_utc")))
        .with_columns(in_bars=(ts + _MINUTE) <= pl.col("close_utc"), _off=(ts + _MINUTE - pl.col("open_utc")))
        # bar_idx = ceil((close time - open_utc) / bar): close time in (T-5m, T] -> the bar ending at T
        .with_columns(bar_idx=((pl.col("_off").dt.total_microseconds() + bar_us - 1) // bar_us).cast(pl.Int32))
    )

    in_bars = joined.filter("in_bars")
    per_bar = in_bars.group_by("session_date", "bar_idx").agg(
        price=pl.col("close").sort_by("ts").last(), n_real_min=pl.len().cast(pl.Int32)
    )
    p_open = joined.group_by("session_date").agg(p_open=pl.col("open").sort_by("ts").first())
    p_close = in_bars.group_by("session_date").agg(p_close=pl.col("close").sort_by("ts").last())
    grid = (
        sched.lazy()
        .select("session_date", "open_utc", bar_idx=pl.int_ranges(1, pl.col("n_sched") + 1, dtype=pl.Int32))
        .explode("bar_idx", empty_as_null=False)  # n_sched >= 1, so no session vanishes
        .with_columns(ts_end=pl.col("open_utc") + pl.duration(microseconds=pl.col("bar_idx").cast(pl.Int64) * bar_us))
    )
    bars = (
        grid.join(per_bar, on=["session_date", "bar_idx"], how="left")
        .with_columns(asset=pl.lit(asset), n_real_min=pl.col("n_real_min").fill_null(0))
        .select(list(BARS_SCHEMA))
        .cast(BARS_SCHEMA)
        .sort("session_date", "bar_idx")
        .collect()
    )
    n_real = bars.lazy().group_by("session_date").agg(n_real_bars=pl.col("price").is_not_null().sum())
    sessions = (
        sched.lazy()
        .join(n_real, on="session_date", how="left")
        .join(p_open, on="session_date", how="left")
        .join(p_close, on="session_date", how="left")
        .with_columns(asset=pl.lit(asset), n_real_bars=pl.col("n_real_bars").fill_null(0))
        .with_columns(coverage=pl.col("n_real_bars") / pl.col("n_sched"))
        .select(list(SESSIONS_SCHEMA))
        .cast(SESSIONS_SCHEMA)
        .sort("session_date")
        .collect()
    )
    return bars, sessions


def read_minutes(
    asset: str, bronze_dir: Path | str | None = None, lo: datetime | None = None, hi: datetime | None = None
) -> pl.DataFrame:
    """Bronze minutes (``ts, open, close, is_real``) with ``lo <= ts < hi``, from the year partitions."""
    return _read_minute_files(_minute_files(asset, bronze_dir, lo, hi), lo, hi)


def _minute_files(asset: str, bronze_dir: Path | str | None, lo: datetime | None, hi: datetime | None) -> list[Path]:
    """Bronze year partitions of ``asset`` that can hold minutes in ``[lo, hi)`` (all of them if unbounded)."""
    root = minute_dir(asset, bronze_dir)
    files = sorted(root.glob("year=*/part.parquet"))
    if lo is not None and hi is not None:
        files = [f for f in files if lo.year <= int(f.parent.name.split("=")[1]) <= hi.year]
    if not files:
        raise FileNotFoundError(f"no bronze minute files for {asset} under {root}")
    return files


def _read_minute_files(files: list[Path], lo: datetime | None, hi: datetime | None) -> pl.DataFrame:
    cols = ("ts", "open", "close", "is_real")
    frames = [pl.scan_parquet(f, hive_partitioning=False).select(cols) for f in files]
    lf = pl.concat(frames, how="vertical_relaxed")
    ts_dtype = lf.collect_schema()["ts"]
    if ts_dtype != UTC_US:
        lf = lf.with_columns(utc_us_expr("ts", ts_dtype))
    if lo is not None:
        lf = lf.filter(pl.col("ts") >= lo)
    if hi is not None:
        lf = lf.filter(pl.col("ts") < hi)
    return lf.with_columns(pl.col("open", "close").cast(pl.Float64), pl.col("is_real").cast(pl.Boolean)).collect()


def build_bars(
    asset: str,
    bronze_dir: Path | str | None = None,
    out_dir: Path | str | None = None,
    start: date | None = None,
    end: date | None = None,
    *,
    force: bool = False,
) -> dict:
    """Build and write ``silver/bars5m`` and ``silver/sessions`` for one asset (SPEC §4.2).

    ``bronze_dir`` / ``out_dir`` are the bronze and silver layer roots (default ``data/bronze``,
    ``data/silver``); sessions run from ``start`` (default: the asset's config start) to ``end``
    (default: ``dates.data_end``). Returns a summary dict with ``cfg_hash`` and ``skipped``; its data-quality
    counts cover dev sessions only, holdout sessions are only counted (:func:`_dev_diagnostics`).

    SPEC §12: unless ``force``, nothing is recomputed when the build marker (:func:`marker_path`) shows that
    both outputs were built for the current config hash, schedule, bronze files and code and are unchanged
    since; the stored summary is returned instead.
    """
    start = start or C.asset(asset).start
    end = end or C.data_end()
    sched = session_schedule(asset, start, end)
    if sched.height == 0:
        raise ValueError(f"{asset}: no scheduled sessions between {start} and {end}")
    lo, hi = sched["open_utc"].min(), sched["close_utc"].max()
    files = _minute_files(asset, bronze_dir, lo, hi)
    bp, sp, mp = bars_path(asset, out_dir), sessions_path(asset, out_dir), marker_path(asset, out_dir)
    outputs = {"bars5m": bp, "sessions": sp}
    key = _build_key(asset, start, end, sched, files)
    if not force:
        cached = _current_summary(mp, key, outputs)
        if cached is not None:
            log.info("bars %s: outputs current for config %s, skipped (force to rebuild)", asset, key["cfg_hash"])
            return {**cached, "bars_path": str(bp), "sessions_path": str(sp), "skipped": True}

    minutes = _read_minute_files(files, lo, hi)
    bars, sessions = bars_from_minutes(minutes, sched, asset)
    write_parquet(bars, bp)
    write_parquet(sessions, sp)

    min_cov = float(C.load()["sessions"]["min_coverage"])
    summary = {
        "asset": asset,
        "first_session": sched["session_date"].min(),
        "last_session": sched["session_date"].max(),
        "n_sessions": sessions.height,
        "n_bars": bars.height,
        **_dev_diagnostics(minutes, bars, sessions, sched, min_cov),
        "bars_path": str(bp),
        "sessions_path": str(sp),
        "cfg_hash": key["cfg_hash"],
    }
    _write_marker(mp, key, outputs, summary)  # last: a crash before this leaves a marker that no longer matches
    mean_cov = summary["mean_coverage"]
    log.info(
        "bars %s: %d sessions %s..%s, %d bars; dev (%s): %d sessions, %d null bars, mean coverage %s, %d below %.2f;"
        " holdout: %d sessions (row count only)",
        asset,
        summary["n_sessions"],
        summary["first_session"],
        summary["last_session"],
        summary["n_bars"],
        summary["diagnostics_scope"],
        summary["n_dev_sessions"],
        summary["n_null_bars"],
        "n/a" if mean_cov is None else f"{mean_cov:.3f}",
        summary["n_low_coverage"],
        min_cov,
        summary["n_holdout_sessions"],
    )
    return {**summary, "skipped": False}


def _dev_diagnostics(
    minutes: pl.DataFrame, bars: pl.DataFrame, sessions: pl.DataFrame, sched: pl.DataFrame, min_cov: float
) -> dict:
    """Data-quality counts of the dev split only; the holdout split gets a row count (SPEC §10, §11).

    The silver tables hold every session up to ``data_end``, but the summary that is logged and stored in the
    build marker describes only sessions with ``session_date < holdout_start`` and the real minutes that open
    before the close of the last such session — as :func:`volrisk.measures.build_gold` does for gold.
    """
    h0 = C.holdout_start()
    dev = pl.col("session_date") < h0
    dev_sched, dev_sessions, dev_bars = sched.filter(dev), sessions.filter(dev), bars.filter(dev)
    if dev_sched.height:
        dev_minutes = minutes.filter(pl.col("ts") < dev_sched["close_utc"].max())
        n_real_minutes = int(dev_minutes["is_real"].fill_null(False).sum())
    else:
        n_real_minutes = 0
    n_used = int(dev_bars["n_real_min"].sum())
    return {
        "diagnostics_scope": f"session_date < {h0.isoformat()}",
        "n_dev_sessions": dev_sessions.height,
        "n_holdout_sessions": sessions.height - dev_sessions.height,
        "n_null_bars": int(dev_bars["price"].null_count()),
        "n_real_minutes": n_real_minutes,
        "n_real_minutes_used": n_used,
        "n_real_minutes_outside_sessions": n_real_minutes - n_used,
        "mean_coverage": float(dev_sessions["coverage"].mean()) if dev_sessions.height else None,
        "n_low_coverage": int((dev_sessions["coverage"] < min_cov).sum()),
        "n_no_price": int((dev_sessions["p_open"].is_null() | dev_sessions["p_close"].is_null()).sum()),
    }


# --------------------------------------------------------------------------------- build marker (SPEC §12)
_DATE_KEYS = ("first_session", "last_session")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_key(asset: str, start: date, end: date, sched: pl.DataFrame, files: list[Path]) -> dict:
    """Everything the silver outputs depend on: config hash, schedule, bronze files and the bar/session code."""
    here = Path(__file__)
    code = hashlib.sha256(b"".join(p.read_bytes() for p in (here, here.with_name("sessions.py"))))
    return {
        "asset": asset,
        "cfg_hash": C.cfg_hash(),
        "start": start.isoformat(),
        "end": end.isoformat(),
        # the schedule also captures exchange_calendars' holiday data
        "schedule_sha256": hashlib.sha256(sched.write_csv().encode()).hexdigest(),
        "code_sha256": code.hexdigest(),
        "bronze_sha256": {f"{f.parent.name}/{f.name}": _sha256(f) for f in files},
    }


def _current_summary(marker: Path, key: dict, outputs: dict[str, Path]) -> dict | None:
    """Stored summary if ``marker`` was written for ``key`` and every output is still the one it recorded."""
    try:
        rec = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("key") != key or not isinstance(rec.get("summary"), dict):
        return None
    recorded = rec.get("outputs") or {}
    if any(not p.is_file() or recorded.get(name) != _sha256(p) for name, p in outputs.items()):
        return None
    summary = dict(rec["summary"])
    try:
        for k in _DATE_KEYS:
            summary[k] = date.fromisoformat(summary[k])
    except (KeyError, TypeError, ValueError):
        return None
    return summary


def _write_marker(marker: Path, key: dict, outputs: dict[str, Path], summary: dict) -> None:
    rec = {
        "key": key,
        "outputs": {name: _sha256(p) for name, p in outputs.items()},
        "summary": {k: v.isoformat() if k in _DATE_KEYS else v for k, v in summary.items()},
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_suffix(marker.suffix + ".part")
    tmp.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    os.replace(tmp, marker)


def _check_schedule(schedule: pl.DataFrame) -> pl.DataFrame:
    """Schedule cast to SCHEDULE_SCHEMA, sorted, with non-overlapping sessions (needed by the as-of join)."""
    missing = set(SCHEDULE_SCHEMA) - set(schedule.columns)
    if missing:
        raise ValueError(f"schedule is missing columns {sorted(missing)}")
    s = schedule.select(list(SCHEDULE_SCHEMA)).cast(SCHEDULE_SCHEMA).sort("open_utc")
    if s.height:
        bad = (s["close_utc"] <= s["open_utc"]).any() or (s["open_utc"].shift(-1) < s["close_utc"]).any()
        if bad or s["session_date"].is_duplicated().any():
            raise ValueError("schedule sessions must be unique, non-empty and non-overlapping")
    return s
