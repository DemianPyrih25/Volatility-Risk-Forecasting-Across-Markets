"""Data-quality report (SPEC §10).

Inputs: bronze minutes (§3), silver sessions and 5-minute bars (§4.2), the dev gold table (§5.3, read only through
``volrisk.io.load_daily``, which never touches ``data/holdout/``), the newest FRED ``SP500`` snapshot (§2.4) and
the ingestion flag files ``data/bronze/flags/{source}_{ASSET}.parquet`` (§2.2 continuity / BID-ASK alignment,
§2.3 moved / duplicate rows and incident dates; optional). The ingesters clean bronze before writing it, so the
raw duplicate/misaligned timestamp counts come from the flag files; the bronze-level counts are an integrity check.
Every reader stops at ``until`` (default ``config.dev_end()``); a date inside the holdout raises
:class:`volrisk.io.HoldoutSealedError` until the holdout is unsealed (§11). Gold membership (sample start incl. the
SPX probe of §1, first session without a previous close) mirrors ``volrisk.measures``. The report only describes
the data: nothing is filtered or corrected on the basis of it.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Sequence
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from volrisk import config as C
from volrisk import io

log = logging.getLogger(__name__)

ScheduleFn = Callable[[str, date, date], pl.DataFrame]
"""``session_schedule(asset, start, end) -> [session_date, open_utc, close_utc, n_sched]`` (SPEC §4.1)."""

GAP_MIN = 5  # a gap is > 5 minutes between the open times of consecutive real minutes of a session
JUMP_SHARE_SUSPECT = 0.25
REVERSAL_FRAC = 0.5  # jump review: an adjacent opposite return of >= half the size marks an isolated bad print
CFD_START = date(2016, 10, 3)
CFD_TOL_BP = 25.0
SIGNATURE_FREQS = (1, 2, 5, 10, 15, 30)
BINANCE_INCIDENTS = (date(2018, 2, 8), date(2018, 2, 9), date(2019, 5, 15), date(2023, 3, 24))
CONTINUITY_MAX_ROWS = 100  # continuity flags listed in the report (largest |rel_change| first)
FLAG_ROWS_PER_FILE = 20  # other flag rows listed per flag file

# Raw-level timestamp counts per month from the ingestion flag files (SPEC §10 "duplicate/misaligned timestamps"):
# column -> (source, flag, "rows" = sum of the flag's n_rows | "days" = number of flagged days).
FLAG_COLUMNS: dict[str, tuple[str, str, str]] = {
    "n_dup_raw": ("binance", "duplicate", "rows"),
    "n_moved_raw": ("binance", "moved", "rows"),
    "n_spacing_raw": ("binance", "spacing", "rows"),
    "n_misaligned_days": ("dukascopy", "misaligned", "days"),
    "n_missing_side_days": ("dukascopy", "missing_side", "days"),
    "n_decode_error_days": ("dukascopy", "decode_error", "days"),
    "n_continuity_days": ("dukascopy", "continuity", "days"),
}

REPORT_SECTIONS = (
    "## 1. Coverage by asset and year",
    "## 2. Dropped and flagged sessions",
    "## 3. Jump-day share",
    "## 4. Largest 5-minute returns",
    "## 5. CFD check: SPX vs FRED SP500",
    "## 6. Timezone check (DST)",
    "## 7. Volatility signature",
    "## 8. Ingestion flags: Binance incidents and Dukascopy continuity",
)

_UTC = pl.Datetime("us", "UTC")

# Figure styling: recessive axes, fixed colour per entity (validated categorical slots 1-4).
_INK2, _GRID = "#52514e", "#e4e3df"
_ASSET_COLOR = {"BTC": "#2a78d6", "ETH": "#eb6834", "EURUSD": "#1baf7a", "SPX": "#eda100"}
_DST_COLOR, _STD_COLOR = "#2a78d6", "#eb6834"


def _min_cov() -> float:
    return float(C.load()["sessions"]["min_coverage"])


def _bar_min() -> int:
    return int(C.load()["sessions"]["bar_minutes"])


def _until(until: date | None) -> date:
    """``until`` or ``config.dev_end()``; a holdout date needs the unlock token (SPEC §10 "dev data only", §11)."""
    until = until or C.dev_end()
    if until >= C.holdout_start() and not io.holdout_unlocked():
        raise io.HoldoutSealedError(
            f"data-quality inputs up to {until} reach into the sealed holdout (from {C.holdout_start()}); the report "
            "describes dev data only until the holdout is unsealed"
        )
    return until


def _source(asset: str) -> str | None:
    try:
        return C.asset(asset).source
    except KeyError:
        return None


def _is_dukascopy(asset: str) -> bool:
    return _source(asset) == "dukascopy"


# --------------------------------------------------------------------------------------------- readers
def read_bronze(asset: str, bronze_dir: Path = C.BRONZE, until: date | None = None) -> pl.DataFrame:
    """Bronze minutes of one asset (SPEC §3) with ``ts`` before ``until + 1 day``; duplicates kept, sorted by ts.

    Columns: ts, open, close, is_real, spread (null when the source has no quotes).
    """
    until = _until(until)
    root = Path(bronze_dir) / "minute" / f"asset={asset}"
    files = []
    for p in sorted(root.glob("year=*/part.parquet")):
        year = p.parent.name.split("=", 1)[1]
        if year.isdigit() and int(year) <= until.year:
            files.append(p)
    if not files:
        raise FileNotFoundError(f"no bronze minutes for {asset} under {root}")
    cut = datetime.combine(until + timedelta(days=1), time(0), tzinfo=timezone.utc)
    frames = []
    for f in files:
        lf = pl.scan_parquet(f)
        names = lf.collect_schema().names()
        spread = pl.col("spread").cast(pl.Float64) if "spread" in names else pl.lit(None, pl.Float64).alias("spread")
        frames.append(
            lf.select(
                pl.col("ts").cast(_UTC),
                pl.col("open").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("is_real").cast(pl.Boolean),
                spread,
            ).filter(pl.col("ts") < cut)
        )
    return pl.concat(frames).sort("ts").collect()


def _real_minutes(raw: pl.DataFrame) -> pl.DataFrame:
    """Real, minute-aligned, de-duplicated minutes (misaligned/duplicate rows are only counted, never used)."""
    aligned = pl.col("ts").dt.truncate("1m") == pl.col("ts")
    return raw.filter(pl.col("is_real") & aligned).unique("ts", keep="last").sort("ts")


def _read_sessions(asset: str, silver_dir: Path, until: date) -> pl.DataFrame:
    p = Path(silver_dir) / "sessions" / f"asset={asset}" / "part.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no silver sessions for {asset}: {p}")
    s = pl.read_parquet(p)
    if "coverage" not in s.columns:
        s = s.with_columns(coverage=pl.col("n_real_bars") / pl.col("n_sched"))
    return (
        s.select(
            pl.col("session_date").cast(pl.Date),
            pl.col("open_utc").cast(_UTC),
            pl.col("close_utc").cast(_UTC),
            pl.col("n_sched").cast(pl.Int32),
            pl.col("n_real_bars").cast(pl.Int32),
            pl.col("coverage").cast(pl.Float64),
            pl.col("p_open").cast(pl.Float64).fill_nan(None),
            pl.col("p_close").cast(pl.Float64).fill_nan(None),
        )
        .filter(pl.col("session_date") <= until)
        .sort("session_date")
    )


def _read_bars(asset: str, silver_dir: Path, until: date) -> pl.DataFrame:
    p = Path(silver_dir) / "bars5m" / f"asset={asset}" / "part.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no silver bars for {asset}: {p}")
    return (
        pl.read_parquet(p)
        .select(
            pl.col("session_date").cast(pl.Date),
            pl.col("bar_idx").cast(pl.Int32),
            pl.col("ts_end").cast(_UTC),
            pl.col("price").cast(pl.Float64),
        )
        .filter(pl.col("session_date") <= until)
        .sort("session_date", "bar_idx")
    )


def _start_info(asset: str, silver_dir: Path, until: date) -> tuple[date, str]:
    """Modelling start of ``asset`` and how it was set (see :func:`sample_start`)."""
    from volrisk import measures  # resolved at call time, like the sessions module

    sess = _read_sessions(asset, silver_dir, until)
    p = Path(silver_dir) / "bars5m" / f"asset={asset}" / "part.parquet"
    if not p.exists():
        raise FileNotFoundError(f"no silver bars for {asset}: {p}")
    bars = (
        pl.read_parquet(p, columns=["session_date", "n_real_min"])
        .with_columns(pl.col("session_date").cast(pl.Date))
        .filter(pl.col("session_date") <= until)
    )
    cfg = C.asset(asset).start
    try:
        start = measures.sample_start(asset, sess, bars)
    except ValueError as e:  # no month passes the SPX probe: gold cannot be built either
        log.warning("%s: sample-start probe failed (%s); config start %s used in the report", asset, e, cfg)
        return cfg, f"config start; SPX probe failed: {e}"
    if C.clock(asset) == "xnys":
        return start, "SPX probe: first month with >= 95% of sessions having >= 95% real RTH minutes (SPEC §1)"
    return start, "config start"


def sample_start(asset: str, silver_dir: Path = C.SILVER, until: date | None = None) -> date:
    """First session date the gold table keeps (``volrisk.measures.sample_start`` on the silver sessions/bars):
    the config start, or for SPX the probe of SPEC §1. If the probe finds no month the config start is returned."""
    return _start_info(asset, silver_dir, _until(until))[0]


def _cov_ok() -> pl.Expr:
    return pl.col("coverage").fill_nan(None).fill_null(0.0) >= _min_cov()


def _usable(asset: str) -> pl.Expr:
    """Sessions valid under SPEC §4.3: prices present; EURUSD/SPX also coverage >= 0.80 (``M >= 5`` follows)."""
    ok = pl.col("p_open").is_not_null() & pl.col("p_close").is_not_null()
    return ok if asset in C.CRYPTO else ok & _cov_ok()


def _first(cond: pl.Expr) -> pl.Expr:
    """The first row (in frame order) where ``cond`` holds."""
    return cond & (cond.cast(pl.Int32).cum_sum() == 1)


def _in_gold(asset: str, start: date) -> pl.Expr:
    """Sessions that reach the gold table (mirrors ``volrisk.measures``): valid, not the first valid session (it
    has no previous close for ``gap``/``r_cc``) and on or after the modelling ``start``. Rows sorted by date."""
    usable = _usable(asset)
    return usable & ~_first(usable) & (pl.col("session_date") >= start)


def _schedule(fn: ScheduleFn, asset: str, start: date, end: date) -> pl.DataFrame:
    s = fn(asset, start, end)
    return (
        s.select(
            pl.col("session_date").cast(pl.Date),
            pl.col("open_utc").cast(_UTC),
            pl.col("close_utc").cast(_UTC),
            pl.col("n_sched").cast(pl.Int32),
        )
        .filter(pl.col("session_date").is_between(start, end))
        .sort("session_date")
    )


def default_schedule() -> ScheduleFn:
    """``volrisk.sessions.session_schedule`` (resolved at call time; written by the sessions module)."""
    try:
        from volrisk.sessions import session_schedule
    except ImportError as e:  # pragma: no cover - depends on the sessions module being present
        raise RuntimeError("volrisk.sessions.session_schedule is not available; pass schedule_fn") from e
    return session_schedule


def silver_schedule(silver_dir: Path = C.SILVER) -> ScheduleFn:
    """Schedule function backed by the silver sessions table (fallback when the calendar module is unavailable)."""

    def fn(asset: str, start: date, end: date) -> pl.DataFrame:
        return _read_sessions(asset, silver_dir, end).filter(pl.col("session_date") >= start)

    return fn


def _attach_sessions(minutes: pl.DataFrame, sessions: pl.DataFrame) -> pl.DataFrame:
    """Assign each minute to the session whose window ``(open_utc, close_utc]`` contains its close time ``ts+60s``
    (SPEC §4.2); minutes outside every session are dropped."""
    m = minutes.with_columns(tc=pl.col("ts") + pl.duration(minutes=1)).sort("tc")
    s = sessions.select("session_date", "open_utc", "close_utc", "n_sched").sort("open_utc")
    j = m.join_asof(s, left_on="tc", right_on="open_utc", strategy="backward", allow_exact_matches=False)
    return j.filter(pl.col("tc") <= pl.col("close_utc"))


def read_flags(flags_dir: Path | None = None, until: date | None = None) -> pl.DataFrame:
    """All ingestion flag files (``*.parquet``) concatenated with a ``source_file`` column; empty if none exist.

    Rows dated after ``until`` (holdout) in any Date/Datetime column, or in a text column whose name contains
    ``date``, are dropped (per file, before the files are combined).
    """
    until = _until(until)
    d = Path(flags_dir) if flags_dir is not None else C.BRONZE / "flags"
    files = sorted(d.glob("*.parquet")) if d.exists() else []
    frames = [_dev_rows(pl.read_parquet(f), until).with_columns(source_file=pl.lit(f.stem)) for f in files]
    if not frames:
        return pl.DataFrame(schema={"source_file": pl.Utf8})
    try:
        return pl.concat(frames, how="diagonal_relaxed")
    except pl.exceptions.PolarsError:  # incompatible schemas across sources: fall back to text
        return pl.concat([f.with_columns(pl.all().cast(pl.Utf8)) for f in frames], how="diagonal")


def _dev_rows(df: pl.DataFrame, until: date) -> pl.DataFrame:
    cut = datetime.combine(until + timedelta(days=1), time(0))
    for name, dtype in df.schema.items():
        col = pl.col(name)
        if dtype == pl.Date:
            df = df.filter(col.is_null() | (col <= until))
        elif isinstance(dtype, pl.Datetime):
            limit = cut.replace(tzinfo=timezone.utc) if dtype.time_zone else cut
            df = df.filter(col.is_null() | (col < pl.lit(limit).cast(dtype)))
        elif dtype == pl.Utf8 and "date" in name.lower():
            parsed = col.str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)
            df = df.filter(parsed.is_null() | (parsed <= until))
    return df


def _as_date(name: str, dtype: pl.DataType) -> pl.Expr:
    col = pl.col(name)
    if dtype == pl.Date:
        return col
    if isinstance(dtype, pl.Datetime):
        return col.dt.date()
    return col.cast(pl.Utf8).str.slice(0, 10).str.to_date("%Y-%m-%d", strict=False)


def flags_file(asset: str, flags_dir: Path | None = None) -> Path:
    """Ingestion flag file of ``asset``: ``{source}_{ASSET}.parquet`` (Binance / Dukascopy ingester naming)."""
    d = Path(flags_dir) if flags_dir is not None else C.BRONZE / "flags"
    return d / f"{_source(asset)}_{asset}.parquet"


def flag_counts(asset: str, flags_dir: Path | None = None, until: date | None = None) -> pl.DataFrame | None:
    """Raw-level duplicate/misaligned timestamp counts of one asset per UTC month of the flagged day (SPEC §10).

    Bronze is cleaned by the ingesters (Binance floors open times and de-duplicates; Dukascopy keeps the common
    BID/ASK minutes and skips days whose record times are not increasing minute offsets), so the counts come from
    the asset's flag file, see :data:`FLAG_COLUMNS`: Binance ``n_dup_raw`` / ``n_moved_raw`` / ``n_spacing_raw`` =
    raw rows dropped as duplicates / open times off the minute grid / open-time steps that are not a positive
    multiple of 60 s (sums of ``n_rows``); Dukascopy ``n_*_days`` = flagged days. Columns of the other source are
    null. Columns: month, *FLAG_COLUMNS. None when the asset has no flag file.
    """
    until = _until(until)
    src, path = _source(asset), flags_file(asset, flags_dir)
    if src is None or not path.exists():
        return None
    f = _dev_rows(pl.read_parquet(path), until)
    if "asset" in f.columns:
        f = f.filter(pl.col("asset") == asset)
    if "n_rows" not in f.columns:  # a flag file without row counts: one row per flag
        f = f.with_columns(n_rows=pl.lit(1, pl.Int64))
    f = f.with_columns(_as_date("date", f.schema["date"]).alias("date")).filter(pl.col("date").is_not_null())
    n_rows = pl.col("n_rows").cast(pl.Int64, strict=False).fill_null(1)
    aggs = []
    for col, (s, flag, how) in FLAG_COLUMNS.items():
        if s == src:
            hit = pl.col("flag") == flag
            agg = n_rows.filter(hit).sum() if how == "rows" else pl.col("date").filter(hit).n_unique()
            aggs.append(agg.cast(pl.Int64).alias(col))
    other = [pl.lit(None, pl.Int64).alias(c) for c, (s, _, _) in FLAG_COLUMNS.items() if s != src]
    return (
        f.group_by(month=pl.col("date").dt.truncate("1mo"))
        .agg(aggs)
        .with_columns(other)
        .select("month", *FLAG_COLUMNS)
        .sort("month")
    )


def newest_fred(implied_dir: Path = C.RAW / "implied") -> Path | None:
    """Newest ``fred_SP500_*.csv`` snapshot (download date in the file name), or None."""
    files = sorted(Path(implied_dir).glob("fred_SP500_*.csv"))
    return files[-1] if files else None


# --------------------------------------------------------------------------------------------- coverage
def session_issues(
    asset: str,
    silver_dir: Path = C.SILVER,
    schedule_fn: ScheduleFn | None = None,
    until: date | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Every session with its SPEC §4.3/§5.3 treatment in the gold table (mirrors ``volrisk.measures``).

    Sessions come from silver; with ``schedule_fn`` the scheduled sessions absent from silver between its first
    and last session are added. ``reason`` = the first that applies, or null: ``missing`` (scheduled, absent from
    silver), ``no_price`` (``p_open`` or ``p_close`` null), ``low_coverage`` (< 0.80), ``before_sample_start``
    (before ``start``, default :func:`sample_start`), ``no_prev_close`` (the first valid session: gold needs the
    previous close for ``gap``/``r_cc``). For crypto ``low_coverage`` is checked last, as it only flags.
    ``action``: ``dropped`` (not in gold) for every reason except crypto ``low_coverage`` (``flagged``, kept).
    """
    until = _until(until)
    start = start if start is not None else sample_start(asset, silver_dir, until)
    s = _read_sessions(asset, silver_dir, until).with_columns(in_silver=pl.lit(True))
    if schedule_fn is not None and s.height:
        sched = _schedule(schedule_fn, asset, s["session_date"].min(), s["session_date"].max())
        miss = sched.join(s.select("session_date"), on="session_date", how="anti").with_columns(
            n_real_bars=pl.lit(0, pl.Int32),
            coverage=pl.lit(0.0),
            p_open=pl.lit(None, pl.Float64),
            p_close=pl.lit(None, pl.Float64),
            in_silver=pl.lit(False),
        )
        s = pl.concat([s, miss.select(s.columns)]).sort("session_date")
    crypto = asset in C.CRYPTO
    low = ("low_coverage", ~_cov_ok())
    checks = [("missing", ~pl.col("in_silver")),
              ("no_price", pl.col("p_open").is_null() | pl.col("p_close").is_null())]
    checks += [] if crypto else [low]
    checks += [("before_sample_start", pl.col("session_date") < start), ("no_prev_close", _first(_usable(asset)))]
    checks += [low] if crypto else []
    reason: pl.Expr = pl.lit(None, pl.Utf8)
    for name, cond in reversed(checks):
        reason = pl.when(cond).then(pl.lit(name)).otherwise(reason)
    s = s.with_columns(reason=reason)
    kept = pl.col("reason").is_null() | ((pl.col("reason") == "low_coverage") if crypto else pl.lit(False))
    action = (
        pl.when(pl.col("reason").is_null())
        .then(pl.lit(None, pl.Utf8))
        .when(kept)
        .then(pl.lit("flagged"))
        .otherwise(pl.lit("dropped"))
    )
    return s.with_columns(action=action).drop("in_silver")


def monthly_coverage(
    asset: str,
    bronze_dir: Path = C.BRONZE,
    silver_dir: Path = C.SILVER,
    schedule_fn: ScheduleFn | None = None,
    until: date | None = None,
    *,
    flags_dir: Path | None = None,
    daily: pd.DataFrame | pl.DataFrame | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Per asset × month data-quality table (SPEC §10).

    Session-based columns are keyed by the month of ``session_date``: scheduled vs real minutes (real =
    minute-aligned, de-duplicated real minutes inside the session window) and bars, ``null_bar_share``,
    ``n_gaps_gt5`` (intervals > 5 min between consecutive real minutes of a session), longest such interval,
    median/p99 spread in price units and bp of mid (Dukascopy only, real in-session minutes), crossed quotes,
    session treatment counts (:func:`session_issues` with ``start``) and, with the gold ``daily`` table, the
    jump-day share (``jump_suspect`` when > 25%; null without ``daily``). Raw duplicate/misaligned timestamp counts
    (:data:`FLAG_COLUMNS`, by UTC month of the flagged day) come from :func:`flag_counts` (``flags_dir`` defaults to
    ``bronze_dir/flags``; null when the flag file is absent). ``n_dup_bronze``/``n_misaligned_bronze`` count all
    bronze rows by UTC month of ``ts``: an integrity check, 0 for a correctly cleaned bronze layer.
    """
    until = _until(until)
    sess = session_issues(asset, silver_dir, schedule_fn, until, start)
    raw = read_bronze(asset, bronze_dir, until)
    if sess.height:
        raw = raw.filter(pl.col("ts") < sess["close_utc"].max())

    month_ts = pl.col("ts").dt.truncate("1mo").dt.date().alias("month")
    hygiene = raw.group_by(month_ts).agg(
        n_dup_bronze=(pl.len() - pl.col("ts").n_unique()).cast(pl.Int64),
        n_misaligned_bronze=(pl.col("ts").dt.truncate("1m") != pl.col("ts")).sum().cast(pl.Int64),
    )
    flags = flag_counts(asset, flags_dir if flags_dir is not None else Path(bronze_dir) / "flags", until)
    if flags is None:
        flags = pl.DataFrame(schema={"month": pl.Date, **{c: pl.Int64 for c in FLAG_COLUMNS}})
    jumps = pl.DataFrame(schema={"month": pl.Date, "n_gold_days": pl.Int64, "n_jump_days": pl.Int64})
    if daily is not None:
        g = _to_pl(daily)
        g = g.filter(pl.col("asset") == asset) if "asset" in g.columns else g.with_columns(asset=pl.lit(asset))
        jumps = jump_share(g.filter(pl.col("session_date") <= until), by="month").select(
            "month", n_gold_days=pl.col("n_days").cast(pl.Int64), n_jump_days="n_jump"
        )

    mins = _attach_sessions(_real_minutes(raw), sess).with_columns(
        month=pl.col("session_date").dt.truncate("1mo")
    )
    per_sess = mins.group_by("session_date").agg(
        real_min=pl.len(),
        n_gaps_gt5=(pl.col("ts").sort().diff() > pl.duration(minutes=GAP_MIN)).sum(),
        max_gap_min=pl.col("ts").sort().diff().max().dt.total_minutes(),
    )
    spread_bp = 1e4 * pl.col("spread") / pl.col("close")
    spread = mins.group_by("month").agg(
        spread_med=pl.col("spread").median(),
        spread_p99=pl.col("spread").quantile(0.99, "linear"),
        spread_med_bp=spread_bp.median(),
        spread_p99_bp=spread_bp.quantile(0.99, "linear"),
        n_crossed=(pl.col("spread") < 0).sum().cast(pl.Int64),
    )

    by_sess = sess.join(per_sess, on="session_date", how="left").with_columns(
        month=pl.col("session_date").dt.truncate("1mo")
    )
    monthly = by_sess.group_by("month").agg(
        n_sessions=pl.len(),
        sched_min=(pl.col("n_sched").sum() * _bar_min()).cast(pl.Int64),
        real_min=pl.col("real_min").fill_null(0).sum().cast(pl.Int64),
        sched_bars=pl.col("n_sched").sum().cast(pl.Int64),
        real_bars=pl.col("n_real_bars").sum().cast(pl.Int64),
        n_gaps_gt5=pl.col("n_gaps_gt5").fill_null(0).sum().cast(pl.Int64),
        max_gap_min=pl.col("max_gap_min").max().cast(pl.Int64),
        n_missing=(pl.col("reason") == "missing").sum().cast(pl.Int64),
        n_no_price=(pl.col("reason") == "no_price").sum().cast(pl.Int64),
        n_low_cov=(pl.col("reason") == "low_coverage").sum().cast(pl.Int64),
        n_pre_start=(pl.col("reason") == "before_sample_start").sum().cast(pl.Int64),
        n_dropped=(pl.col("action") == "dropped").sum().cast(pl.Int64),
        n_flagged=(pl.col("action") == "flagged").sum().cast(pl.Int64),
    )
    zero = ("n_sessions", "sched_min", "real_min", "sched_bars", "real_bars", "n_gaps_gt5", "n_missing",
            "n_no_price", "n_low_cov", "n_pre_start", "n_dropped", "n_flagged", "n_dup_bronze", "n_misaligned_bronze")
    has_flags = flags_file(asset, flags_dir if flags_dir is not None else Path(bronze_dir) / "flags").exists()
    src = _source(asset)
    zero += tuple(c for c, (s, _, _) in FLAG_COLUMNS.items() if has_flags and s == src)
    zero += ("n_gold_days", "n_jump_days") if daily is not None else ()
    out = monthly
    for other in (hygiene, flags, jumps):  # months that only appear in bronze, flags or gold are kept
        out = out.join(other, on="month", how="full", coalesce=True)
    out = (
        out.join(spread, on="month", how="left")
        .with_columns(pl.col(c).fill_null(0) for c in zero)
        .with_columns(
            asset=pl.lit(asset),
            real_min_share=pl.when(pl.col("sched_min") > 0).then(pl.col("real_min") / pl.col("sched_min")),
            null_bar_share=pl.when(pl.col("sched_bars") > 0).then(1 - pl.col("real_bars") / pl.col("sched_bars")),
            jump_share=pl.when(pl.col("n_gold_days") > 0).then(pl.col("n_jump_days") / pl.col("n_gold_days")),
        )
        .with_columns(jump_suspect=pl.col("jump_share") > JUMP_SHARE_SUSPECT)
        .sort("month")
    )
    return out.select(
        "asset", "month", "n_sessions", "sched_min", "real_min", "real_min_share", "sched_bars", "real_bars",
        "null_bar_share", "n_gaps_gt5", "max_gap_min", *FLAG_COLUMNS, "n_dup_bronze", "n_misaligned_bronze",
        "spread_med", "spread_p99", "spread_med_bp", "spread_p99_bp", "n_crossed", "n_missing", "n_no_price",
        "n_low_cov", "n_pre_start", "n_dropped", "n_flagged", "n_gold_days", "n_jump_days", "jump_share",
        "jump_suspect",
    )


def yearly_coverage(monthly: pl.DataFrame) -> pl.DataFrame:
    """Monthly coverage aggregated to calendar years (spread: median of monthly medians, max of monthly p99;
    raw flag counts stay null when they do not apply to the asset's source or its flag file is absent)."""
    s = pl.col

    def total(c: str) -> pl.Expr:
        return pl.when(s(c).is_not_null().any()).then(s(c).sum()).alias(c)

    return (
        monthly.group_by("asset", year=s("month").dt.year())
        .agg(
            s("n_sessions").sum(),
            s("sched_bars").sum(),
            s("real_bars").sum(),
            s("sched_min").sum(),
            s("real_min").sum(),
            s("n_gaps_gt5").sum(),
            s("max_gap_min").max(),
            *[total(c) for c in FLAG_COLUMNS],
            s("n_dup_bronze").sum(),
            s("n_misaligned_bronze").sum(),
            s("spread_med_bp").median(),
            s("spread_p99_bp").max(),
            s("n_crossed").sum(),
            s("n_pre_start").sum(),
            s("n_dropped").sum(),
            s("n_flagged").sum(),
        )
        .with_columns(
            null_bar_share=pl.when(s("sched_bars") > 0).then(1 - s("real_bars") / s("sched_bars")),
            real_min_share=pl.when(s("sched_min") > 0).then(s("real_min") / s("sched_min")),
        )
        .sort("asset", "year")
    )


# --------------------------------------------------------------------------------------------- gold-based checks
def _to_pl(daily: pd.DataFrame | pl.DataFrame) -> pl.DataFrame:
    df = daily if isinstance(daily, pl.DataFrame) else pl.from_pandas(daily)
    return df.with_columns(pl.col("session_date").cast(pl.Date))


def jump_share(daily: pd.DataFrame | pl.DataFrame, by: str = "month") -> pl.DataFrame:
    """Share of gold sessions with a significant jump (``j > 0``) per asset × month (SPEC §10; ``month`` = first
    day of the month of ``session_date``) or, with ``by="year"``, per asset-year as a summary. ``suspect`` when the
    share exceeds 25% (SPEC §10 "suspect data": a flag for manual review, see :func:`jump_review`; never a filter).
    Columns: asset, month | year, n_days, n_jump, share, suspect."""
    if by not in ("month", "year"):
        raise ValueError(f"by must be 'month' or 'year', not {by!r}")
    d = _to_pl(daily)
    sd = pl.col("session_date")
    key = (sd.dt.truncate("1mo") if by == "month" else sd.dt.year()).alias(by)
    return (
        d.group_by("asset", key)
        .agg(n_days=pl.len(), n_jump=(pl.col("j") > 0).sum().cast(pl.Int64))
        .with_columns(share=pl.col("n_jump") / pl.col("n_days"))
        .with_columns(suspect=pl.col("share") > JUMP_SHARE_SUSPECT)
        .sort("asset", by)
    )


def jump_review(
    asset: str,
    daily: pd.DataFrame | pl.DataFrame,
    silver_dir: Path = C.SILVER,
    until: date | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Review aids for the jump-day share of one asset per month (SPEC §10: > 25% ⇒ suspect data, manual review).

    ``n_days``/``n_jump``/``share``/``suspect`` are those of :func:`jump_share` on the gold rows (``daily``) of
    ``asset``. The other columns show the usual signatures of data errors behind a detected jump, from the §5.1
    returns of the same sessions (:func:`intraday_returns`): ``n_ret``/``n_zero``/``zero_ret_share`` = 5-minute
    returns of all gold sessions of the month and the share exactly 0 (stale or coarse prices); on jump days
    (``j > 0``) the session's largest |return| is classified as ``n_reversed`` (an adjacent return of the same
    session has the opposite sign and at least :data:`REVERSAL_FRAC` of its size: an isolated bad print),
    ``n_spanning`` (it spans null bars) and ``n_partial`` counts jump days flagged partial (crypto
    ``flag_partial``; else coverage < ``min_coverage``; null when ``daily`` has neither column). A suspect month
    whose jump days look like the asset's other months on these measures points to a property of the market or of
    the sampling rather than to a feed error. Nothing is filtered on the basis of it.
    Columns: asset, month, n_days, n_jump, share, suspect, n_ret, n_zero, zero_ret_share, n_reversed, n_spanning,
    n_partial.
    """
    until = _until(until)
    g = _to_pl(daily)
    g = g.filter(pl.col("asset") == asset) if "asset" in g.columns else g
    g = g.filter(pl.col("session_date") <= until)
    if "flag_partial" in g.columns:
        partial = pl.col("flag_partial").cast(pl.Boolean)
    elif "coverage" in g.columns:
        partial = ~_cov_ok()
    else:
        partial = pl.lit(None, pl.Boolean)
    r = intraday_returns(asset, silver_dir, until, True, start).sort("session_date", "bar_idx")
    r_abs = pl.col("r").abs()

    def opposite(nb: pl.Expr) -> pl.Expr:
        return ((nb * pl.col("r") < 0) & (nb.abs() >= REVERSAL_FRAC * r_abs)).fill_null(False)

    prev_r, next_r = (pl.col("r").shift(k).over("session_date") for k in (1, -1))
    r = r.with_columns(
        reversed=opposite(prev_r) | opposite(next_r),
        spanning=(pl.col("bar_idx") - pl.col("idx_prev")) > 1,
    )

    def at_top(c: str) -> pl.Expr:  # value of ``c`` at the session's largest |return| (ties: earliest bar)
        return pl.col(c).sort_by([r_abs, pl.col("bar_idx")], descending=[True, False]).first()

    per = r.group_by("session_date").agg(
        n_ret=pl.len().cast(pl.Int64),
        n_zero=(pl.col("r") == 0).sum().cast(pl.Int64),
        top_reversed=at_top("reversed"),
        top_spanning=at_top("spanning"),
    )
    jump = pl.col("j") > 0
    d = g.select("session_date", jump.alias("jump"), partial.alias("partial")).join(per, on="session_date", how="left")
    jd = pl.col("jump")
    return (
        d.group_by(month=pl.col("session_date").dt.truncate("1mo"))
        .agg(
            n_days=pl.len().cast(pl.Int64),
            n_jump=jd.sum().cast(pl.Int64),
            n_ret=pl.col("n_ret").sum().cast(pl.Int64),
            n_zero=pl.col("n_zero").sum().cast(pl.Int64),
            n_reversed=(jd & pl.col("top_reversed").fill_null(False)).sum().cast(pl.Int64),
            n_spanning=(jd & pl.col("top_spanning").fill_null(False)).sum().cast(pl.Int64),
            n_partial=pl.when(pl.col("partial").is_not_null().any())
            .then((jd & pl.col("partial").fill_null(False)).sum())
            .cast(pl.Int64),
        )
        .with_columns(
            asset=pl.lit(asset),
            share=pl.col("n_jump") / pl.col("n_days"),
            zero_ret_share=pl.when(pl.col("n_ret") > 0).then(pl.col("n_zero") / pl.col("n_ret")),
        )
        .with_columns(suspect=pl.col("share") > JUMP_SHARE_SUSPECT)
        .select("asset", "month", "n_days", "n_jump", "share", "suspect", "n_ret", "n_zero", "zero_ret_share",
                "n_reversed", "n_spanning", "n_partial")
        .sort("month")
    )


def read_fred_sp500(fred_csv: Path) -> pl.DataFrame:
    """FRED ``SP500`` csv (``observation_date,SP500``; empty or '.' = missing) as [date, close], nulls dropped."""
    raw = pl.read_csv(fred_csv, infer_schema=False)
    date_col, val_col = raw.columns[:2]
    return (
        raw.select(
            pl.col(date_col).str.strip_chars().str.to_date("%Y-%m-%d").alias("date"),
            pl.col(val_col).str.strip_chars().cast(pl.Float64, strict=False).alias("close"),
        )
        .drop_nulls()
        .sort("date")
    )


def cfd_check(
    daily_spx: pd.DataFrame | pl.DataFrame,
    fred_csv: Path,
    start: date = CFD_START,
    tol_bp: float = CFD_TOL_BP,
) -> dict:
    """SPX CFD ``r_cc`` vs FRED SP500 close-to-close log returns (percent), SPEC §10.

    Aligned by ``session_date`` = FRED observation date (NY trading dates) from ``start`` on. A date is compared
    only when both series' previous observation is the same date (a dropped SPX session or a missing FRED value
    makes the returns span different periods; those dates are counted in ``n_prev_mismatch``).
    Returns ``corr``, ``n``, ``days_over_25bp`` (list of dicts session_date, r_cc, r_fred, diff_bp),
    ``mean_abs_diff_bp``, ``n_prev_mismatch``, ``start``.
    """
    d = _to_pl(daily_spx)
    if "asset" in d.columns:
        d = d.filter(pl.col("asset") == "SPX")
    spx = d.sort("session_date").select("session_date", "r_cc", prev_spx=pl.col("session_date").shift(1))
    fred = read_fred_sp500(fred_csv).select(
        session_date=pl.col("date"),
        r_fred=100 * (pl.col("close") / pl.col("close").shift(1)).log(),
        prev_fred=pl.col("date").shift(1),
    )
    j = (
        spx.join(fred, on="session_date", how="inner")
        .filter((pl.col("session_date") >= start) & pl.col("r_cc").is_not_null() & pl.col("r_fred").is_not_null())
        .sort("session_date")
    )
    same = j.filter(pl.col("prev_spx") == pl.col("prev_fred")).with_columns(
        diff_bp=100 * (pl.col("r_cc") - pl.col("r_fred"))
    )
    n = same.height
    corr = float(np.corrcoef(same["r_cc"].to_numpy(), same["r_fred"].to_numpy())[0, 1]) if n >= 3 else float("nan")
    over = same.filter(pl.col("diff_bp").abs() > tol_bp).select("session_date", "r_cc", "r_fred", "diff_bp")
    return {
        "corr": corr,
        "n": n,
        "days_over_25bp": over.to_dicts(),
        "mean_abs_diff_bp": float(same["diff_bp"].abs().mean()) if n else float("nan"),
        "n_prev_mismatch": j.height - n,
        "start": start,
    }


# --------------------------------------------------------------------------------------------- 5-minute returns
def intraday_returns(
    asset: str,
    silver_dir: Path = C.SILVER,
    until: date | None = None,
    valid_only: bool = True,
    start: date | None = None,
) -> pl.DataFrame:
    """Intraday percent log returns of SPEC §5.1 from silver bars: path ``[p_open, non-null bars]``.

    ``valid_only``: only sessions that reach the gold table (valid, not the first valid session, on or after
    ``start``, default :func:`sample_start`); otherwise every silver session. A return spanning null bars is one
    return (``bar_idx - idx_prev > 1``); ``idx_prev = 0`` marks the first return from ``p_open`` (stamped at
    ``open_utc``). Columns: asset, session_date, n_sched, bar_idx, idx_prev, ts_prev, ts_end, p_prev, price, r.
    """
    until = _until(until)
    sess = _read_sessions(asset, silver_dir, until)
    if valid_only:
        start = start if start is not None else sample_start(asset, silver_dir, until)
        sess = sess.filter(_in_gold(asset, start))
    bars = _read_bars(asset, silver_dir, until).join(sess.select("session_date"), on="session_date", how="semi")
    first = sess.filter(pl.col("p_open").is_not_null()).select(
        "session_date", bar_idx=pl.lit(0, pl.Int32), ts_end="open_utc", price="p_open"
    )
    pts = bars.filter(pl.col("price").is_not_null()).select("session_date", "bar_idx", "ts_end", "price")
    path = pl.concat([first, pts]).sort("session_date", "bar_idx")
    prev = {c: pl.col(src).shift(1).over("session_date") for c, src in
            (("idx_prev", "bar_idx"), ("ts_prev", "ts_end"), ("p_prev", "price"))}
    return (
        path.with_columns(**prev)
        .filter(pl.col("p_prev").is_not_null())
        .with_columns(r=100 * (pl.col("price") / pl.col("p_prev")).log())
        .join(sess.select("session_date", "n_sched"), on="session_date", how="left")
        .select(
            pl.lit(asset).alias("asset"), "session_date", "n_sched", "bar_idx", "idx_prev", "ts_prev", "ts_end",
            "p_prev", "price", "r",
        )
    )


def top_returns(
    asset: str,
    silver_dir: Path = C.SILVER,
    n: int = 20,
    until: date | None = None,
    valid_only: bool = True,
    start: date | None = None,
) -> pl.DataFrame:
    """The ``n`` largest |5-minute returns| with timestamps, for manual review (never auto-filtered); sessions as
    in :func:`intraday_returns`."""
    r = intraday_returns(asset, silver_dir, until, valid_only, start)
    return (
        r.with_columns(abs_r=pl.col("r").abs(), span_bars=pl.col("bar_idx") - pl.col("idx_prev"))
        .sort(["abs_r", "session_date", "bar_idx"], descending=[True, False, False])
        .head(n)
        .select("asset", "session_date", "ts_prev", "ts_end", "bar_idx", "span_bars", "p_prev", "price", "r")
    )


def _ny_dst(d: pl.Expr) -> pl.Expr:
    """Whether America/New_York observes daylight time at noon of the date."""
    noon = (d.cast(pl.Datetime("us")) + pl.duration(hours=12)).dt.replace_time_zone("America/New_York")
    return noon.dt.dst_offset().dt.total_seconds() != 0


def tz_check(
    asset: str,
    silver_dir: Path = C.SILVER,
    fig_path: Path | None = None,
    until: date | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Mean squared 5-minute return by session-local ``bar_idx``, NY daylight vs standard time (SPEC §10).

    Uses single-bar returns (no span over null bars) of full-length gold sessions (:func:`intraday_returns`;
    modal ``n_sched``, so SPX half-days are excluded). Columns: bar_idx, n_dst, mean_r2_dst, n_std, mean_r2_std.
    A profile feature that moves between the two regimes on a session-local clock signals a timezone bug (see
    :func:`profile_alignment`).
    """
    r = intraday_returns(asset, silver_dir, until, True, start).filter(pl.col("bar_idx") - pl.col("idx_prev") == 1)
    if r.is_empty():
        return pl.DataFrame(schema={"bar_idx": pl.Int32, "n_dst": pl.UInt32, "mean_r2_dst": pl.Float64,
                                    "n_std": pl.UInt32, "mean_r2_std": pl.Float64})
    n_full = int(r["n_sched"].mode().max())
    r = r.filter(pl.col("n_sched") == n_full).with_columns(dst=_ny_dst(pl.col("session_date")))
    out = pl.DataFrame({"bar_idx": pl.int_range(1, n_full + 1, eager=True).cast(pl.Int32)})
    for flag, tag in ((True, "dst"), (False, "std")):
        g = r.filter(pl.col("dst") == flag).group_by("bar_idx").agg(
            pl.len().alias(f"n_{tag}"), (pl.col("r") ** 2).mean().alias(f"mean_r2_{tag}")
        )
        out = out.join(g, on="bar_idx", how="left")
    out = out.with_columns(pl.col("n_dst").fill_null(0), pl.col("n_std").fill_null(0)).sort("bar_idx")
    if fig_path is not None:
        _plot_tz(asset, out, Path(fig_path))
    return out


def profile_alignment(tz: pl.DataFrame, max_shift: int | None = None) -> dict:
    """How the daylight-time profile lines up with the standard-time profile of :func:`tz_check`.

    ``best_shift`` = bar shift ``s`` maximising corr(log m_dst[i], log m_std[i+s]) (ties: smaller |s|),
    ``corr_best`` its correlation and ``corr_0`` the correlation without shift. On a correct session-local clock
    ``best_shift = 0``. Positive ``s``: features occur ``s`` bars earlier in the session during NY daylight
    time (crypto sessions are UTC days, so US-hours activity is expected at ``s = +12``). All None when the
    profiles are too short or flat.
    """
    a = np.log(tz["mean_r2_dst"].fill_null(np.nan).to_numpy().astype(float))
    b = np.log(tz["mean_r2_std"].fill_null(np.nan).to_numpy().astype(float))
    n = len(a)
    max_shift = max_shift if max_shift is not None else min(24, n // 3)
    corr: dict[int, float] = {}
    for s in sorted(range(-max_shift, max_shift + 1), key=abs):
        x, y = (a[: n - s], b[s:]) if s >= 0 else (a[-s:], b[: n + s])
        ok = np.isfinite(x) & np.isfinite(y)
        if ok.sum() < max(3, n // 2) or np.std(x[ok]) == 0 or np.std(y[ok]) == 0:
            continue
        corr[s] = float(np.corrcoef(x[ok], y[ok])[0, 1])
    if not corr:
        return {"best_shift": None, "corr_best": None, "corr_0": None}
    best = max(corr, key=lambda k: (corr[k], -abs(k)))
    return {"best_shift": best, "corr_best": corr[best], "corr_0": corr.get(0)}


def profile_shift(tz: pl.DataFrame, max_shift: int | None = None) -> int | None:
    """Best-aligning bar shift of :func:`profile_alignment` (0 on a correct session-local clock)."""
    return profile_alignment(tz, max_shift)["best_shift"]


# --------------------------------------------------------------------------------------------- signature
def signature(
    asset: str,
    bronze_dir: Path = C.BRONZE,
    schedule_fn: ScheduleFn | None = None,
    freqs: Sequence[int] = SIGNATURE_FREQS,
    fig_path: Path | None = None,
    until: date | None = None,
    start: date | None = None,
) -> pl.DataFrame:
    """Volatility signature: mean daily RV (%²) when sampling real-minute closes every k minutes (SPEC §10).

    Per session (``schedule_fn``, default ``volrisk.sessions.session_schedule``) the price path mirrors
    §4.2/§5.1 at a k-minute grid: ``[p_open, close of the last real minute in each right-closed k-minute bucket]``
    with empty buckets skipped. Only sessions dated on or after ``start`` (the modelling start; None = the first
    bronze day) with real minutes covering >= ``min_coverage`` of the scheduled minutes enter, and the same
    sessions are used for every k. Columns: freq_min, mean_rv, n_sessions, rel_5m.
    """
    until = _until(until)
    fn = schedule_fn or default_schedule()
    real = _real_minutes(read_bronze(asset, bronze_dir, until))
    schema = {"freq_min": pl.Int32, "mean_rv": pl.Float64, "n_sessions": pl.Int64, "rel_5m": pl.Float64}
    if real.is_empty():
        return pl.DataFrame(schema=schema)
    first = real["ts"].min().date()
    lo = max(first, start) if start is not None else first
    if lo > until:
        return pl.DataFrame(schema=schema)
    m = _attach_sessions(real, _schedule(fn, asset, lo, until))
    m = (
        m.with_columns(n_real=pl.len().over("session_date"))
        .filter(pl.col("n_real") >= _min_cov() * _bar_min() * pl.col("n_sched"))
        .with_columns(elapsed=(pl.col("tc") - pl.col("open_utc")).dt.total_minutes())
        .sort("ts")
    )
    if m.is_empty():
        return pl.DataFrame(schema=schema)
    p_open = m.group_by("session_date").agg(b=pl.lit(0, pl.Int64), price=pl.col("open").first())
    rows = []
    for k in freqs:
        bucket = ((pl.col("elapsed") + k - 1) // k).cast(pl.Int64)  # right-closed k-minute buckets 1, 2, ...
        pts = m.group_by("session_date", b=bucket).agg(price=pl.col("close").last())
        rv = (
            pl.concat([p_open, pts.select(p_open.columns)])
            .sort("session_date", "b")
            .with_columns(r=100 * pl.col("price").log().diff().over("session_date"))
            .group_by("session_date")
            .agg(rv=(pl.col("r") ** 2).sum())
        )
        rows.append({"freq_min": int(k), "mean_rv": float(rv["rv"].mean()), "n_sessions": rv.height})
    out = pl.DataFrame(rows, schema={"freq_min": pl.Int32, "mean_rv": pl.Float64, "n_sessions": pl.Int64})
    ref = out.filter(pl.col("freq_min") == 5)["mean_rv"]  # the project's 5-minute sampling (SPEC §4.2)
    out = out.with_columns(rel_5m=pl.col("mean_rv") / ref[0] if ref.len() else pl.lit(None, pl.Float64))
    if fig_path is not None:
        _plot_signature(asset, out, Path(fig_path))
    return out


# --------------------------------------------------------------------------------------------- figures
def _figure(nrows: int = 1, width: float = 7.5, height: float = 3.6, sharex: bool = False):
    fig = Figure(figsize=(width, height), dpi=120, layout="constrained")
    FigureCanvasAgg(fig)  # Agg canvas: no pyplot / GUI backend involved
    axes = fig.subplots(nrows, 1, sharex=sharex, squeeze=False)[:, 0]
    for ax in axes:
        ax.grid(True, color=_GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_INK2)
        ax.tick_params(colors=_INK2, labelsize=8)
    return fig, axes


def _save(fig: Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")


def _plot_tz(asset: str, tz: pl.DataFrame, path: Path) -> None:
    fig, (ax,) = _figure()
    x = tz["bar_idx"].to_numpy()
    for tag, color, label in (("dst", _DST_COLOR, "NY daylight time"), ("std", _STD_COLOR, "NY standard time")):
        y = tz[f"mean_r2_{tag}"].fill_null(np.nan).to_numpy().astype(float)
        n = int(tz[f"n_{tag}"].max() or 0)
        ax.plot(x, y, color=color, linewidth=2, label=f"{label} (up to {n} sessions per bar)")
    ax.set_yscale("log")
    ax.set_xlabel("bar index in session (5-minute bars)", fontsize=9, color=_INK2)
    ax.set_ylabel("mean squared return (%²)", fontsize=9, color=_INK2)
    ax.set_title(f"{asset}: intraday variance profile, daylight vs standard time", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    _save(fig, path)


def _plot_signature(asset: str, sig: pl.DataFrame, path: Path) -> None:
    fig, (ax,) = _figure(width=5.5, height=3.4)
    x, y = sig["freq_min"].to_numpy(), sig["mean_rv"].to_numpy()
    ax.plot(x, y, color=_ASSET_COLOR.get(asset, _DST_COLOR), linewidth=2, marker="o", markersize=6)
    ax.set_xticks(x)
    ax.set_ylim(0, float(np.nanmax(y)) * 1.15 if len(y) and np.isfinite(y).any() else 1.0)
    ax.set_xlabel("sampling interval (minutes)", fontsize=9, color=_INK2)
    ax.set_ylabel("mean daily RV (%²)", fontsize=9, color=_INK2)
    ax.set_title(f"{asset}: volatility signature", fontsize=10, loc="left")
    _save(fig, path)


def _plot_coverage(monthly: dict[str, pl.DataFrame], path: Path) -> None:
    fig, axes = _figure(nrows=len(monthly), height=1.6 * len(monthly) + 0.6, sharex=True)
    for ax, (asset, m) in zip(axes, monthly.items()):
        share = (1 - m["null_bar_share"].fill_null(np.nan).to_numpy().astype(float)) * 100
        few = len(share) < 24  # short samples: show the points, a 1-month line would be invisible
        ax.plot(m["month"].to_numpy(), share, color=_ASSET_COLOR.get(asset, _DST_COLOR), linewidth=2,
                marker="o" if few else None, markersize=4)
        ax.set_ylim(min(80.0, float(np.nanmin(share)) - 2 if np.isfinite(share).any() else 80.0), 101)
        ax.set_title(f"{asset}: real 5-minute bars, % of scheduled", fontsize=9, loc="left")
    _save(fig, path)


# --------------------------------------------------------------------------------------------- report
def _fmt(v, digits: int | str | None = None) -> str:
    """Markdown cell text; ``digits`` = fixed decimals (int) or a format spec (str) for floats."""
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "-"
        if isinstance(digits, str):
            return format(v, digits)
        if digits is not None:
            return f"{v:,.{digits}f}"
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.4g}"
    if isinstance(v, datetime):
        return v.strftime("%Y-%m-%d %H:%M")
    if isinstance(v, date):
        return v.isoformat()
    return str(v)


def _md_table(df: pl.DataFrame, digits: dict[str, int | str] | None = None) -> str:
    digits = digits or {}
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "|".join("---" for _ in df.columns) + "|"]
    for row in df.iter_rows():
        lines.append("| " + " | ".join(_fmt(v, digits.get(c)) for c, v in zip(df.columns, row)) + " |")
    return "\n".join(lines)


def _pct(*cols: str) -> list[pl.Expr]:
    return [(pl.col(c) * 100).alias(f"{c.removesuffix('_share')}_%") for c in cols]


def _rel(target: Path, base: Path) -> str:
    try:
        return Path(os.path.relpath(target, base)).as_posix()
    except ValueError:  # different drives on Windows
        return Path(target).as_posix()


def _na(what: str, why: object) -> str:
    return f"_{what}: not available ({why})._\n"


def _peak_bar(tz: pl.DataFrame, col: str) -> int | None:
    v = tz.filter(pl.col(col).is_not_null()).sort(col, descending=True)
    return int(v["bar_idx"][0]) if v.height else None


def _section_coverage(assets, bronze_dir, silver_dir, flags_dir, gold, starts, schedule_fn, until, out_md, fig_dir,
                      table_dir):
    """Section 1 lines plus the per-asset session-treatment tables used by section 2."""
    md = [
        REPORT_SECTIONS[0],
        "",
        "Scheduled bars/minutes come from the session calendar (SPEC §4.1); real minutes are de-duplicated, "
        "minute-aligned real minutes inside the session window. *null bars* = scheduled 5-minute bars without a "
        f"real minute; *gaps* = intervals > {GAP_MIN} min between consecutive real minutes of a session "
        "(*max_gap_min* = longest). Raw timestamp problems come from the ingestion flag files (by UTC month of the "
        "flagged day), because the ingesters clean bronze before writing it: Binance *n_dup_raw* / *n_moved_raw* / "
        "*n_spacing_raw* = raw rows dropped as duplicates (last kept) / open times off the minute grid (floored) / "
        "open-time steps that are not a positive multiple of 60 s; Dukascopy *n_misaligned_days* / "
        "*n_missing_side_days* / *n_decode_error_days* = days whose BID and ASK minutes differ (common minutes "
        "kept) / with one side missing (no rows) / with an undecodable file or record times that are not "
        "increasing minute offsets (no rows). Spreads are in bp of mid over real in-session minutes (Dukascopy "
        "only; median of monthly medians, max of monthly p99; *crossed* = ask < bid). The monthly CSV also holds "
        "the session treatment counts and the jump-day share per month.",
        "",
    ]
    monthly: dict[str, pl.DataFrame] = {}
    issues: dict[str, pl.DataFrame] = {}
    for a in assets:
        md += [f"### {a}", ""]
        start = starts[a][0] if a in starts else None
        try:
            m = monthly_coverage(a, bronze_dir, silver_dir, schedule_fn, until, flags_dir=flags_dir, daily=gold,
                                 start=start)
            issues[a] = session_issues(a, silver_dir, schedule_fn, until, start)
        except FileNotFoundError as e:
            md.append(_na(a, e))
            continue
        monthly[a] = m
        sd = issues[a]["session_date"]
        if sd.len():
            md += [f"Sessions {sd.min()} to {sd.max()} ({sd.len()} scheduled).", ""]
        src = _source(a)
        flag_cols = [c for c, (s, _, _) in FLAG_COLUMNS.items() if s == src and c != "n_continuity_days"]
        cols = ["year", "n_sessions", "sched_bars", "null_bar_%", "real_min_%", "n_gaps_gt5", "max_gap_min",
                *flag_cols]
        if _is_dukascopy(a):
            cols += ["spread_med_bp", "spread_p99_bp", "n_crossed"]
        cols += ["n_dropped", "n_flagged"]
        ytab = yearly_coverage(m).with_columns(_pct("null_bar_share", "real_min_share")).select(cols)
        md += [_md_table(ytab, {"null_bar_%": 2, "real_min_%": 2, "spread_med_bp": 2, "spread_p99_bp": 2}), ""]
        ff = flags_file(a, flags_dir)
        if not ff.exists():
            md += [f"_Ingestion flag file `{ff.name}` not found: raw duplicate/misaligned counts unavailable._", ""]
        md += [
            f"Bronze integrity: {int(m['n_dup_bronze'].sum())} duplicate and {int(m['n_misaligned_bronze'].sum())} "
            "off-minute `ts` rows (expected 0: the ingester removes them).",
            "",
        ]
        worst = (
            m.filter(pl.col("null_bar_share") > 0)
            .sort(["null_bar_share", "month"], descending=[True, False])
            .head(5)
            .with_columns(_pct("null_bar_share"))
            .select("month", "n_sessions", "null_bar_%", "n_gaps_gt5", "n_dropped", "n_flagged")
        )
        if worst.is_empty():
            md += ["No month has null bars.", ""]
        else:
            md += ["Worst months by null-bar share:", "", _md_table(worst, {"null_bar_%": 2}), ""]
        if table_dir is not None:
            csv = Path(table_dir) / f"dq_monthly_{a}.csv"
            csv.parent.mkdir(parents=True, exist_ok=True)
            m.write_csv(csv)
            md += [f"Full monthly table: `{_rel(csv, out_md.parent)}`.", ""]
    if monthly:
        fig = fig_dir / "dq_coverage.png"
        _plot_coverage(monthly, fig)
        md += [f"![coverage]({_rel(fig, out_md.parent)})", ""]
    return md, issues


def _section_sessions(assets, issues, starts) -> list[str]:
    md = [
        REPORT_SECTIONS[1],
        "",
        "Treatment of every session in the gold table (SPEC §4.3, §5.3; first reason that applies). *missing* = "
        "scheduled but absent from silver; *no_price* = no open or close price (no real minute in the session); "
        f"*low_coverage* = coverage < {_min_cov():.2f}; *before_sample_start* = before the modelling start (SPX: "
        "probe of SPEC §1); *no_prev_close* = the first valid session, which has no previous close for `gap` and "
        "`r_cc`. EURUSD/SPX sessions are dropped for every reason; crypto sessions with low coverage are kept and "
        "flagged (low coverage is reported for crypto only when no other reason applies). Crypto sessions with fewer "
        "than 5 returns are flagged in the gold table (`flag_partial`) and not repeated here.",
        "",
    ]
    for a in assets:
        if a not in issues:
            md.append(_na(a, "no silver sessions"))
            continue
        bad = issues[a].filter(pl.col("reason").is_not_null())
        md += [f"### {a}: {bad.height} of {issues[a].height} sessions", ""]
        if a in starts:
            md += [f"Modelling sample starts {starts[a][0]} ({starts[a][1]}).", ""]
        if bad.is_empty():
            md += ["No dropped or flagged sessions.", ""]
            continue
        counts = (
            bad.group_by(pl.col("session_date").dt.year().alias("year"), "reason", "action")
            .agg(n=pl.len())
            .sort("year", "reason")
        )
        listed = bad.sort(["coverage", "session_date"]).head(15)
        listed = listed.select("session_date", "n_sched", "n_real_bars", "coverage", "reason", "action")
        md += [_md_table(counts), "", f"Lowest-coverage examples ({listed.height} of {bad.height}):", "",
               _md_table(listed, {"coverage": 3}), ""]
    return md


def _review_summary(review: pl.DataFrame) -> pl.DataFrame:
    """Suspect vs other months per asset: pooled jump-day share, zero-return share and the shares of jump days
    whose largest return is reversed / spans null bars / falls on a partial session (:func:`jump_review`)."""
    s = pl.col
    return (
        review.group_by("asset", months=pl.when(s("suspect")).then(pl.lit("suspect")).otherwise(pl.lit("other")))
        .agg(n_months=pl.len(), n_days=s("n_days").sum(), n_jump=s("n_jump").sum(), n_ret=s("n_ret").sum(),
             n_zero=s("n_zero").sum(), n_reversed=s("n_reversed").sum(), n_spanning=s("n_spanning").sum(),
             n_partial=pl.when(s("n_partial").is_not_null().any()).then(s("n_partial").sum()))
        .select(
            "asset", "months", "n_months", "n_days", "n_jump",
            (100 * s("n_jump") / s("n_days")).alias("share_%"),
            (100 * s("n_zero") / s("n_ret")).alias("zero_ret_%"),
            *[pl.when(s("n_jump") > 0).then(100 * s(f"n_{c}") / s("n_jump")).alias(f"{c}_%")
              for c in ("reversed", "spanning", "partial")],
        )
        .sort("asset", pl.col("months") == "other")
    )


def _section_jumps(assets, gold, silver_dir, until, starts) -> list[str]:
    md = [REPORT_SECTIONS[2], ""]
    if gold is None:
        return md + [_na("gold daily table", "missing")]
    g = gold.filter(pl.col("asset").is_in(list(assets)))
    by_month, by_year = jump_share(g, by="month"), jump_share(g, by="year")
    suspect = by_month.filter(pl.col("suspect")).sort("asset", "month")
    reviews, missing = [], []
    for a in assets:
        if g.filter(pl.col("asset") == a).is_empty():
            continue
        try:
            reviews.append(jump_review(a, g, silver_dir, until, start=starts[a][0] if a in starts else None))
        except FileNotFoundError as e:
            missing.append(_na(f"{a} jump review", e))
    review = pl.concat(reviews) if reviews else None
    md += [
        "Share of gold sessions with a significant jump (`j > 0`, BNS ratio test at 99.9%, SPEC §5.2) per asset "
        f"× month; a share above {JUMP_SHARE_SUSPECT:.0%} marks the month *suspect data* (SPEC §10): a flag for "
        "manual review, not a finding of bad data, and never a filter. Months with few sessions are noisy (see "
        "`n_days`). Every month is in the monthly CSV (`jump_share`, `jump_suspect`). "
        f"Suspect asset-months: {suspect.height} of {by_month.height}.",
        "",
        "Review aids (5-minute returns of the same gold sessions): *zero_ret_%* = share of returns exactly 0 "
        "(stale or coarse prices); on jump days, the session's largest |return| is *reversed* when an adjacent "
        f"return has the opposite sign and at least {REVERSAL_FRAC:.0%} of its size (the signature of an isolated "
        "bad print) and *spanning* when it spans null bars; *partial* = jump days on partial sessions. Suspect "
        "months whose jump days look like the asset's other months on these measures point to a property of the "
        "market or the 5-minute sampling rather than to feed errors; a confirmed data error is fixed at bronze.",
        "",
        *missing,
    ]
    if suspect.height:
        shown = suspect.head(60)
        if review is not None:
            shown = shown.join(
                review.select("asset", "month", "zero_ret_share", "n_reversed", "n_spanning", "n_partial"),
                on=["asset", "month"], how="left",
            ).with_columns(_pct("zero_ret_share"))
        cols = ["asset", "month", "n_days", "n_jump", "share_%"]
        cols += [c for c in ("zero_ret_%", "n_reversed", "n_spanning", "n_partial") if c in shown.columns]
        shown = shown.with_columns(_pct("share")).select(cols)
        md += [_md_table(shown, {"share_%": 1, "zero_ret_%": 2}), ""]
        if suspect.height > shown.height:
            md += [f"({suspect.height - shown.height} more in the monthly CSVs.)", ""]
    if review is not None and review.height:
        digits = {c: 1 for c in ("share_%", "reversed_%", "spanning_%", "partial_%")} | {"zero_ret_%": 2}
        md += ["Suspect vs other months per asset (pooled over the months):", "",
               _md_table(_review_summary(review), digits), ""]
    md += [
        "Per asset-year summary:",
        "",
        _md_table(by_year.with_columns(_pct("share")).drop("share"), {"share_%": 1}),
        "",
    ]
    return md


def _section_top_returns(assets, silver_dir, until, starts) -> list[str]:
    md = [
        REPORT_SECTIONS[3],
        "",
        "The 20 largest absolute 5-minute returns per asset in sessions that enter the gold table (valid, from the "
        "modelling start, with a previous close; `r` in %; `span_bars > 1` = one return across null bars; "
        "`ts_prev` = session open marks the first return from `p_open`). For manual review only; nothing is "
        "filtered.",
        "",
    ]
    for a in assets:
        md += [f"### {a}", ""]
        try:
            t = top_returns(a, silver_dir, 20, until, start=starts[a][0] if a in starts else None)
        except FileNotFoundError as e:
            md.append(_na(a, e))
            continue
        md += [_md_table(t.drop("asset"), {"r": 3, "p_prev": ".6g", "price": ".6g"}), ""]
    return md


def _section_cfd(assets, gold, implied_dir) -> list[str]:
    md = [REPORT_SECTIONS[4], ""]
    fred = newest_fred(implied_dir)
    if "SPX" not in assets:
        return md + ["_SPX is not part of this report._", ""]
    if gold is None or fred is None:
        return md + [_na("CFD check", "gold daily table or FRED SP500 snapshot missing")]
    res = cfd_check(gold, fred)
    over = res["days_over_25bp"]
    md += [
        f"SPX CFD `r_cc` vs FRED SP500 close-to-close log returns from {res['start']} (snapshot `{fred.name}`), "
        "aligned by NY trading date; a date is compared only when both previous observations coincide "
        f"({res['n_prev_mismatch']} dates excluded).",
        "",
        f"- correlation: **{_fmt(res['corr'], 5)}** (expected > 0.999) over n = {res['n']} days",
        f"- mean |difference|: {_fmt(res['mean_abs_diff_bp'], 2)} bp",
        f"- days with |difference| > {CFD_TOL_BP:.0f} bp: {len(over)}",
        "",
    ]
    if over:
        tab = pl.DataFrame(over)
        md.append(_md_table(tab.head(50), {"r_cc": 3, "r_fred": 3, "diff_bp": 1}))
        if tab.height > 50:
            md.append(f"\n({tab.height - 50} more not listed.)")
        md.append("")
    return md


def _section_tz(assets, silver_dir, until, starts, out_md, fig_dir) -> list[str]:
    md = [
        REPORT_SECTIONS[5],
        "",
        "Mean squared 5-minute return by session-local bar index over full-length gold sessions, split by whether "
        "New York observes daylight time on the session date. On a session-local clock (EURUSD, SPX) the profiles "
        "must line up "
        "(best-aligning shift 0); a shifted opening spike indicates a timezone bug. Crypto sessions are UTC days, "
        "so US-hours activity is expected 12 bars earlier in daylight time (shift +12). `corr_*` = correlation "
        "of the log profiles at the best shift and at shift 0 (a flat, noisy profile makes the shift meaningless).",
        "",
    ]
    rows, figs = [], []
    for a in assets:
        fig = fig_dir / f"dq_tz_{a}.png"
        try:
            tz = tz_check(a, silver_dir, fig, until, start=starts[a][0] if a in starts else None)
        except FileNotFoundError as e:
            md.append(_na(a, e))
            continue
        n_dst = int(tz["n_dst"].max() or 0) if tz.height else 0
        n_std = int(tz["n_std"].max() or 0) if tz.height else 0
        both = n_dst > 0 and n_std > 0
        al = profile_alignment(tz) if both else {"best_shift": None, "corr_best": None, "corr_0": None}
        rows.append({
            "asset": a,
            "sessions_dst": n_dst,
            "sessions_std": n_std,
            "peak_bar_dst": _peak_bar(tz, "mean_r2_dst") if both else None,
            "peak_bar_std": _peak_bar(tz, "mean_r2_std") if both else None,
            **al,
        })
        if fig.exists():
            figs.append(f"![tz {a}]({_rel(fig, out_md.parent)})")
    if rows:
        md += [_md_table(pl.DataFrame(rows), {"corr_best": 3, "corr_0": 3}), "", *figs, ""]
    return md


def _section_signature(assets, bronze_dir, schedule_fn, until, starts, out_md, fig_dir) -> list[str]:
    md = [
        REPORT_SECTIONS[6],
        "",
        "Mean daily RV (%²) when sampling real-minute closes every k minutes within sessions from the modelling "
        f"start (same sessions for every k; sessions with < {_min_cov():.0%} real minutes excluded). A strong rise "
        "at 1-2 minutes signals "
        "microstructure noise; 5 minutes is the project's sampling choice. `<asset> /5m` = mean RV relative to "
        "5-minute sampling.",
        "",
    ]
    tab = pl.DataFrame({"freq_min": list(SIGNATURE_FREQS)}, schema={"freq_min": pl.Int32})
    digits: dict[str, int | str] = {}
    n_sess, figs = [], []
    for a in assets:
        fig = fig_dir / f"dq_signature_{a}.png"
        try:
            sig = signature(a, bronze_dir, schedule_fn, SIGNATURE_FREQS, fig, until,
                            start=starts[a][0] if a in starts else None)
        except FileNotFoundError as e:
            md.append(_na(a, e))
            continue
        tab = tab.join(
            sig.select("freq_min", pl.col("mean_rv").alias(f"{a} RV"), pl.col("rel_5m").alias(f"{a} /5m")),
            on="freq_min", how="left",
        )
        digits |= {f"{a} RV": ".4g", f"{a} /5m": 3}
        n_sess.append(f"{a} {int(sig['n_sessions'].max() or 0)}")
        if fig.exists():
            figs.append(f"![signature {a}]({_rel(fig, out_md.parent)})")
    if n_sess:
        md += [_md_table(tab, digits), "", f"Sessions used: {', '.join(n_sess)}.", "", *figs, ""]
    return md


def _section_flags(assets, gold, flags_dir, until) -> list[str]:
    md = [REPORT_SECTIONS[7], "", "### Binance incident dates (flagged, never deleted)", ""]
    crypto = [a for a in assets if a in C.CRYPTO]
    if not crypto:
        md.append("_No crypto asset in this report._")
    elif gold is None:
        md.append(_na("gold daily table", "missing"))
    else:
        g = gold.filter(pl.col("asset").is_in(crypto)).with_columns(
            rv_pctile=pl.col("rv").rank("average").over("asset") / pl.len().over("asset") * 100
        )
        inc = g.filter(pl.col("session_date").is_in(list(BINANCE_INCIDENTS)))
        if inc.is_empty():
            dates = ", ".join(d.isoformat() for d in BINANCE_INCIDENTS)
            md.append(f"No gold sessions on the incident dates ({dates}) in this sample.")
        else:
            keep = [c for c in ("asset", "session_date", "coverage", "flag_partial", "M", "r_cc", "rv", "rv_pctile",
                                "j") if c in inc.columns]
            md.append(_md_table(inc.select(keep).sort("asset", "session_date"),
                                {"coverage": 3, "r_cc": 3, "rv": 3, "rv_pctile": 1, "j": 3}))
    md += ["", "### Ingestion flag files", ""]
    flags = read_flags(flags_dir, until)
    if "asset" in flags.columns:
        flags = flags.filter(pl.col("asset").is_null() | pl.col("asset").is_in(list(assets)))
    if flags.is_empty():
        md += [f"_No flag rows found (`{Path(flags_dir).name}/*.parquet`)._", ""]
        return md
    keys = ["source_file", *[c for c in ("asset", "flag") if c in flags.columns]]
    aggs = [pl.len().alias("n_flags")]
    if "n_rows" in flags.columns:
        aggs.append(pl.col("n_rows").cast(pl.Int64, strict=False).sum().alias("n_rows"))
    md += [
        "One flag row per flagged day (`n_flags`); `n_rows` sums the rows behind them (Binance: raw rows moved or "
        "dropped as duplicates, offending open-time steps, or real minutes in bronze on an incident date).",
        "",
        _md_table(flags.group_by(keys).agg(aggs).sort(keys)),
        "",
    ]
    md += _continuity_table(flags)
    other = flags.filter(pl.col("flag") != "continuity") if "flag" in flags.columns else flags
    md += ["### Other ingestion flags", "",
           f"Up to {FLAG_ROWS_PER_FILE} rows per flag file, by date (all rows are in the flag files).", ""]
    if other.is_empty():
        md += ["No other flags.", ""]
    for name in sorted(set(other["source_file"].to_list())):
        rows = other.filter(pl.col("source_file") == name)
        rows = rows.sort("date") if "date" in rows.columns else rows
        shown = rows.head(FLAG_ROWS_PER_FILE)
        shown = shown.select([c for c in shown.columns if c != "source_file" and shown[c].null_count() < shown.height])
        what = "all" if shown.height == rows.height else f"first {shown.height} by date"
        md += [f"#### `{name}` ({what} of {rows.height})", "", _md_table(shown), ""]
    return md


def _continuity_table(flags: pl.DataFrame) -> list[str]:
    """Dukascopy continuity flags (SPEC §2.2), largest |rel_change| first."""
    md = ["### Dukascopy continuity flags", ""]
    cont = flags.filter(pl.col("flag") == "continuity") if "flag" in flags.columns else flags.head(0)
    if cont.is_empty():
        return md + ["No continuity flags.", ""]
    if "rel_change" in cont.columns:
        rel = pl.col("rel_change").cast(pl.Float64, strict=False)
        cont = cont.with_columns((rel * 100).alias("rel_change_%")).sort(rel.abs(), descending=True, nulls_last=True)
    elif "date" in cont.columns:
        cont = cont.sort("date")
    cols = [c for c in ("asset", "date", "prev_date", "prev_mid", "first_mid", "rel_change_%", "detail")
            if c in cont.columns]
    shown = cont.head(CONTINUITY_MAX_ROWS).select(cols or cont.columns)
    what = "all" if shown.height == cont.height else f"the {shown.height} largest |rel_change|"
    md += [
        f"First real mid of a day more than 2% away from the previous real day's last mid (catches 0-based month "
        f"bugs; genuine large moves trigger it too). Listed: {what} of {cont.height}.",
        "",
        _md_table(shown, {"prev_mid": ".6g", "first_mid": ".6g", "rel_change_%": 2}),
        "",
    ]
    return md


def build_report(
    assets: Sequence[str] = C.ASSETS,
    out_md: Path = C.REPORTS / "data_quality.md",
    fig_dir: Path = C.FIGURES,
    table_dir: Path | None = C.TABLES,
    bronze_dir: Path = C.BRONZE,
    silver_dir: Path = C.SILVER,
    implied_dir: Path = C.RAW / "implied",
    flags_dir: Path | None = None,
    daily: pd.DataFrame | pl.DataFrame | None = None,
    schedule_fn: ScheduleFn | None = None,
    until: date | None = None,
) -> Path:
    """Write the data-quality report (SPEC §10) with figures ``dq_*.png`` and return its path.

    ``until`` defaults to ``config.dev_end()``; a holdout date raises :class:`volrisk.io.HoldoutSealedError` unless
    the holdout is unsealed (then ``daily`` defaults to dev + holdout rows). ``daily`` defaults to the dev gold
    table via ``volrisk.io.load_daily``; ``schedule_fn`` defaults to ``volrisk.sessions.session_schedule`` when
    importable (otherwise scheduled sessions absent from silver are not detected and the signature uses the silver
    sessions); ``flags_dir`` defaults to ``bronze_dir/flags``. The modelling start per asset is computed once from
    silver (:func:`sample_start`). Full monthly tables are written to ``table_dir`` as CSV (None: skipped). Missing
    inputs are reported as unavailable instead of failing the report.
    """
    until = _until(until)
    holdout = until >= C.holdout_start()  # only reachable once the holdout is unsealed
    out_md, fig_dir = Path(out_md), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    flags_dir = Path(flags_dir) if flags_dir is not None else Path(bronze_dir) / "flags"
    if schedule_fn is None:
        try:
            schedule_fn = default_schedule()
        except RuntimeError:
            log.warning("volrisk.sessions unavailable: scheduled sessions absent from silver are not detected")
    if daily is None:
        try:
            daily = io.load_daily(include_holdout=holdout)
        except FileNotFoundError:
            log.warning("gold daily table not found; gold-based checks are skipped")
    gold = _to_pl(daily).filter(pl.col("session_date") <= until) if daily is not None else None
    starts: dict[str, tuple[date, str]] = {}
    for a in assets:
        try:
            starts[a] = _start_info(a, silver_dir, until)
        except FileNotFoundError as e:
            log.warning("%s: modelling start unknown (%s)", a, e)

    scope = (
        f"including holdout sessions (holdout unsealed): sessions up to {until}" if holdout else
        f"from development data only: sessions up to {until} (the sealed holdout in `data/holdout/` is not read)"
    )
    md = [
        "# Data-quality report",
        "",
        f"Generated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC {scope}. Assets: {', '.join(assets)}. "
        "Nothing is filtered or corrected on the basis of this report; flagged items are for manual review.",
        "",
    ]
    cov, issues = _section_coverage(assets, bronze_dir, silver_dir, flags_dir, gold, starts, schedule_fn, until,
                                    out_md, fig_dir, table_dir)
    md += cov
    md += _section_sessions(assets, issues, starts)
    md += _section_jumps(assets, gold, silver_dir, until, starts)
    md += _section_top_returns(assets, silver_dir, until, starts)
    md += _section_cfd(assets, gold, implied_dir)
    md += _section_tz(assets, silver_dir, until, starts, out_md, fig_dir)
    md += _section_signature(assets, bronze_dir, schedule_fn or silver_schedule(silver_dir), until, starts, out_md,
                             fig_dir)
    md += _section_flags(assets, gold, flags_dir, until)

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md), encoding="utf-8")
    log.info("data-quality report -> %s", out_md)
    return out_md
