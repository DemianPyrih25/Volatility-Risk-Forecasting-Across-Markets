"""Binance Public Data spot 1m klines for BTC and ETH: download, checksum, parsing, bronze (SPEC §2.1, §2.3, §3).

Raw (immutable): ``data/raw/binance/{SYMBOL}/monthly/{SYMBOL}-1m-{YYYY}-{MM}.zip`` or, for a month whose monthly
zip is 404, every daily zip ``data/raw/binance/{SYMBOL}/daily/{SYMBOL}-1m-{YYYY}-{MM}-{DD}.zip`` — never both for
one month. Each zip sits next to its ``.CHECKSUM`` (sha256) file.
Bronze: ``data/bronze/minute/asset={ASSET}/year={YYYY}/part.parquet`` (``bid_close/ask_close/spread`` null).
Ingestion flags for the data-quality report (SPEC §2.3, §10) — rows moved by flooring, dropped duplicates,
60-second spacing violations, known incident dates and rejected files — go to
``data/bronze/flags/binance_{ASSET}.parquet``. Prices are USDT-quoted (documented limitation).
"""

from __future__ import annotations

import logging
import re
import zipfile
from calendar import monthrange
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl

from volrisk import config as C
from volrisk.data import http
from volrisk.io import write_parquet

log = logging.getLogger(__name__)

SOURCE = "binance"
BASE_URL = "https://data.binance.vision/data/spot"
RAW_DIR = C.RAW / SOURCE
BRONZE_DIR = C.BRONZE / "minute"
FLAGS_DIR = C.BRONZE / "flags"

# Flagged (never deleted) in the data-quality report (SPEC §2.3, §10).
KNOWN_INCIDENTS: tuple[date, ...] = (date(2018, 2, 8), date(2018, 2, 9), date(2019, 5, 15), date(2023, 3, 24))

US_THRESHOLD = 10**14  # open_time > 1e14 -> microseconds (spot klines from 2025-01), else milliseconds
MINUTE_US = 60_000_000
_US_PER_DAY = 86_400_000_000
_EPOCH = date(1970, 1, 1)
_TS = pl.Datetime("us", "UTC")

# Full kline CSV layout; only the first six columns and the trade count are kept.
_CSV_SCHEMA = {
    "open_time": pl.Int64,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "close_time": pl.Int64,
    "quote_volume": pl.Float64,
    "n_trades": pl.Float64,
    "taker_buy_base": pl.Float64,
    "taker_buy_quote": pl.Float64,
    "ignore": pl.Utf8,
}
CANDLE_SCHEMA = {
    "ts": _TS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "n_trades": pl.Float64,
    "is_real": pl.Boolean,
}
# parse_zip columns plus the unfloored open time in µs (to log rows moved by flooring).
_KLINE_SCHEMA = {"ts": _TS, "ts_raw": pl.Int64, **{k: v for k, v in CANDLE_SCHEMA.items() if k != "ts"}}
# Same column order as every bronze writer (SPEC §3).
BRONZE_SCHEMA = {
    "ts": _TS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "is_real": pl.Boolean,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread": pl.Float64,
    "n_trades": pl.Float64,
}
# Common columns (asset, date, flag, detail) match the Dukascopy flag file, so quality.read_flags can stack both.
FLAG_SCHEMA = {
    "asset": pl.Utf8,
    "date": pl.Date,  # UTC day; for rejected_file the first day of the file's period
    "flag": pl.Utf8,  # moved | duplicate | spacing | incident | rejected_file
    # moved / duplicate: raw rows that day; spacing: offending steps; incident: real minutes in bronze;
    # rejected_file: null
    "n_rows": pl.Int64,
    "detail": pl.Utf8,
}

FetchFn = Callable[..., list[http.ManifestRow]]


class RejectedFilesError(RuntimeError):
    """Raw zips failed checksum verification or parsing; bronze was left untouched."""

    def __init__(self, asset: str, rejected: dict[str, str]):
        self.asset, self.rejected = asset, dict(rejected)
        shown = "; ".join(f"{k}: {v}" for k, v in list(rejected.items())[:10])
        super().__init__(
            f"binance bronze {asset}: {len(rejected)} raw file(s) rejected ({shown}); nothing written. Re-run the "
            "download (it fetches missing CHECKSUMs and re-downloads mismatches) or pass allow_rejected=True to "
            "build without those periods."
        )


# --------------------------------------------------------------------------------------------- urls & paths


def monthly_url(symbol: str, year: int, month: int) -> str:
    return f"{BASE_URL}/monthly/klines/{symbol}/1m/{symbol}-1m-{year:04d}-{month:02d}.zip"


def daily_url(symbol: str, day: date) -> str:
    return f"{BASE_URL}/daily/klines/{symbol}/1m/{symbol}-1m-{day.isoformat()}.zip"


def checksum_url(zip_url: str) -> str:
    return zip_url + ".CHECKSUM"


def checksum_path(zip_path: Path) -> Path:
    return zip_path.with_name(zip_path.name + ".CHECKSUM")


def raw_path(url: str, raw_dir: Path | None = None) -> Path:
    """Local path of a zip (or CHECKSUM) url: ``{raw_dir}/{SYMBOL}/{monthly|daily}/{file}``."""
    root = Path(raw_dir) if raw_dir is not None else RAW_DIR
    parts = url.split("/")  # .../{monthly|daily}/klines/{SYM}/1m/{file}
    return root / parts[-3] / parts[-5] / parts[-1]


def bronze_path(asset: str, year: int, out_dir: Path | None = None) -> Path:
    root = Path(out_dir) if out_dir is not None else BRONZE_DIR
    return root / f"asset={asset}" / f"year={year}" / "part.parquet"


def flags_path(asset: str, flags_dir: Path | None = None) -> Path:
    root = Path(flags_dir) if flags_dir is not None else FLAGS_DIR
    return root / f"{SOURCE}_{asset}.parquet"


def _symbol(asset: str) -> str:
    a = C.asset(asset)
    if a.source != SOURCE:
        raise ValueError(f"{asset} is not a Binance asset (source={a.source})")
    return a.symbol


def _file_re(symbol: str) -> re.Pattern[str]:
    return re.compile(rf"^{re.escape(symbol)}-1m-(\d{{4}})-(\d{{2}})(?:-(\d{{2}}))?\.zip$")


def _file_period(name: str) -> tuple[date, date] | None:
    """Nominal UTC period ``[first, last]`` of a kline zip/csv from its file name (None if unrecognised)."""
    m = re.search(r"-1m-(\d{4})-(\d{2})(?:-(\d{2}))?\.(?:zip|csv)$", name)
    if not m:
        return None
    y, mo = int(m[1]), int(m[2])
    if m[3]:
        d = date(y, mo, int(m[3]))
        return d, d
    return date(y, mo, 1), date(y, mo, monthrange(y, mo)[1])


# --------------------------------------------------------------------------------------------- checksum


def read_checksum(path: Path) -> str:
    """Expected sha256 from a Binance ``.CHECKSUM`` file (``<sha256>  <file name>``)."""
    text = Path(path).read_text(encoding="ascii", errors="replace").strip()
    return text.split()[0].lower() if text else ""


def verify_checksum(zip_path: Path, checksum_file: Path | None = None) -> bool:
    """True iff ``zip_path`` and its ``.CHECKSUM`` exist and the sha256 matches."""
    zip_path = Path(zip_path)
    checksum_file = Path(checksum_file) if checksum_file is not None else checksum_path(zip_path)
    if not (zip_path.exists() and checksum_file.exists()):
        return False
    expected = read_checksum(checksum_file)
    return bool(expected) and http.sha256_file(zip_path) == expected


# --------------------------------------------------------------------------------------------- download


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def _months(first: date, last: date) -> Iterator[tuple[int, int]]:
    y, m = first.year, first.month
    while (y, m) <= (last.year, last.month):
        yield y, m
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)


def _month_end(y: int, m: int) -> date:
    return date(y, m, monthrange(y, m)[1])


def _days(y: int, m: int, first: date, last: date) -> list[date]:
    lo, hi = max(first, date(y, m, 1)), min(last, _month_end(y, m))
    return [lo + timedelta(days=k) for k in range((hi - lo).days + 1)]


@contextmanager
def _manifest_under(raw_dir: Path | None) -> Iterator[None]:
    """Redirect ``http.fetch_many``'s module-level manifest to ``raw_dir`` for non-default raw dirs.

    Keeps scratch/test downloads from writing ``data/raw/manifest.parquet``. Process-global while active.
    """
    if raw_dir is None or Path(raw_dir).resolve() == RAW_DIR.resolve():
        yield
        return
    old = http.MANIFEST
    http.MANIFEST = Path(raw_dir) / "manifest.parquet"
    try:
        yield
    finally:
        http.MANIFEST = old


def _fetch_verified(urls: list[str], raw_dir: Path | None, fetch: FetchFn, workers: int) -> dict[str, str]:
    """Fetch zips + CHECKSUMs that are not on disk yet, verify every zip, re-download a mismatch once.

    Returns ``{zip_url: status}`` with status ``ok`` (verified on disk), ``missing`` (404) or ``error``.
    """

    def jobs_for(us: list[str], force: bool = False) -> list[http.Job]:
        out = []
        for u in us:
            for v in (u, checksum_url(u)):
                p = raw_path(v, raw_dir)
                if force or not p.exists():
                    out.append(http.Job(SOURCE, v, p))
        return out

    def run(js: list[http.Job]) -> dict[str, str]:
        if not js:
            return {}
        # Local files are the cache (they honour raw_dir); 404s are re-checked because Binance publishes
        # a monthly zip a few days after the month ends.
        return {r.url: r.status for r in fetch(js, workers=workers, skip_done=False)}

    fetched = run(jobs_for(urls))
    status: dict[str, str] = {}
    retry: list[str] = []
    for u in urls:
        z, c = raw_path(u, raw_dir), raw_path(checksum_url(u), raw_dir)
        if fetched.get(u) == "missing":
            status[u] = "missing"
        elif not z.exists() or not c.exists():
            log.error("binance: %s or its CHECKSUM unavailable (%s / %s)", z.name, fetched.get(u),
                      fetched.get(checksum_url(u)))
            status[u] = "error"
        elif verify_checksum(z, c):
            status[u] = "ok"
        else:
            log.warning("binance: checksum mismatch for %s; deleting and re-downloading once", z.name)
            z.unlink(missing_ok=True)
            c.unlink(missing_ok=True)
            retry.append(u)
    if retry:
        refetched = run(jobs_for(retry, force=True))
        for u in retry:
            z = raw_path(u, raw_dir)
            if verify_checksum(z):
                status[u] = "ok"
            else:
                log.error("binance: %s failed checksum twice (%s); removed", z.name, refetched.get(u))
                z.unlink(missing_ok=True)
                status[u] = "error"
    return status


def download(
    asset: str,
    start: date | None = None,
    end: date | None = None,
    workers: int = 16,
    raw_dir: Path | None = None,
    *,
    today: date | None = None,
    fetch: FetchFn | None = None,
) -> dict:
    """Download and verify the 1m klines of ``asset`` (default: asset start .. config ``data_end``).

    A month is fetched as one monthly zip when its last day is ``<= end`` and before today (UTC); if that zip
    is 404 — or the month is incomplete — every daily zip of the month with ``day <= end`` and
    ``day < today`` is used instead. ``fetch`` defaults to ``volrisk.data.http.fetch_many`` (stubbed in tests).
    """
    sym = _symbol(asset)
    start = start or C.asset(asset).start
    end = end or C.data_end()
    today = today or _utc_today()
    fetch = fetch or http.fetch_many
    last = min(end, today - timedelta(days=1))  # never request the current UTC day or later
    months = list(_months(start, last)) if start <= last else []

    as_monthly = [ym for ym in months if _month_end(*ym) <= last]
    m_urls = {ym: monthly_url(sym, *ym) for ym in as_monthly}
    with _manifest_under(raw_dir):
        m_status = _fetch_verified(list(m_urls.values()), raw_dir, fetch, workers)
        as_daily = [ym for ym in months if ym not in m_urls or m_status[m_urls[ym]] == "missing"]
        d_urls = {d: daily_url(sym, d) for ym in as_daily for d in _days(*ym, start, last)}
        d_status = _fetch_verified(list(d_urls.values()), raw_dir, fetch, workers)

    def key(ym: tuple[int, int]) -> str:
        return f"{ym[0]:04d}-{ym[1]:02d}"

    fallback = [ym for ym in as_daily if ym in m_urls]
    errors = [u.rsplit("/", 1)[-1] for u, s in {**m_status, **d_status}.items() if s == "error"]
    missing_days = [d.isoformat() for d, u in d_urls.items() if d_status[u] == "missing"]
    counts = Counter(d_status.values())
    summary = {
        "asset": asset,
        "symbol": sym,
        "start": start.isoformat(),
        "end": last.isoformat() if months else None,
        "months": len(months),
        "monthly_ok": sum(s == "ok" for s in m_status.values()),
        "daily_months": [key(ym) for ym in as_daily],
        "monthly_404_fallback": [key(ym) for ym in fallback],
        "daily_ok": counts.get("ok", 0),
        "daily_missing": missing_days,
        "errors": errors,
    }
    log.info("binance download %s: %s", asset, summary)
    return summary


# --------------------------------------------------------------------------------------------- parsing


def _read_csv_bytes(raw: bytes) -> pl.DataFrame:
    if raw[:3] == b"\xef\xbb\xbf":
        raw = raw[3:]
    if raw and not raw[:1].isdigit():  # header iff the first character is not a digit
        raw = raw.split(b"\n", 1)[1] if b"\n" in raw else b""
    if not raw.strip():
        return pl.DataFrame(schema={k: _CSV_SCHEMA[k] for k in ("open_time", "open", "high", "low", "close",
                                                                 "volume", "n_trades")})
    return pl.read_csv(raw, has_header=False, schema=_CSV_SCHEMA).select(
        "open_time", "open", "high", "low", "close", "volume", "n_trades"
    )


def _read_klines(path: Path) -> pl.DataFrame:
    """Parsed klines of one zip with ``ts`` (floored minute open) and ``ts_raw`` (unfloored, µs since epoch)."""
    path = Path(path)
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: expected exactly one CSV, found {names}")
        raw = zf.read(names[0])
    df = _read_csv_bytes(raw)
    if df.height:
        ot = df["open_time"]
        is_us = ot > US_THRESHOLD
        if is_us.any() and not is_us.all():
            raise ValueError(f"{path.name}: mixed ms/µs open_time values in one file")
        if not is_us.any():
            df = df.with_columns(pl.col("open_time") * 1000)
    df = df.rename({"open_time": "ts_raw"}).with_columns(
        (pl.col("ts_raw") - pl.col("ts_raw") % MINUTE_US).cast(_TS).alias("ts"),
        (pl.col("volume") > 0).alias("is_real"),
    )
    period = _file_period(path.name)
    if period is not None and df.height:
        lo = (period[0] - _EPOCH).days * _US_PER_DAY
        hi = ((period[1] - _EPOCH).days + 1) * _US_PER_DAY
        n_out = df.filter((pl.col("ts_raw") < lo) | (pl.col("ts_raw") >= hi)).height
        if n_out:  # a wrong ms/µs decision lands decades away from the file's period
            raise ValueError(f"{path.name}: {n_out} rows outside the file period {period} (timestamp unit?)")
    return df.select(list(_KLINE_SCHEMA))


def parse_zip(path: Path) -> pl.DataFrame:
    """One Binance kline zip -> ``ts`` (Datetime us UTC, minute open), OHLC, volume, n_trades, is_real.

    Header iff the first character is not a digit; unit per file: µs if ``open_time > 1e14`` else ms;
    ``ts`` is the open time floored to the minute (deduplication happens in ``build_bronze``).
    """
    return _read_klines(path).select(list(CANDLE_SCHEMA))


# --------------------------------------------------------------------------------------------- bronze


def select_files(asset: str, raw_dir: Path | None = None, start: date | None = None,
                 end: date | None = None) -> dict[str, list[Path]]:
    """Raw zips to use per month ``YYYY-MM`` in ``[start, end]``: the monthly zip if present, else its dailies."""
    sym = _symbol(asset)
    root = (Path(raw_dir) if raw_dir is not None else RAW_DIR) / sym
    start = start or C.asset(asset).start
    end = end or C.data_end()
    rx = _file_re(sym)
    monthly: dict[str, Path] = {}
    daily: dict[str, list[Path]] = defaultdict(list)
    for kind in ("monthly", "daily"):
        for p in (root / kind).glob(f"{sym}-1m-*.zip"):
            m = rx.match(p.name)
            if not m or bool(m[3]) != (kind == "daily"):
                continue
            k = f"{m[1]}-{m[2]}"
            if kind == "monthly":
                monthly[k] = p
            elif start <= date(int(m[1]), int(m[2]), int(m[3])) <= end:
                daily[k].append(p)
    lo, hi = f"{start:%Y-%m}", f"{end:%Y-%m}"
    out: dict[str, list[Path]] = {}
    for k in sorted(set(monthly) | set(daily)):
        if not lo <= k <= hi:
            continue
        if k in monthly:
            if daily.get(k):
                log.info("binance %s %s: monthly zip present, ignoring %d daily zips", asset, k, len(daily[k]))
            out[k] = [monthly[k]]
        else:
            out[k] = sorted(daily[k])
    return out


def _load(path: Path, verify: bool) -> pl.DataFrame | str:
    if verify and not verify_checksum(path):
        return "checksum missing or mismatched"
    try:
        return _read_klines(path)
    except (ValueError, zipfile.BadZipFile, pl.exceptions.PolarsError) as exc:
        return f"parse error: {exc}"


def _gaps(ts: pl.Series) -> dict:
    """Missing-minute runs in the final (sorted, unique, whole-minute) bronze ``ts``."""
    d = ts.cast(pl.Int64).diff().drop_nulls()
    gaps = d.filter(d > MINUTE_US) // MINUTE_US - 1
    return {
        "n_gaps": gaps.len(),
        "gap_minutes": int(gaps.sum()) if gaps.len() else 0,
        "max_gap_minutes": int(gaps.max()) if gaps.len() else 0,
    }


def _spacing_violations(df: pl.DataFrame) -> pl.DataFrame:
    """Rows whose step from the previous row of the same file is not a positive multiple of 60 s.

    This is SPEC §2.3's "60-second spacing (except gaps)" check, run on the converted, *unfloored* open times in
    file order. Violations are reported (flag ``spacing``), not raised: real files contain them (e.g. the 2018-02
    incident's open times at hh:mm:14.789) and a raise would drop the whole file. A wrong ms/µs decision is
    caught as a hard error by ``_read_klines``' period check.
    """
    step = pl.col("ts_raw") - pl.col("prev_raw")
    return (
        df.with_columns(prev_raw=pl.col("ts_raw").shift(1).over("_file"))
        .filter(pl.col("prev_raw").is_not_null() & ((step <= 0) | (step % MINUTE_US != 0)))
    )


def _fmt(e: pl.Expr) -> pl.Expr:
    return e.cast(_TS).dt.strftime("%Y-%m-%dT%H:%M:%S%.3f")


def _day_flags(asset: str, flag: str, rows: pl.DataFrame, what: str, example: pl.Expr) -> pl.DataFrame:
    """One flag row per UTC day of ``rows`` (by floored ``ts``): row count and the first example in file order."""
    if not rows.height:
        return pl.DataFrame(schema=FLAG_SCHEMA)
    return (
        rows.sort("_ord")
        .group_by(pl.col("ts").dt.date().alias("date"), maintain_order=True)
        .agg(n_rows=pl.len().cast(pl.Int64), ex=example.first())
        .select(
            pl.lit(asset).alias("asset"),
            "date",
            pl.lit(flag).alias("flag"),
            "n_rows",
            pl.format("{}: {}, e.g. {}", pl.lit(what), "n_rows", "ex").alias("detail"),
        )
    )


def build_bronze(
    asset: str,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    flags_dir: Path | None = None,
    *,
    start: date | None = None,
    end: date | None = None,
    verify: bool = True,
    allow_rejected: bool = False,
    workers: int = 8,
) -> dict:
    """Parse the selected raw zips of ``asset``; write ``asset={ASSET}/year={YYYY}/part.parquet`` and the flags.

    Rows are deduplicated on the floored ``ts`` keeping the last (file order, then row order). Only
    ``[start, end]`` (default: asset start .. data_end) is kept; stale year partitions of the asset are removed.
    Rows moved by flooring, dropped duplicates, 60-second spacing violations (per file, on the unfloored open
    times), the known incident dates in range and rejected files are written to ``flags_path(asset, flags_dir)``;
    ``flags_dir`` defaults to the ``flags`` sibling of ``out_dir`` (``data/bronze/flags`` for the default).

    With ``verify`` every zip is checked against its ``.CHECKSUM`` first. If any zip fails verification or
    parsing, :class:`RejectedFilesError` is raised and nothing is written — a silently missing month would break
    the continuous crypto calendar (SPEC §4.3). ``allow_rejected=True`` builds without those files instead (their
    periods are absent from bronze and listed as ``rejected_file`` flags).
    """
    start = start or C.asset(asset).start
    end = end or C.data_end()
    if flags_dir is None:  # keep scratch/test builds (non-default out_dir) away from data/bronze/flags
        flags_dir = Path(out_dir).parent / "flags" if out_dir is not None else FLAGS_DIR
    by_month = select_files(asset, raw_dir, start, end)
    paths = [p for ps in by_month.values() for p in ps]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        loaded = list(ex.map(lambda p: _load(p, verify), paths))
    rejected = {p.name: r for p, r in zip(paths, loaded) if isinstance(r, str)}
    for name, reason in rejected.items():
        log.error("binance bronze %s: rejected %s (%s)", asset, name, reason)
    if rejected and not allow_rejected:
        raise RejectedFilesError(asset, rejected)
    frames = [r.with_columns(pl.lit(i, pl.UInt32).alias("_file"))
              for i, r in enumerate(loaded) if isinstance(r, pl.DataFrame) and r.height]

    df = pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema={**_KLINE_SCHEMA, "_file": pl.UInt32})
    t0 = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    t1 = datetime(end.year, end.month, end.day, tzinfo=timezone.utc) + timedelta(days=1)
    df = df.with_row_index("_ord").filter((pl.col("ts") >= t0) & (pl.col("ts") < t1))
    n_raw = df.height

    spacing = _spacing_violations(df).with_columns(
        pl.col("_file").replace_strict({i: p.name for i, p in enumerate(paths)}, return_dtype=pl.Utf8).alias("file")
    )
    if spacing.height:
        log.warning("binance bronze %s: %d steps between open times are not a positive multiple of 60 s, e.g. %s",
                    asset, spacing.height, spacing.head(3).select(_fmt(pl.col("prev_raw")), _fmt(pl.col("ts_raw")),
                                                                  "file").rows())
    moved = df.filter(pl.col("ts").cast(pl.Int64) != pl.col("ts_raw"))
    if moved.height:
        log.warning("binance bronze %s: %d rows moved when floored to the minute, e.g. %s", asset, moved.height,
                    moved.head(5).select(_fmt(pl.col("ts_raw")), _fmt(pl.col("ts"))).rows())
    df = df.sort("ts", "_ord")
    dup = df.filter(~pl.col("ts").is_last_distinct())
    df = df.filter(pl.col("ts").is_last_distinct())
    if dup.height:
        log.warning("binance bronze %s: dropped %d duplicate-ts rows (kept last)", asset, dup.height)
    gaps = _gaps(df["ts"])

    out = df.with_columns(
        pl.lit(None, dtype=pl.Float64).alias("bid_close"),
        pl.lit(None, dtype=pl.Float64).alias("ask_close"),
        pl.lit(None, dtype=pl.Float64).alias("spread"),
    ).select([pl.col(k).cast(v) for k, v in BRONZE_SCHEMA.items()])
    day = out.filter(pl.col("is_real")).group_by(pl.col("ts").dt.date().alias("d")).len()
    real_by_day = dict(zip(day["d"].to_list(), day["len"].to_list()))
    incidents = {d: real_by_day.get(d, 0) for d in KNOWN_INCIDENTS if start <= d <= end}
    flags = _flags(asset, moved, dup, spacing, incidents, rejected).sort("date", "flag")

    parts = out.with_columns(pl.col("ts").dt.year().alias("_y")).partition_by("_y", as_dict=True, include_key=False)
    years = sorted(int(y) for (y,) in parts)
    fpath = flags_path(asset, flags_dir)
    if not years:  # nothing parsed (wrong raw_dir / not downloaded): leave any existing bronze and flags untouched
        log.error("binance bronze %s: no rows in %d files; nothing written", asset, len(paths))
    for (year,), part in parts.items():
        write_parquet(part, bronze_path(asset, int(year), out_dir))
    if years:
        write_parquet(flags, fpath)
        for stale in bronze_path(asset, 0, out_dir).parent.parent.glob("year=*/part.parquet"):
            if stale.parent.name.removeprefix("year=") not in {str(y) for y in years}:
                log.info("binance bronze %s: removing stale partition %s", asset, stale.parent.name)
                stale.unlink()

    def per_month(frame: pl.DataFrame) -> dict[str, int]:
        if not frame.height:
            return {}
        g = frame.group_by(pl.col("ts").dt.strftime("%Y-%m").alias("m")).len().sort("m")
        return dict(zip(g["m"].to_list(), g["len"].to_list()))

    summary = {
        "asset": asset,
        "symbol": _symbol(asset),
        "months": len(by_month),
        "daily_months": [k for k, ps in by_month.items() if ps[0].parent.name == "daily"],
        "files": len(paths),
        "rejected": rejected,
        "rows_raw": n_raw,
        "moved": moved.height,
        "moved_by_month": per_month(moved),
        "duplicates": dup.height,
        "duplicates_by_month": per_month(dup),
        "spacing_violations": spacing.height,
        "rows": out.height,
        "real": int(out["is_real"].sum()) if out.height else 0,
        "first_ts": out["ts"].min().isoformat() if out.height else None,
        "last_ts": out["ts"].max().isoformat() if out.height else None,
        **gaps,
        "years": years,
        "incidents": {d.isoformat(): n for d, n in incidents.items()},
        "flag_rows": flags.height,
        "flags_path": str(fpath) if years else None,
    }
    log.info("binance bronze %s: %s", asset, {k: v for k, v in summary.items()
                                               if k not in ("moved_by_month", "duplicates_by_month")})
    return summary


def _flags(asset: str, moved: pl.DataFrame, dup: pl.DataFrame, spacing: pl.DataFrame, incidents: dict[date, int],
           rejected: dict[str, str]) -> pl.DataFrame:
    """Flag rows of one build (SPEC §2.3 "log moved rows", "flag known incident dates"; §10 dup/misaligned)."""
    other = []
    for d, n in incidents.items():
        other.append({"asset": asset, "date": d, "flag": "incident", "n_rows": n,
                      "detail": f"known Binance incident date; {n} real minutes in bronze (flagged, never deleted)"})
    for name, reason in rejected.items():
        period = _file_period(name)
        lo, hi = period if period is not None else (None, None)
        other.append({"asset": asset, "date": lo, "flag": "rejected_file", "n_rows": None,
                      "detail": f"{name}: {reason}; {lo}..{hi} absent from bronze"})
    return pl.concat(
        [
            _day_flags(asset, "moved", moved, "open times off the minute grid (floored)",
                       pl.format("{} -> {}", _fmt(pl.col("ts_raw")), _fmt(pl.col("ts")))),
            _day_flags(asset, "duplicate", dup, "duplicate minutes dropped (kept the last)", _fmt(pl.col("ts"))),
            _day_flags(asset, "spacing", spacing, "open-time steps not a positive multiple of 60 s",
                       pl.format("{} -> {} in {}", _fmt(pl.col("prev_raw")), _fmt(pl.col("ts_raw")), "file")),
            pl.DataFrame(other, schema=FLAG_SCHEMA),
        ],
        how="vertical",
    )
