"""Dukascopy minute candles: URLs, download jobs, ``.bi5`` decoding and the bronze layer (SPEC §2.2, §3).

Raw (immutable): ``data/raw/dukascopy/{INSTR}/{YYYY}/{YYYY-MM-DD}_{SIDE}.bi5`` — one LZMA file per UTC day and
side (BID/ASK). Bronze: ``data/bronze/minute/asset={ASSET}/year={YYYY}/part.parquet`` with the mid OHLC of
BID and ASK. Continuity / missing-side / misaligned / decode flags go to
``data/bronze/flags/dukascopy_{ASSET}.parquet``. Bronze mirrors the raw files: a rebuild removes year partitions
that no longer yield any row.
"""

from __future__ import annotations

import logging
import lzma
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import numpy as np
import polars as pl

from volrisk import config as C
from volrisk.data import http
from volrisk.io import write_parquet

log = logging.getLogger(__name__)

SOURCE = "dukascopy"
BASE_URL = "http://datafeed.dukascopy.com/datafeed"  # http only: https times out (SPEC §2.2)
SIDES = ("BID", "ASK")
RAW_DIR = C.RAW / SOURCE
BRONZE_DIR = C.BRONZE / "minute"
FLAGS_DIR = C.BRONZE / "flags"
CONTINUITY_TOL = 0.02
STATUSES = ("ok", "empty", "missing", "error")

# One 24-byte big-endian record per minute. NB the price fields are open, CLOSE, LOW, HIGH — not OHLC.
RECORD = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])

_TS = pl.Datetime("us", "UTC")
CANDLE_SCHEMA = {
    "ts": _TS,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
}
BRONZE_SCHEMA = {
    **CANDLE_SCHEMA,
    "is_real": pl.Boolean,
    "bid_close": pl.Float64,
    "ask_close": pl.Float64,
    "spread": pl.Float64,
    "n_trades": pl.Float64,  # Binance-only column, null here (SPEC §3)
}
FLAG_SCHEMA = {
    "asset": pl.Utf8,
    "date": pl.Date,
    "flag": pl.Utf8,  # continuity | missing_side | misaligned | decode_error
    "prev_date": pl.Date,
    "prev_mid": pl.Float64,
    "first_mid": pl.Float64,
    "rel_change": pl.Float64,
    "detail": pl.Utf8,
}
_FILE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_(BID|ASK)\.bi5$")
_US_PER_DAY = 86_400_000_000
_EPOCH = date(1970, 1, 1)


# ---------------------------------------------------------------------------------------------- paths & URLs


def _side(side: str) -> str:
    s = side.upper()
    if s not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")
    return s


def _instrument(asset: str) -> tuple[str, float]:
    a = C.asset(asset)
    if a.source != SOURCE or a.price_scale is None:
        raise ValueError(f"{asset} is not a Dukascopy asset with a price scale")
    return a.symbol, a.price_scale


def url(instr: str, day: date, side: str) -> str:
    """Datafeed URL of one day's minute candles; the month in the path is **0-based**."""
    return (
        f"{BASE_URL}/{instr}/{day.year:04d}/{day.month - 1:02d}/{day.day:02d}/{_side(side)}_candles_min_1.bi5"
    )


def raw_path(instr: str, day: date, side: str, raw_dir: Path | None = None) -> Path:
    root = Path(raw_dir) if raw_dir is not None else RAW_DIR
    return root / instr / f"{day.year:04d}" / f"{day.isoformat()}_{_side(side)}.bi5"


def bronze_path(asset: str, year: int, out_dir: Path | None = None) -> Path:
    root = Path(out_dir) if out_dir is not None else BRONZE_DIR
    return root / f"asset={asset}" / f"year={year}" / "part.parquet"


def flags_path(asset: str, flags_dir: Path | None = None) -> Path:
    root = Path(flags_dir) if flags_dir is not None else FLAGS_DIR
    return root / f"{SOURCE}_{asset}.parquet"


# ---------------------------------------------------------------------------------------------- download


def _utc_today() -> date:
    return datetime.now(timezone.utc).date()


def jobs(asset: str, start: date, end: date, raw_dir: Path | None = None) -> list[http.Job]:
    """BID and ASK jobs for every non-Saturday day in ``[start, end]``.

    Capped at the config ``data_end`` and at yesterday (UTC): the current day's file is never requested.
    """
    instr, _ = _instrument(asset)
    last = min(end, C.data_end(), _utc_today() - timedelta(days=1))
    out: list[http.Job] = []
    d = start
    while d <= last:
        if d.weekday() != 5:  # Saturday: market closed
            out.extend(http.Job(SOURCE, url(instr, d, s), raw_path(instr, d, s, raw_dir)) for s in SIDES)
        d += timedelta(days=1)
    return out


@contextmanager
def _manifest_under(raw_dir: Path | None) -> Iterator[None]:
    """``http.fetch_many`` keeps its manifest at a module-level path (``data/raw/manifest.parquet``).

    For a non-default ``raw_dir`` (tests, scratch runs) the manifest is redirected to ``raw_dir`` so nothing is
    written under ``data/raw``. Process-global while active, so do not run concurrent downloads through it.
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


def download(
    asset: str,
    start: date | None = None,
    end: date | None = None,
    workers: int = 16,
    raw_dir: Path | None = None,
) -> dict:
    """Fetch all BID/ASK day files of ``asset`` (default: asset start .. data_end). Returns counts by status."""
    start = start or C.asset(asset).start
    end = end or C.data_end()
    js = jobs(asset, start, end, raw_dir)
    with _manifest_under(raw_dir):
        rows = http.fetch_many(js, workers=workers)
    counts = Counter(r.status for r in rows)
    summary = {"asset": asset, "jobs": len(js), "skipped": len(js) - len(rows)}
    summary.update({s: counts.get(s, 0) for s in STATUSES})
    log.info("dukascopy download %s: %s", asset, summary)
    return summary


# ---------------------------------------------------------------------------------------------- decoding


def _records(raw: bytes) -> np.ndarray:
    """Decompress one ``.bi5`` body into validated RECORD rows (0-byte body -> no rows)."""
    if not raw:
        return np.empty(0, dtype=RECORD)
    buf = lzma.decompress(raw)
    if len(buf) % RECORD.itemsize:
        raise ValueError(f"decompressed size {len(buf)} is not a multiple of {RECORD.itemsize}")
    rec = np.frombuffer(buf, dtype=RECORD)
    t = rec["t"]
    if rec.size and (t[0] < 0 or t[-1] >= 86_400 or np.any(t % 60) or np.any(np.diff(t) <= 0)):
        raise ValueError("record times are not increasing minute offsets within the day")
    return rec


def _ts_us(day: date, t: np.ndarray) -> np.ndarray:
    return (day - _EPOCH).days * _US_PER_DAY + t.astype(np.int64) * 1_000_000


def decode_bi5(raw: bytes, day: date, scale: float) -> pl.DataFrame:
    """One side's minute candles: ``ts`` (UTC minute open), open, high, low, close (/scale), volume."""
    rec = _records(raw)
    return pl.DataFrame(
        {
            "ts": _ts_us(day, rec["t"]),
            "open": rec["o"] / scale,
            "high": rec["h"] / scale,
            "low": rec["l"] / scale,
            "close": rec["c"] / scale,
            "volume": rec["v"].astype(np.float64),
        },
        schema={k: (pl.Int64 if k == "ts" else v) for k, v in CANDLE_SCHEMA.items()},
    ).with_columns(pl.col("ts").cast(_TS))


# ---------------------------------------------------------------------------------------------- bronze


def _merge(ts: np.ndarray, bid: np.ndarray, ask: np.ndarray, scale: float) -> pl.DataFrame:
    """Vectorised BID+ASK -> bronze rows (inputs already aligned minute by minute)."""

    def f(side: np.ndarray, k: str) -> np.ndarray:
        return side[k].astype(np.float64)

    mid = {k: (f(bid, k) + f(ask, k)) / (2.0 * scale) for k in ("o", "h", "l", "c")}
    bid_v, ask_v = f(bid, "v"), f(ask, "v")
    return (
        pl.DataFrame(
            {
                "ts": ts,
                "open": mid["o"],
                "high": mid["h"],
                "low": mid["l"],
                "close": mid["c"],
                "volume": bid_v + ask_v,
                "is_real": (bid_v > 0) | (ask_v > 0),
                "bid_close": f(bid, "c") / scale,
                "ask_close": f(ask, "c") / scale,
                "spread": (f(ask, "c") - f(bid, "c")) / scale,
            }
        )
        .with_columns(pl.col("ts").cast(_TS), pl.lit(None, dtype=pl.Float64).alias("n_trades"))
        .select(list(BRONZE_SCHEMA))
    )


def _flag(asset: str, day: date, flag: str, detail: str) -> dict:
    return {"asset": asset, "date": day, "flag": flag, "detail": detail}


def _drop_partition(path: Path) -> None:
    """Remove a stale bronze year partition (and its directory when it is left empty)."""
    if not path.exists():
        return
    log.warning("removing stale bronze partition %s (no rows from the current raw files)", path)
    path.unlink()
    try:
        path.parent.rmdir()
    except OSError:  # directory not empty
        pass


def _drop_stale_years(asset: str, out_dir: Path, keep: set[int]) -> None:
    """Remove ``year=*`` partitions of ``asset`` whose year has no raw file any more."""
    for p in (out_dir / f"asset={asset}").glob("year=*/part.parquet"):
        y = p.parent.name.split("=", 1)[1]
        if y.isdigit() and int(y) not in keep:
            _drop_partition(p)


def _build_year(
    asset: str, instr: str, scale: float, year: int, days: list[date], raw_dir: Path, out_dir: Path
) -> tuple[pl.DataFrame, list[dict], Counter]:
    """Decode, merge and write one year. Returns per-day first/last real mids, flags and day counts.

    Day outcomes: ``empty`` (both sides on disk with no record, not flagged), ``missing_side`` (a side absent,
    a side without records, or no common minute: no rows), ``decode_error`` (no rows), otherwise ``merged`` on
    the common minutes (flagged ``misaligned`` when BID and ASK minutes differ). A year without rows has its
    partition removed, so bronze never keeps rows the current raw files no longer produce.
    """
    ts_parts, bid_parts, ask_parts = [], [], []
    flags: list[dict] = []
    n = Counter()
    for day in days:
        paths = {s: raw_path(instr, day, s, raw_dir) for s in SIDES}
        try:
            rec = {s: _records(p.read_bytes()) if p.exists() else None for s, p in paths.items()}
        except (lzma.LZMAError, ValueError, EOFError) as exc:
            n["decode_error"] += 1
            flags.append(_flag(asset, day, "decode_error", str(exc)))
            log.warning("%s %s: undecodable file (%s); day skipped", asset, day, exc)
            continue
        sizes = {s: (-1 if r is None else r.size) for s, r in rec.items()}
        if all(v == 0 for v in sizes.values()):
            n["empty"] += 1
            continue
        if any(v <= 0 for v in sizes.values()):
            n["missing_side"] += 1
            detail = ", ".join(f"{s}: {'no file' if v < 0 else f'{v} rows'}" for s, v in sizes.items())
            flags.append(_flag(asset, day, "missing_side", detail))
            log.info("%s %s: one side missing (%s); day yields no rows", asset, day, detail)
            continue
        bid, ask = rec["BID"], rec["ASK"]
        t, ib, ia = np.intersect1d(bid["t"], ask["t"], assume_unique=True, return_indices=True)
        if t.size != bid.size or t.size != ask.size:
            detail = f"BID {bid.size} / ASK {ask.size} / common {t.size}"
            if t.size == 0:  # no minute has both sides: the one-side-missing rule applies
                n["missing_side"] += 1
                flags.append(_flag(asset, day, "missing_side", detail))
                log.warning("%s %s: BID and ASK share no minute (%s); day yields no rows", asset, day, detail)
                continue
            n["misaligned"] += 1
            flags.append(_flag(asset, day, "misaligned", detail))
            log.warning("%s %s: misaligned minutes (%s); common minutes kept", asset, day, detail)
        ts_parts.append(_ts_us(day, t))
        bid_parts.append(bid[ib])
        ask_parts.append(ask[ia])
        n["merged"] += 1

    if not ts_parts:
        _drop_partition(bronze_path(asset, year, out_dir))
        return pl.DataFrame(schema={"date": pl.Date, "first_mid": pl.Float64, "last_mid": pl.Float64}), flags, n
    df = _merge(np.concatenate(ts_parts), np.concatenate(bid_parts), np.concatenate(ask_parts), scale)
    df = df.sort("ts").unique(subset="ts", keep="last", maintain_order=True)
    write_parquet(df, bronze_path(asset, year, out_dir))
    n["rows"] += df.height
    n["real_rows"] += int(df["is_real"].sum())
    ends = (
        df.filter(pl.col("is_real"))
        .group_by(pl.col("ts").dt.date().alias("date"))
        .agg(first_mid=pl.col("open").sort_by("ts").first(), last_mid=pl.col("close").sort_by("ts").last())
    )
    return ends, flags, n


def _continuity(asset: str, ends: pl.DataFrame, tol: float = CONTINUITY_TOL) -> pl.DataFrame:
    """Flag days whose first real mid moves more than ``tol`` from the previous real day's last mid."""
    return (
        ends.sort("date")
        .with_columns(prev_date=pl.col("date").shift(1), prev_mid=pl.col("last_mid").shift(1))
        .with_columns(rel_change=pl.col("first_mid") / pl.col("prev_mid") - 1.0)
        .filter(pl.col("rel_change").abs() > tol)
        .with_columns(
            asset=pl.lit(asset),
            flag=pl.lit("continuity"),
            # concat_str, not pl.format: polars 1.44 pl.format panics on Float64 input in small multi-row frames
            detail=pl.concat_str(
                pl.lit("first real mid "),
                (pl.col("rel_change") * 100).round(2).cast(pl.Utf8),
                pl.lit("% from previous real day"),
            ),
        )
        .select(list(FLAG_SCHEMA))
    )


def _raw_days(instr: str, raw_dir: Path) -> dict[int, list[date]]:
    """Days with at least one side on disk, grouped by year."""
    days: set[date] = set()
    for p in (raw_dir / instr).glob("*/*.bi5"):
        m = _FILE_RE.match(p.name)
        if m:
            days.add(date.fromisoformat(m.group(1)))
    by_year: dict[int, list[date]] = {}
    for d in sorted(days):
        by_year.setdefault(d.year, []).append(d)
    return by_year


def build_bronze(
    asset: str,
    raw_dir: Path | None = None,
    out_dir: Path | None = None,
    flags_dir: Path | None = None,
    workers: int | None = None,
) -> dict:
    """Merge BID+ASK day files into bronze minute rows (one parquet per year) and write the flags file.

    The asset's bronze partitions and flags file are rebuilt from the raw files found under ``raw_dir``: years
    without any resulting row lose their partition. Years are processed in parallel threads (LZMA, NumPy and
    Polars release the GIL). Returns a summary dict.
    """
    instr, scale = _instrument(asset)
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DIR
    out_dir = Path(out_dir) if out_dir is not None else BRONZE_DIR
    by_year = _raw_days(instr, raw_dir)
    if not by_year:
        log.warning("%s: no raw Dukascopy files under %s", asset, raw_dir / instr)
    _drop_stale_years(asset, out_dir, set(by_year))
    workers = workers or min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(by_year)))) as ex:
        results = list(
            ex.map(lambda y: _build_year(asset, instr, scale, y, by_year[y], raw_dir, out_dir), sorted(by_year))
        )

    counts: Counter = Counter()
    flag_rows: list[dict] = []
    ends = []
    for e, fl, n in results:
        ends.append(e)
        flag_rows.extend(fl)
        counts.update(n)
    cont = _continuity(asset, pl.concat(ends)) if ends else pl.DataFrame(schema=FLAG_SCHEMA)
    for r in cont.iter_rows(named=True):
        log.warning("%s %s: continuity flag, %s", asset, r["date"], r["detail"])
    other = pl.DataFrame(flag_rows, schema={k: FLAG_SCHEMA[k] for k in ("asset", "date", "flag", "detail")})
    flags = pl.concat([cont, other], how="diagonal_relaxed").select(
        [pl.col(k).cast(v) for k, v in FLAG_SCHEMA.items()]
    ).sort(["date", "flag"])
    fpath = flags_path(asset, flags_dir)
    write_parquet(flags, fpath)

    rows, real = counts["rows"], counts["real_rows"]
    summary = {
        "asset": asset,
        "instrument": instr,
        "years": sorted(by_year),
        "days": sum(len(v) for v in by_year.values()),
        "days_merged": counts["merged"],
        "days_empty": counts["empty"],
        "missing_side": counts["missing_side"],
        "decode_errors": counts["decode_error"],
        "misaligned_days": counts["misaligned"],
        "rows": rows,
        "real_rows": real,
        "real_share": real / rows if rows else float("nan"),
        "continuity_flags": cont.height,
        "flags_path": str(fpath),
    }
    log.info("dukascopy bronze %s: %s", asset, summary)
    return summary
