"""Implied-volatility and reference-series ingestion (SPEC §2.4; used by the IV benchmarks of §7).

Snapshots are immutable files in ``data/raw/implied/`` named ``{prefix}_{YYYY-MM-DD}.{ext}`` with the UTC
download date; the newest snapshot of each series is used:

- ``dvol_{BTC|ETH}_{date}.json`` — Deribit DVOL daily candles ``[ts_ms, open, high, low, close]``
- ``vix_{date}.csv`` — CBOE ``VIX_History.csv``
- ``fred_{EVZCLS|SP500}_{date}.csv`` — FRED graph CSV (``SP500`` only feeds the DQ report)

``build_implied`` writes ``data/gold/implied.parquet``: ``asset, origin (Date), iv (vol points),
iv_var_30d (%² = iv²·30/365), source``. ``origin`` is always a scheduled session of the asset (§4.1):
CBOE has printed VIX on US exchange holidays since 2022-05-30 (Global Trading Hours), and those rows are
dropped, so lagging the table by one row always gives the previous session's value.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Callable

import polars as pl
import requests

from volrisk import config as C
from volrisk import io, sessions
from volrisk.data.http import USER_AGENT

log = logging.getLogger(__name__)

RAW_IMPLIED = C.RAW / "implied"

DVOL_URL = "https://www.deribit.com/api/v2/public/get_volatility_index_data"
VIX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv"

DVOL_CURRENCIES = ("BTC", "ETH")
DVOL_START = date(2021, 3, 1)
DVOL_CHUNK_DAYS = 500
FRED_SERIES = ("EVZCLS", "SP500")
FRED_TIMEOUT = 120.0  # FRED is slow; SPEC requires >= 90 s
FRED_RETRIES = 6

DAY_MS = 86_400_000
IV_DAYS = 30  # DVOL, VIX and EVZ are all 30-calendar-day implied vols, annualised on 365 days

IMPLIED_SCHEMA = {
    "asset": pl.Utf8,
    "origin": pl.Date,
    "iv": pl.Float64,
    "iv_var_30d": pl.Float64,
    "source": pl.Utf8,
}

# DVOL fetcher: (url, query params) -> decoded JSON response
Fetch = Callable[[str, dict], dict]

_NAME_DATE = re.compile(r"_(\d{4}-\d{2}-\d{2})\.[A-Za-z]+$")


# ------------------------------------------------------------------------------------------------ helpers


def _raw_dir(raw_dir: Path | str | None) -> Path:
    return Path(raw_dir) if raw_dir is not None else RAW_IMPLIED


def _utcnow(now: datetime | None) -> datetime:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("`now` must be timezone-aware (UTC)")
    return now.astimezone(timezone.utc)


def _ms(d: date) -> int:
    """Epoch milliseconds of ``d`` at 00:00 UTC."""
    return int(datetime.combine(d, dtime(0), tzinfo=timezone.utc).timestamp() * 1000)


def _snapshot_path(raw_dir: Path, prefix: str, day: date, ext: str) -> Path:
    return raw_dir / f"{prefix}_{day.isoformat()}.{ext}"


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _http_get(
    url: str, params: dict | None = None, timeout: float = 60.0, retries: int = 5
) -> requests.Response | None:
    """GET with exponential backoff + jitter (redirects followed). ``None`` once all attempts failed.

    Retries timeouts, connection errors, 429 and 5xx; any other status is returned to the caller.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(
                url, params=params, timeout=timeout, headers={"User-Agent": USER_AGENT}, allow_redirects=True
            )
            if resp.status_code != 429 and resp.status_code < 500:
                return resp
            log.debug("attempt %d for %s: HTTP %d", attempt, url, resp.status_code)
        except requests.RequestException as exc:
            log.debug("attempt %d for %s failed: %s", attempt, url, exc)
        if attempt < retries:
            time.sleep(min(60.0, (2**attempt) * 0.5 + random.uniform(0, 0.5)))
    return None


def _default_fetch(url: str, params: dict) -> dict:
    resp = _http_get(url, params=params, timeout=60.0, retries=5)
    if resp is None:
        raise RuntimeError(f"Deribit unreachable: {url} {params}")
    resp.raise_for_status()
    return resp.json()


def _download_csv(url: str, params: dict | None, header_token: str, timeout: float, retries: int) -> bytes | None:
    """Body of a CSV download whose header line contains ``header_token``; None (logged) otherwise.

    Validated before anything is written, so a failed re-download never replaces a good snapshot.
    """
    resp = _http_get(url, params=params, timeout=timeout, retries=retries)
    if resp is None:
        log.warning("%s unreachable after %d attempts", url, retries + 1)
        return None
    body = resp.content
    if resp.status_code != 200 or not body:
        log.warning("%s returned HTTP %d with %d bytes", url, resp.status_code, len(body))
        return None
    header = body.split(b"\n", 1)[0].decode("utf-8-sig", errors="replace").upper()
    if header_token not in header:  # e.g. an HTML error page served with HTTP 200
        log.warning("%s: unexpected body (header %r)", url, header[:80])
        return None
    return body


def evz_last_date() -> date:
    """Last EVZ date used (``config dates.evz_last``; EVZ is stale afterwards and discontinued)."""
    d = C.load()["dates"]["evz_last"]
    return d if isinstance(d, date) else date.fromisoformat(str(d))


# ------------------------------------------------------------------------------------------------ download


def download_dvol(
    currency: str,
    raw_dir: Path | str | None = None,
    *,
    start: date = DVOL_START,
    chunk_days: int = DVOL_CHUNK_DAYS,
    now: datetime | None = None,
    fetch: Fetch | None = None,
    force: bool = False,
) -> Path:
    """Snapshot Deribit DVOL daily candles for ``currency`` into ``dvol_{CUR}_{YYYY-MM-DD}.json``.

    Requests ``[start, today 00:00 UTC)`` in ``chunk_days`` windows (never the current UTC day) and follows
    ``result.continuation`` (the next ``end_timestamp``) until it is null. The JSON stores the concatenated
    rows plus ``downloaded_at``. An existing same-day snapshot is reused unless ``force``.
    """
    currency = currency.upper()
    if currency not in DVOL_CURRENCIES:
        raise ValueError(f"DVOL currency must be one of {DVOL_CURRENCIES}, got {currency!r}")
    now = _utcnow(now)
    out = _snapshot_path(_raw_dir(raw_dir), f"dvol_{currency}", now.date(), "json")
    if out.exists() and out.stat().st_size and not force:
        log.info("DVOL %s snapshot for %s exists: %s", currency, now.date(), out)
        return out
    fetch = fetch or _default_fetch

    last_ms = _ms(now.date()) - 1  # never request the current (live) UTC day
    rows: list[list] = []
    n_requests = 0
    chunk_start = _ms(start)
    while chunk_start <= last_ms:
        chunk_end = min(chunk_start + chunk_days * DAY_MS - 1, last_ms)
        end = chunk_end
        while True:
            params = {
                "currency": currency,
                "start_timestamp": chunk_start,
                "end_timestamp": end,
                "resolution": "1D",
            }
            payload = fetch(DVOL_URL, params)
            n_requests += 1
            if "error" in payload:
                raise RuntimeError(f"Deribit error for {params}: {payload['error']}")
            result = payload["result"]
            rows.extend(result.get("data") or [])
            cont = result.get("continuation")
            if cont is None or cont < chunk_start:
                break
            if cont >= end:  # a continuation must move backwards; anything else would loop forever
                raise RuntimeError(f"Deribit continuation {cont} does not precede end_timestamp {end}")
            end = int(cont)
        chunk_start = chunk_end + 1

    if not rows:
        raise RuntimeError(f"Deribit returned no DVOL rows for {currency}")
    rows.sort(key=lambda r: r[0])
    snapshot = {
        "source": "deribit",
        "currency": currency,
        "resolution": "1D",
        "downloaded_at": now.isoformat(),
        "start_timestamp": _ms(start),
        "end_timestamp": last_ms,
        "n_requests": n_requests,
        "columns": ["ts_ms", "open", "high", "low", "close"],
        "data": rows,
    }
    _write_atomic(out, json.dumps(snapshot).encode("utf-8"))
    log.info("DVOL %s: %d rows in %d requests -> %s", currency, len(rows), n_requests, out)
    return out


def download_vix(raw_dir: Path | str | None = None, *, now: datetime | None = None, force: bool = False) -> Path:
    """Snapshot CBOE ``VIX_History.csv`` into ``vix_{YYYY-MM-DD}.csv`` (raises if unreachable)."""
    now = _utcnow(now)
    out = _snapshot_path(_raw_dir(raw_dir), "vix", now.date(), "csv")
    if out.exists() and out.stat().st_size and not force:
        log.info("VIX snapshot for %s exists: %s", now.date(), out)
        return out
    body = _download_csv(VIX_URL, None, "CLOSE", timeout=60.0, retries=5)
    if body is None:
        raise RuntimeError(f"VIX download failed: {VIX_URL}")
    _write_atomic(out, body)
    return out


def download_fred(
    series_id: str, raw_dir: Path | str | None = None, *, now: datetime | None = None, force: bool = False
) -> Path | None:
    """Snapshot a FRED series into ``fred_{ID}_{YYYY-MM-DD}.csv``; ``None`` (with a warning) if unreachable."""
    series_id = series_id.upper()
    if series_id not in FRED_SERIES:
        raise ValueError(f"FRED series must be one of {FRED_SERIES}, got {series_id!r}")
    now = _utcnow(now)
    out = _snapshot_path(_raw_dir(raw_dir), f"fred_{series_id}", now.date(), "csv")
    if out.exists() and out.stat().st_size and not force:
        log.info("FRED %s snapshot for %s exists: %s", series_id, now.date(), out)
        return out
    body = _download_csv(FRED_URL, {"id": series_id}, series_id, timeout=FRED_TIMEOUT, retries=FRED_RETRIES)
    if body is None:
        log.warning("FRED %s unavailable; continuing without it", series_id)
        return None
    _write_atomic(out, body)
    return out


def download_all(raw_dir: Path | str | None = None, *, force: bool = False) -> dict[str, Path | None]:
    """All implied/reference snapshots. Keys: DVOL_BTC, DVOL_ETH, VIX, EVZCLS, SP500 (FRED may be None)."""
    paths: dict[str, Path | None] = {}
    for cur in DVOL_CURRENCIES:
        paths[f"DVOL_{cur}"] = download_dvol(cur, raw_dir, force=force)
    paths["VIX"] = download_vix(raw_dir, force=force)
    for sid in FRED_SERIES:
        paths[sid] = download_fred(sid, raw_dir, force=force)
    return paths


# ------------------------------------------------------------------------------------------------ parse


def latest_snapshot(prefix: str, raw_dir: Path | str | None = None) -> Path | None:
    """Newest non-empty ``{prefix}_{YYYY-MM-DD}.*`` snapshot (by the date in its name), or None."""
    pat = re.compile(rf"^{re.escape(prefix)}_(\d{{4}}-\d{{2}}-\d{{2}})\.(json|csv)$")
    d = _raw_dir(raw_dir)
    if not d.exists():
        return None
    best: tuple[date, Path] | None = None
    for p in d.iterdir():
        m = pat.match(p.name)
        if not m or not p.is_file() or p.stat().st_size == 0:
            continue
        day = date.fromisoformat(m.group(1))
        if best is None or day > best[0]:
            best = (day, p)
    return best[1] if best else None


def _snapshot_time(path: Path, payload: dict, asof: datetime | None) -> datetime:
    """Download time used to drop the live candle: ``asof`` > JSON ``downloaded_at`` > file-name date at
    00:00 UTC (conservative) > Deribit ``usIn`` (raw API response)."""
    if asof is not None:
        return _utcnow(asof)
    if payload.get("downloaded_at"):
        return _utcnow(datetime.fromisoformat(payload["downloaded_at"]))
    m = _NAME_DATE.search(path.name)
    if m:
        return datetime.combine(date.fromisoformat(m.group(1)), dtime(0), tzinfo=timezone.utc)
    if payload.get("usIn"):
        return datetime.fromtimestamp(int(payload["usIn"]) / 1e6, tz=timezone.utc)
    raise ValueError(f"cannot determine the download time of {path}; pass asof=")


def parse_dvol(path: Path | str, asof: datetime | None = None) -> pl.DataFrame:
    """DVOL snapshot -> ``[origin: Date, iv: Float64]``; ``iv`` at origin D = close of the candle stamped D.

    Accepts both the snapshot written by :func:`download_dvol` and a raw Deribit response. Candles whose
    start + 1 day is after the download time are still live and dropped.
    """
    path = Path(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["data"] if "data" in payload else payload["result"]["data"]
    cutoff_ms = int(_snapshot_time(path, payload, asof).timestamp() * 1000)
    df = pl.DataFrame(
        {
            "ts": pl.Series([r[0] for r in rows], dtype=pl.Int64),
            "close": pl.Series([None if r[4] is None else float(r[4]) for r in rows], dtype=pl.Float64),
        }
    )
    misaligned = df.filter(pl.col("ts") % DAY_MS != 0)
    if misaligned.height:
        raise ValueError(f"{path}: {misaligned.height} DVOL candles do not start at 00:00 UTC")
    n_live = df.filter(pl.col("ts") + DAY_MS > cutoff_ms).height
    if n_live:
        log.info("%s: dropping %d live candle(s)", path.name, n_live)
    return (
        df.filter((pl.col("ts") + DAY_MS <= cutoff_ms) & pl.col("close").is_not_null())
        .sort("ts", maintain_order=True)
        .unique(subset="ts", keep="last", maintain_order=True)
        .select(
            pl.col("ts").cast(pl.Datetime("ms", "UTC")).dt.date().alias("origin"),
            pl.col("close").alias("iv"),
        )
    )


def parse_vix(path: Path | str) -> pl.DataFrame:
    """CBOE VIX history -> ``[origin: Date, iv: Float64]`` (``DATE`` as ``%m/%d/%Y``, ``CLOSE``)."""
    df = pl.read_csv(path, infer_schema=False)
    df = df.rename({c: c.strip().upper() for c in df.columns})
    return (
        df.select(
            pl.col("DATE").str.strip_chars().str.strptime(pl.Date, "%m/%d/%Y").alias("origin"),
            pl.col("CLOSE").str.strip_chars().cast(pl.Float64, strict=False).alias("iv"),
        )
        .drop_nulls()
        .sort("origin", maintain_order=True)
        .unique(subset="origin", keep="last", maintain_order=True)
    )


def parse_fred(path: Path | str, series_id: str) -> pl.DataFrame:
    """FRED graph CSV -> ``[date: Date, value: Float64]`` without missing (``.``/empty) observations.

    For ``EVZCLS`` a value equal to the previous day's (the previous row of the file) is stale and dropped;
    a missing previous day breaks the comparison, so the first value after a ``.`` is kept. Only dates
    ``<= config dates.evz_last`` are kept.
    """
    series_id = series_id.upper()
    df = pl.read_csv(path, infer_schema=False)
    date_col = df.columns[0]  # 'observation_date' (current FRED) or 'DATE' (older files)
    val_col = series_id if series_id in df.columns else df.columns[1]
    value = pl.col(val_col).str.strip_chars()
    out = (
        df.select(
            pl.col(date_col).str.strip_chars().str.strptime(pl.Date, "%Y-%m-%d").alias("date"),
            pl.when(value.is_in(["", "."])).then(None).otherwise(value).cast(pl.Float64).alias("value"),
        )
        .sort("date", maintain_order=True)
        .unique(subset="date", keep="last", maintain_order=True)
    )
    if series_id != "EVZCLS":
        return out.drop_nulls("value")
    evz_last = evz_last_date()
    n0 = out["value"].is_not_null().sum()
    # shift on the unfiltered series: null == x is null -> not stale, so a '.' day breaks a run
    stale = (pl.col("value") == pl.col("value").shift(1)).fill_null(False)
    out = out.filter(~stale & (pl.col("date") <= evz_last)).drop_nulls("value")
    log.info("EVZ: kept %d of %d observations (stale repeats dropped, <= %s)", out.height, n0, evz_last)
    return out


# ------------------------------------------------------------------------------------------------ gold


def _session_rows(df: pl.DataFrame, asset_name: str) -> pl.DataFrame:
    """Rows of ``[origin, ...]`` whose ``origin`` is a scheduled session of ``asset_name`` (SPEC §4.1).

    Sessions come from the calendar, never from the data: e.g. VIX rows on XNYS holidays are dropped.
    """
    if df.is_empty():
        return df
    sched = sessions.session_schedule(asset_name, df["origin"].min(), df["origin"].max())
    on_session = pl.col("origin").is_in(sched["session_date"].implode())
    dropped = df.filter(~on_session)["origin"]
    if dropped.len():
        log.info(
            "%s: dropping %d IV rows on non-session dates (%s .. %s), e.g. %s",
            asset_name,
            dropped.len(),
            dropped.min(),
            dropped.max(),
            ", ".join(str(d) for d in dropped.head(5)),
        )
    return df.filter(on_session)


def build_implied(raw_dir: Path | str | None = None, out_path: Path | str | None = None) -> pl.DataFrame:
    """Gold implied-vol table from the newest snapshots; writes ``out_path`` (default ``io.IMPLIED``).

    BTC/ETH <- DVOL, SPX <- VIX are required; EURUSD <- EVZ may be missing (warning, no rows). Only rows
    whose ``origin`` is a session of the asset (§4.1) and ``<= config dates.data_end`` are kept.
    """
    raw = _raw_dir(raw_dir)
    out_path = Path(out_path) if out_path is not None else io.IMPLIED
    parts: list[pl.DataFrame] = []
    for asset_name in C.ASSETS:
        source = C.asset(asset_name).iv
        if source == "DVOL":
            snap = latest_snapshot(f"dvol_{asset_name}", raw)
            df = parse_dvol(snap) if snap else None
        elif source == "VIX":
            snap = latest_snapshot("vix", raw)
            df = parse_vix(snap) if snap else None
        elif source == "EVZ":
            snap = latest_snapshot("fred_EVZCLS", raw)
            df = parse_fred(snap, "EVZCLS").rename({"date": "origin", "value": "iv"}) if snap else None
        else:
            raise ValueError(f"unknown IV source {source!r} for {asset_name}")
        if df is None:
            if source == "EVZ":
                log.warning("no EVZ snapshot in %s: EURUSD IV benchmark unavailable", raw)
                continue
            raise FileNotFoundError(f"no {source} snapshot for {asset_name} in {raw}; run download_all()")
        log.info("%s <- %s (%s): %d rows", asset_name, source, snap.name, df.height)
        df = _session_rows(df.filter(pl.col("origin") <= C.data_end()), asset_name)
        parts.append(df.with_columns(pl.lit(asset_name).alias("asset"), pl.lit(source).alias("source")))

    out = pl.concat(parts, how="vertical")
    n_bad = out.filter(~(pl.col("iv") > 0)).height
    if n_bad:
        log.warning("implied: dropping %d non-positive IV values", n_bad)
    out = (
        out.filter(pl.col("iv") > 0)
        .with_columns((pl.col("iv") ** 2 * IV_DAYS / 365.0).alias("iv_var_30d"))
        .select(list(IMPLIED_SCHEMA))
        .cast(IMPLIED_SCHEMA)
        .sort(["asset", "origin"])
    )
    io.write_parquet(out, out_path)
    log.info("implied: %d rows -> %s", out.height, out_path)
    return out
