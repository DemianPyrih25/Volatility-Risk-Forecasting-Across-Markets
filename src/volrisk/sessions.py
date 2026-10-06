"""Trading-session calendars per asset clock (SPEC §4.1).

A session ``t`` is the half-open interval ``(open_utc, close_utc]`` of minute *close* times, equivalently
``[open_utc, close_utc)`` of minute *open* times (bronze ``ts``):

- crypto: every calendar day, ``[00:00, 24:00)`` UTC;
- fx (EURUSD): Mon–Fri except Dec 25 and Jan 1, ``(17:00 New York on t-1, 17:00 New York on t]``, so
  Monday's session opens on Sunday 17:00 New York (22:00 UTC in winter, 21:00 UTC in summer);
- xnys (SPX): the XNYS regular session from ``exchange_calendars`` (holidays and 13:00 half-days included).
  Sessions are never inferred from data: the CFD also trades on US holidays and outside RTH.

``n_sched = (close_utc - open_utc) / 5 min``: 288 for crypto and fx, 78 (or 42 on half-days) for xnys.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, time, timedelta, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import exchange_calendars as xc
import pandas as pd
import polars as pl

from volrisk import config as C

NY = "America/New_York"
FX_CUT = time(17, 0)  # New York wall-clock time that ends one FX session and starts the next
FX_HOLIDAYS = ((12, 25), (1, 1))  # (month, day) never an FX session date
UTC_US = pl.Datetime("us", "UTC")
SCHEDULE_SCHEMA = {
    "session_date": pl.Date,
    "open_utc": UTC_US,
    "close_utc": UTC_US,
    "n_sched": pl.Int32,
}


def bar_minutes() -> int:
    """Bar length in minutes (config ``sessions.bar_minutes``; 5 by SPEC §4)."""
    return int(C.load()["sessions"]["bar_minutes"])


def session_schedule(asset: str, start: date, end: date) -> pl.DataFrame:
    """Scheduled sessions of ``asset`` with ``start <= session_date <= end`` (SPEC §4.1).

    Columns ``session_date: Date, open_utc, close_utc: Datetime(us, UTC), n_sched: Int32``, sorted by date.
    """
    clk = C.clock(asset)
    if start > end:
        return pl.DataFrame(schema=SCHEDULE_SCHEMA)
    if clk == "crypto":
        days = pl.date_range(start, end, "1d", eager=True)
        df = pl.DataFrame({"session_date": days}).with_columns(
            open_utc=pl.col("session_date").cast(pl.Datetime("us")).dt.replace_time_zone("UTC")
        )
        df = df.with_columns(close_utc=pl.col("open_utc") + pl.duration(days=1))
    elif clk == "fx":
        df = _fx_frame(start, end)
    elif clk == "xnys":
        df = _xnys_frame(start, end)
    else:
        raise ValueError(f"unknown clock {clk!r} for asset {asset!r}")
    bar_us = bar_minutes() * 60_000_000
    length_us = (pl.col("close_utc") - pl.col("open_utc")).dt.total_microseconds()
    df = df.with_columns(n_sched=(length_us // bar_us).cast(pl.Int32), _rem=length_us % bar_us)
    if df.height and (df["_rem"] != 0).any():
        bad = df.filter(pl.col("_rem") != 0)["session_date"].to_list()
        raise ValueError(f"{asset}: session lengths not a multiple of the bar length on {bad[:5]}")
    return df.select(list(SCHEDULE_SCHEMA)).cast(SCHEDULE_SCHEMA).sort("session_date")


def _fx_frame(start: date, end: date) -> pl.DataFrame:
    ny = ZoneInfo(NY)
    days = [start + timedelta(days=k) for k in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 and (d.month, d.day) not in FX_HOLIDAYS]

    def cut(d: date) -> datetime:  # 17:00 New York on calendar day d, in UTC (DST-aware)
        return datetime.combine(d, FX_CUT, tzinfo=ny).astimezone(timezone.utc)

    return pl.DataFrame(
        {
            "session_date": days,
            "open_utc": [cut(d - timedelta(days=1)) for d in days],
            "close_utc": [cut(d) for d in days],
        },
        schema={"session_date": pl.Date, "open_utc": UTC_US, "close_utc": UTC_US},
    )


@lru_cache(maxsize=4)
def _xnys_calendar(first_year: int, last_year: int) -> xc.ExchangeCalendar:
    # Explicit bounds: the default calendar only spans ~20 years back / 1 year ahead of today.
    return xc.get_calendar("XNYS", start=f"{first_year}-01-01", end=f"{last_year}-12-31")


def _xnys_frame(start: date, end: date) -> pl.DataFrame:
    cal = _xnys_calendar(min(start.year, 2000), max(end.year, 2027))
    s = cal.schedule.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    def utc(col: str) -> pl.Series:
        ns = s[col].dt.tz_convert("UTC").dt.tz_localize(None).to_numpy().astype("datetime64[us]")
        return pl.Series(col, ns).dt.replace_time_zone("UTC")

    return pl.DataFrame(
        {
            "session_date": pl.Series(s.index.to_numpy().astype("datetime64[D]")).cast(pl.Date),
            "open_utc": utc("open"),
            "close_utc": utc("close"),
        }
    )


def session_of(asset: str, ts_utc: pl.Series | Sequence[datetime] | datetime) -> pl.Series:
    """Session date containing each instant ``ts_utc`` (null when it lies in no scheduled session).

    An instant belongs to session ``t`` iff ``open_utc <= ts < close_utc``, so passing a bronze minute's open
    time ``ts`` returns the session that holds its close time ``ts + 60s`` in ``(open_utc, close_utc]``.
    Candidate dates follow SPEC §4.1 — crypto: UTC date of ``ts``; fx: date of ``ts`` in New York + 7h;
    xnys: New York date — and are then checked against :func:`session_schedule`, which drops weekends,
    FX Dec 25/Jan 1, exchange holidays and non-RTH instants. Naive datetimes are taken as UTC.
    Returns a ``Date`` series named ``session_date`` aligned with the input.
    """
    s = ts_utc if isinstance(ts_utc, pl.Series) else pl.Series([ts_utc] if isinstance(ts_utc, datetime) else ts_utc)
    df = pl.DataFrame({"ts": to_utc_us(s)})
    clk = C.clock(asset)
    if clk == "crypto":
        cand = pl.col("ts").dt.date()
    elif clk == "fx":
        cand = (pl.col("ts").dt.convert_time_zone(NY) + pl.duration(hours=7)).dt.date()
    elif clk == "xnys":
        cand = pl.col("ts").dt.convert_time_zone(NY).dt.date()
    else:
        raise ValueError(f"unknown clock {clk!r} for asset {asset!r}")
    df = df.with_columns(cand=cand)
    lo, hi = df["cand"].min(), df["cand"].max()
    if lo is None:
        return pl.Series("session_date", [None] * df.height, dtype=pl.Date)
    sched = session_schedule(asset, lo, hi)
    out = df.join(sched, left_on="cand", right_on="session_date", how="left", maintain_order="left")
    inside = (pl.col("ts") >= pl.col("open_utc")) & (pl.col("ts") < pl.col("close_utc"))
    return out.select(session_date=pl.when(inside).then(pl.col("cand")))["session_date"]


def utc_us_expr(col: str, dtype: pl.DataType) -> pl.Expr:
    """Column ``col`` (of ``dtype``) as ``Datetime(us, UTC)``; naive values are UTC, other zones converted."""
    if dtype == pl.Null:
        return pl.col(col).cast(UTC_US)
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"expected a Datetime column {col!r}, got {dtype}")
    e = pl.col(col).dt.cast_time_unit("us")
    return e.dt.replace_time_zone("UTC") if dtype.time_zone is None else e.dt.convert_time_zone("UTC")


def to_utc_us(s: pl.Series) -> pl.Series:
    """Datetime series as ``Datetime(us, UTC)`` (see :func:`utc_us_expr`)."""
    return s.to_frame("x").select(utc_us_expr("x", s.dtype)).to_series().alias(s.name)
