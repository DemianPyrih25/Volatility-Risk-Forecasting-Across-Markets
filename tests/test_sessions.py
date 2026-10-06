"""Session calendars (SPEC §4.1): DST, holidays, half-days, FX week boundaries, session_of consistency."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from volrisk.sessions import SCHEDULE_SCHEMA, session_of, session_schedule

UTC = timezone.utc
NY = ZoneInfo("America/New_York")


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def rows(asset: str, start: date, end: date) -> dict[date, tuple[datetime, datetime, int]]:
    s = session_schedule(asset, start, end)
    return {r["session_date"]: (r["open_utc"], r["close_utc"], r["n_sched"]) for r in s.iter_rows(named=True)}


@pytest.mark.parametrize("asset", ["BTC", "ETH", "EURUSD", "SPX"])
def test_schema_sorted_unique(asset):
    s = session_schedule(asset, date(2024, 1, 1), date(2024, 12, 31))
    assert dict(s.schema) == SCHEDULE_SCHEMA
    assert s["session_date"].is_sorted() and s["session_date"].n_unique() == s.height
    assert (s["close_utc"] > s["open_utc"]).all()
    assert (s["open_utc"].shift(-1).drop_nulls() >= s["close_utc"].head(s.height - 1)).all()  # no overlap
    assert s["session_date"].min() >= date(2024, 1, 1) and s["session_date"].max() <= date(2024, 12, 31)
    empty = session_schedule(asset, date(2024, 1, 2), date(2024, 1, 1))
    assert empty.height == 0 and dict(empty.schema) == SCHEDULE_SCHEMA


# ------------------------------------------------------------------------------------------------- crypto
def test_crypto_every_utc_day_288():
    r = rows("BTC", date(2024, 2, 27), date(2024, 3, 2))
    assert list(r) == [date(2024, 2, 27) + timedelta(days=k) for k in range(5)]  # incl. leap day
    for d, (o, c, n) in r.items():
        assert o == utc(d.year, d.month, d.day) and c == o + timedelta(days=1) and n == 288


def test_crypto_full_sample_288():
    s = session_schedule("ETH", date(2018, 1, 1), date(2026, 9, 30))
    assert s.height == (date(2026, 9, 30) - date(2018, 1, 1)).days + 1
    assert (s["n_sched"] == 288).all()


# ----------------------------------------------------------------------------------------------------- FX
def test_fx_dst_spring_2026():
    # US DST starts Sunday 2026-03-08: 17:00 NY is 22:00 UTC before, 21:00 UTC after.
    r = rows("EURUSD", date(2026, 3, 5), date(2026, 3, 10))
    assert list(r) == [date(2026, 3, 5), date(2026, 3, 6), date(2026, 3, 9), date(2026, 3, 10)]
    assert r[date(2026, 3, 6)][:2] == (utc(2026, 3, 5, 22), utc(2026, 3, 6, 22))
    assert r[date(2026, 3, 9)][:2] == (utc(2026, 3, 8, 21), utc(2026, 3, 9, 21))  # opens Sunday
    assert r[date(2026, 3, 10)][:2] == (utc(2026, 3, 9, 21), utc(2026, 3, 10, 21))


def test_fx_dst_autumn_2026():
    # US DST ends Sunday 2026-11-01.
    r = rows("EURUSD", date(2026, 10, 29), date(2026, 11, 3))
    assert list(r) == [date(2026, 10, 29), date(2026, 10, 30), date(2026, 11, 2), date(2026, 11, 3)]
    assert r[date(2026, 10, 30)][:2] == (utc(2026, 10, 29, 21), utc(2026, 10, 30, 21))
    assert r[date(2026, 11, 2)][:2] == (utc(2026, 11, 1, 22), utc(2026, 11, 2, 22))
    assert r[date(2026, 11, 3)][:2] == (utc(2026, 11, 2, 22), utc(2026, 11, 3, 22))


def test_fx_monday_opens_sunday_22_winter_21_summer():
    r = rows("EURUSD", date(2024, 1, 1), date(2024, 12, 31))
    assert r[date(2024, 1, 8)][0] == utc(2024, 1, 7, 22)
    assert r[date(2024, 7, 8)][0] == utc(2024, 7, 7, 21)
    mondays = [d for d in r if d.weekday() == 0]
    for d in mondays:
        o = r[d][0].astimezone(NY)
        assert (o.date(), o.hour, o.minute) == (d - timedelta(days=1), 17, 0)
        assert o.date().weekday() == 6  # Sunday


def test_fx_weekends_and_holidays_absent():
    r = rows("EURUSD", date(2012, 1, 1), date(2026, 9, 30))
    assert all(d.weekday() < 5 for d in r)
    for y in range(2012, 2027):
        assert date(y, 1, 1) not in r
        if y < 2026:
            assert date(y, 12, 25) not in r
    assert date(2024, 12, 24) in r and date(2024, 12, 26) in r and date(2025, 1, 2) in r
    assert r[date(2024, 12, 26)][0] == utc(2024, 12, 25, 22)  # opens on Christmas evening NY
    assert all(n == 288 for _, _, n in r.values())
    # Every session is exactly 17:00 NY (t-1) -> 17:00 NY (t).
    for d, (o, c, _) in r.items():
        prev = d - timedelta(days=1)
        assert o.astimezone(NY).replace(tzinfo=None) == datetime(prev.year, prev.month, prev.day, 17)
        assert c.astimezone(NY).replace(tzinfo=None) == datetime(d.year, d.month, d.day, 17)


# ---------------------------------------------------------------------------------------------------- SPX
def test_spx_dst_2026():
    r = rows("SPX", date(2026, 3, 5), date(2026, 3, 10))
    assert r[date(2026, 3, 6)] == (utc(2026, 3, 6, 14, 30), utc(2026, 3, 6, 21), 78)
    assert r[date(2026, 3, 9)] == (utc(2026, 3, 9, 13, 30), utc(2026, 3, 9, 20), 78)
    r = rows("SPX", date(2026, 10, 29), date(2026, 11, 3))
    assert r[date(2026, 10, 30)] == (utc(2026, 10, 30, 13, 30), utc(2026, 10, 30, 20), 78)
    assert r[date(2026, 11, 2)] == (utc(2026, 11, 2, 14, 30), utc(2026, 11, 2, 21), 78)
    assert date(2026, 3, 7) not in r and date(2026, 11, 1) not in r


def test_spx_thanksgiving_and_half_day():
    r = rows("SPX", date(2025, 11, 24), date(2025, 12, 1))
    assert date(2025, 11, 27) not in r  # Thanksgiving
    assert r[date(2025, 11, 28)] == (utc(2025, 11, 28, 14, 30), utc(2025, 11, 28, 18), 42)
    assert r[date(2025, 11, 26)][2] == 78


def test_spx_labor_day_2026_absent():
    r = rows("SPX", date(2026, 9, 1), date(2026, 9, 10))
    assert date(2026, 9, 7) not in r
    assert date(2026, 9, 4) in r and date(2026, 9, 8) in r


def test_spx_n_sched_only_78_or_42():
    s = session_schedule("SPX", date(2012, 1, 1), date(2026, 9, 30))
    assert set(s["n_sched"].unique().to_list()) == {78, 42}
    assert s.filter(pl.col("session_date") == date(2025, 12, 24))["n_sched"].item() == 42
    # 252 ± a few sessions per full year
    per_year = s.group_by(pl.col("session_date").dt.year()).len().filter(pl.col("session_date") < 2026)
    assert per_year["len"].min() >= 248 and per_year["len"].max() <= 253


def test_spx_sessions_from_calendar_not_weekdays():
    s = session_schedule("SPX", date(2012, 10, 29), date(2012, 10, 31))  # Hurricane Sandy closure
    assert s.height == 1 and s["session_date"].item() == date(2012, 10, 31)


# ----------------------------------------------------------------------------------------------- session_of
def test_session_of_crypto():
    out = session_of("BTC", [utc(2024, 1, 1, 23, 59), utc(2024, 1, 2, 0, 0), datetime(2024, 1, 2, 0, 1)])
    assert out.dtype == pl.Date and out.name == "session_date"
    assert out.to_list() == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 2)]


def test_session_of_fx():
    ts = [
        utc(2024, 1, 7, 21, 59),  # Sunday before the open -> weekend
        utc(2024, 1, 7, 22, 0),  # first minute of Monday's session (winter)
        utc(2024, 7, 7, 21, 0),  # first minute of Monday's session (summer)
        utc(2024, 7, 7, 20, 59),
        utc(2024, 1, 5, 21, 59),  # last minute of Friday's session
        utc(2024, 1, 5, 22, 0),  # would be Saturday's session
        utc(2024, 12, 24, 22, 0),  # Dec 25 session -> excluded
        utc(2024, 12, 25, 21, 59),
        utc(2024, 12, 25, 22, 0),  # Dec 26 session
    ]
    exp = [None, date(2024, 1, 8), date(2024, 7, 8), None, date(2024, 1, 5), None, None, None, date(2024, 12, 26)]
    assert session_of("EURUSD", ts).to_list() == exp


def test_session_of_spx_rth_only():
    ts = [
        utc(2024, 1, 2, 14, 29),
        utc(2024, 1, 2, 14, 30),
        utc(2024, 1, 2, 20, 59),
        utc(2024, 1, 2, 21, 0),
        utc(2025, 11, 27, 15, 0),  # Thanksgiving: CFD trades, no session
        utc(2025, 11, 28, 17, 59),
        utc(2025, 11, 28, 18, 0),  # after the half-day close
        utc(2026, 3, 9, 13, 30),  # first minute after the DST switch
    ]
    exp = [None, date(2024, 1, 2), date(2024, 1, 2), None, None, date(2025, 11, 28), None, date(2026, 3, 9)]
    assert session_of("SPX", pl.Series(ts)).to_list() == exp


def test_session_of_input_types():
    # Non-UTC zones are converted; naive values are UTC; empty / all-null input is fine.
    s = pl.Series([utc(2024, 1, 2, 14, 30)]).dt.convert_time_zone("America/New_York")
    assert session_of("SPX", s).to_list() == [date(2024, 1, 2)]
    assert session_of("SPX", pl.Series([datetime(2024, 1, 2, 14, 30)]).dt.cast_time_unit("ms")).to_list() == [
        date(2024, 1, 2)
    ]
    assert session_of("BTC", pl.Series([], dtype=pl.Datetime("us", "UTC"))).len() == 0
    assert session_of("EURUSD", []).len() == 0
    assert session_of("BTC", pl.Series([None], dtype=pl.Datetime("us", "UTC"))).to_list() == [None]


@pytest.mark.parametrize(
    "asset,start,end",
    [
        ("BTC", date(2026, 3, 6), date(2026, 3, 10)),
        ("EURUSD", date(2026, 3, 5), date(2026, 3, 11)),
        ("EURUSD", date(2026, 10, 29), date(2026, 11, 4)),
        ("EURUSD", date(2024, 12, 23), date(2025, 1, 3)),
        ("SPX", date(2026, 3, 5), date(2026, 3, 11)),
        ("SPX", date(2025, 11, 25), date(2025, 12, 2)),
    ],
)
def test_session_of_matches_schedule(asset, start, end):
    """Every minute open time maps to the schedule row whose [open_utc, close_utc) contains it."""
    sched = session_schedule(asset, start - timedelta(days=3), end + timedelta(days=3))
    ts = pl.datetime_range(utc(start.year, start.month, start.day), utc(end.year, end.month, end.day), "1m", eager=True)
    got = session_of(asset, ts)
    exp = (
        pl.DataFrame({"ts": ts})
        .join_asof(sched, left_on="ts", right_on="open_utc", strategy="backward")
        .select(pl.when(pl.col("ts") < pl.col("close_utc")).then(pl.col("session_date")))
        .to_series()
    )
    assert got.to_list() == exp.to_list()
    # The minute's close time ts+60s lies in (open_utc, close_utc] of that session (right-closed sessions).
    df = pl.DataFrame({"ts": ts, "session_date": got}).drop_nulls().join(sched, on="session_date")
    tc = pl.col("ts") + pl.duration(minutes=1)
    assert df.select(((tc > pl.col("open_utc")) & (tc <= pl.col("close_utc"))).all()).item()
