"""5-minute bars (SPEC §4.2): right-closed labelling, real minutes only, p_open/p_close, coverage, files."""

from __future__ import annotations

import bisect
import io
import json
import lzma
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from volrisk import bars as bars_mod
from volrisk import config as C
from volrisk.bars import (
    BARS_SCHEMA,
    SESSIONS_SCHEMA,
    bars_from_minutes,
    bars_path,
    build_bars,
    marker_path,
    sessions_path,
)
from volrisk.sessions import session_schedule

UTC = timezone.utc
FIX = Path(__file__).parent / "fixtures"
M1, M5 = timedelta(minutes=1), timedelta(minutes=5)


def utc(*a: int) -> datetime:
    return datetime(*a, tzinfo=UTC)


def minutes_frame(rows: list[tuple[datetime, float, float, bool]]) -> pl.DataFrame:
    ts, o, c, r = zip(*rows) if rows else ((), (), (), ())
    return pl.DataFrame(
        {"ts": list(ts), "open": list(o), "close": list(c), "is_real": list(r)},
        schema={"ts": pl.Datetime("us", "UTC"), "open": pl.Float64, "close": pl.Float64, "is_real": pl.Boolean},
    )


def bar(bars: pl.DataFrame, ts_end: datetime) -> dict:
    out = bars.filter(pl.col("ts_end") == ts_end)
    assert out.height == 1, ts_end
    return out.row(0, named=True)


def one_session(asset: str, d: date) -> pl.DataFrame:
    return session_schedule(asset, d, d)


# ------------------------------------------------------------------------------------------ labelling rules
def test_right_closed_right_labelled():
    sched = one_session("BTC", date(2024, 1, 2))
    m = minutes_frame([(utc(2024, 1, 2, 10, 4), 1.0, 2.0, True), (utc(2024, 1, 2, 10, 5), 3.0, 4.0, True)])
    bars, _ = bars_from_minutes(m, sched, "BTC")
    b1, b2 = bar(bars, utc(2024, 1, 2, 10, 5)), bar(bars, utc(2024, 1, 2, 10, 10))
    assert (b1["price"], b1["n_real_min"], b1["bar_idx"]) == (2.0, 1, 121)  # opens 10:04, closes 10:05
    assert (b2["price"], b2["n_real_min"], b2["bar_idx"]) == (4.0, 1, 122)  # opens 10:05, closes 10:06
    assert bars.filter(pl.col("price").is_not_null()).height == 2


def test_price_is_close_of_last_real_minute():
    sched = one_session("BTC", date(2024, 1, 2))
    t = utc(2024, 1, 2, 10, 0)
    m = minutes_frame(
        [
            (t + 3 * M1, 30.0, 3.0, True),  # rows deliberately unsorted
            (t, 10.0, 1.0, True),
            (t + M1, 20.0, 2.0, True),
            (t + 4 * M1, 99.0, 99.0, False),  # flat fill after the last real minute
        ]
    )
    bars, _ = bars_from_minutes(m, sched, "BTC")
    b = bar(bars, t + M5)
    assert b["price"] == 3.0 and b["n_real_min"] == 3


def test_flat_fill_minutes_give_null_bars_not_repeated_prices():
    sched = one_session("BTC", date(2024, 1, 2))
    t0 = utc(2024, 1, 2, 10, 0)
    rows = [(t0 + k * M1, 100.0 + k, 100.5 + k, True) for k in range(5)]  # bar 10:05 real
    rows += [(t0 + k * M1, 104.5, 104.5, False) for k in range(5, 15)]  # zero-volume flat fill
    rows += [(t0 + k * M1, 200.0 + k, 200.5 + k, True) for k in range(15, 20)]  # bar 10:20 real
    bars, sess = bars_from_minutes(minutes_frame(rows), sched, "BTC")
    assert bar(bars, t0 + M5)["price"] == 104.5
    for k in (2, 3):
        b = bar(bars, t0 + k * M5)
        assert b["price"] is None and b["n_real_min"] == 0
    assert bar(bars, t0 + 4 * M5)["price"] == 219.5
    # No forward fill anywhere: every other bar of the day is null too.
    assert bars["price"].null_count() == 288 - 2
    assert sess["n_real_bars"].item() == 2 and sess["coverage"].item() == pytest.approx(2 / 288)


def test_full_grid_unique_keys_and_ts_end():
    sched = session_schedule("SPX", date(2025, 11, 24), date(2025, 12, 2))
    bars, sess = bars_from_minutes(minutes_frame([]), sched, "SPX")
    assert dict(bars.schema) == BARS_SCHEMA and dict(sess.schema) == SESSIONS_SCHEMA
    assert bars.height == int(sched["n_sched"].sum())
    assert bars.select(pl.struct("asset", "session_date", "bar_idx").is_unique().all()).item()
    j = bars.join(sched, on="session_date")
    assert (j["ts_end"] == j["open_utc"] + j["bar_idx"].cast(pl.Int64) * M5).all()
    assert (
        j.group_by("session_date")
        .agg(
            ok=(pl.col("bar_idx").max() == pl.col("n_sched").first())
            & (pl.col("bar_idx").min() == 1)
            & (pl.len() == pl.col("n_sched").first())
        )["ok"]
        .all()
    )
    # No minutes at all: every bar present with null price, sessions with zero coverage and no prices.
    assert bars["price"].null_count() == bars.height and (bars["n_real_min"] == 0).all()
    assert (sess["coverage"] == 0).all() and sess["p_open"].null_count() == sess.height
    assert sess["p_close"].null_count() == sess.height


# ----------------------------------------------------------------------------------------- p_open / p_close
def _spx_day(real_mask) -> pl.DataFrame:
    """SPX 2024-01-02 minutes 13:00..22:59 UTC; open = 1000 + k, close = 2000 + k for minute index k."""
    t0 = utc(2024, 1, 2, 13, 0)
    return minutes_frame([(t0 + k * M1, 1000.0 + k, 2000.0 + k, real_mask(t0 + k * M1)) for k in range(600)])


def test_p_open_p_close_definitions():
    sched = one_session("SPX", date(2024, 1, 2))  # 14:30 -> 21:00 UTC
    k = lambda h, mi: (h - 13) * 60 + mi  # noqa: E731 - minute index of hh:mm
    bars, sess = bars_from_minutes(_spx_day(lambda t: True), sched, "SPX")
    s = sess.row(0, named=True)
    assert s["p_open"] == 1000.0 + k(14, 30)  # OPEN of the 14:30 minute, not the pre-open 14:29 minute
    assert s["p_close"] == 2000.0 + k(20, 59)  # CLOSE of the minute closing at 21:00; 21:00 minute excluded
    assert s["n_real_bars"] == 78 and s["coverage"] == 1.0
    assert bars["n_real_min"].sum() == 390  # all non-RTH CFD minutes ignored
    assert bar(bars, utc(2024, 1, 2, 21, 0))["price"] == s["p_close"]
    assert bar(bars, utc(2024, 1, 2, 14, 35))["price"] == 2000.0 + k(14, 34)

    # First/last RTH minutes not real -> next/previous real minute is used.
    edge = {utc(2024, 1, 2, 14, 30), utc(2024, 1, 2, 20, 59), utc(2024, 1, 2, 14, 29), utc(2024, 1, 2, 21, 0)}
    _, sess = bars_from_minutes(_spx_day(lambda t: t not in edge), sched, "SPX")
    s = sess.row(0, named=True)
    assert s["p_open"] == 1000.0 + k(14, 31) and s["p_close"] == 2000.0 + k(20, 58)

    # Only pre/post-market real minutes -> no session prices.
    rth = lambda t: utc(2024, 1, 2, 14, 30) <= t < utc(2024, 1, 2, 21, 0)  # noqa: E731
    bars, sess = bars_from_minutes(_spx_day(lambda t: not rth(t)), sched, "SPX")
    s = sess.row(0, named=True)
    assert s["p_open"] is None and s["p_close"] is None and s["n_real_bars"] == 0
    assert bars["price"].null_count() == 78


def test_coverage_counts_real_bars():
    sched = one_session("SPX", date(2024, 1, 2))

    # Real only in the first 30 minutes (6 bars) and the 15:30..15:34 minute block (1 bar).
    def real(t: datetime) -> bool:
        return utc(2024, 1, 2, 14, 30) <= t < utc(2024, 1, 2, 15) or utc(2024, 1, 2, 15, 30) <= t < utc(
            2024, 1, 2, 15, 35
        )

    bars, sess = bars_from_minutes(_spx_day(real), sched, "SPX")
    assert sess["n_real_bars"].item() == 7 and sess["coverage"].item() == pytest.approx(7 / 78)
    assert bars.filter(pl.col("price").is_not_null())["bar_idx"].to_list() == [1, 2, 3, 4, 5, 6, 13]


def test_n_real_min_counts_real_rth_minutes_not_bars():
    """SPEC §1 SPX probe needs real RTH minutes per session; bar coverage cannot provide them.

    One real minute per bar gives coverage 1.0, yet only 20% of the RTH minutes are real. The silver bars must keep
    that information: sum(n_real_min) / (5 * n_sched) is the real-RTH-minute share (pre/post-market excluded).
    """
    sched = one_session("SPX", date(2024, 1, 2))
    rth0 = utc(2024, 1, 2, 14, 30)
    # Real: the 3rd minute of every RTH bar, plus every pre/post-market minute (the CFD trades there).
    real = lambda t: not (rth0 <= t < utc(2024, 1, 2, 21)) or (t - rth0) // M1 % 5 == 2  # noqa: E731
    bars, sess = bars_from_minutes(_spx_day(real), sched, "SPX")
    s = sess.row(0, named=True)
    assert s["coverage"] == 1.0 and s["n_real_bars"] == 78
    assert (bars["n_real_min"] == 1).all()
    assert bars["n_real_min"].sum() / (5 * s["n_sched"]) == pytest.approx(0.2)
    # Fully real RTH -> share 1.0; the n_real_min of a bar never exceeds its 5 minutes.
    full, sess_full = bars_from_minutes(_spx_day(lambda t: True), sched, "SPX")
    assert full["n_real_min"].max() == 5 and full["n_real_min"].sum() == 5 * sess_full["n_sched"].item()


def test_spx_half_day_42_bars():
    sched = one_session("SPX", date(2025, 11, 28))
    t0 = utc(2025, 11, 28, 14, 0)
    m = minutes_frame([(t0 + k * M1, 1.0, 1.0 + k, True) for k in range(300)])  # 14:00..18:59
    bars, sess = bars_from_minutes(m, sched, "SPX")
    assert bars.height == 42 and bars["ts_end"].max() == utc(2025, 11, 28, 18, 0)
    assert bars["price"].null_count() == 0 and bars["n_real_min"].sum() == 210
    assert sess["p_close"].item() == 1.0 + 239  # minute 17:59 closes at 18:00
    assert sess["coverage"].item() == 1.0


def test_fx_weekend_boundary_and_dst():
    # Winter Monday 2024-01-08 opens Sunday 22:00 UTC; summer Monday 2024-07-08 opens Sunday 21:00 UTC.
    for d, open_ in ((date(2024, 1, 8), utc(2024, 1, 7, 22)), (date(2024, 7, 8), utc(2024, 7, 7, 21))):
        sched = session_schedule("EURUSD", d - timedelta(days=3), d)  # Fri + Mon
        assert sched["session_date"].to_list() == [d - timedelta(days=3), d]
        m = minutes_frame([(open_ + k * M1, 1.0 + k, 1.5 + k, True) for k in range(-60, 60)])
        bars, sess = bars_from_minutes(m, sched, "EURUSD")
        mon = bars.filter(pl.col("session_date") == d)
        assert mon["ts_end"].min() == open_ + M5 and mon.height == 288
        assert bar(mon, open_ + M5)["price"] == 1.5 + 4 and bar(mon, open_ + M5)["n_real_min"] == 5
        s = sess.filter(pl.col("session_date") == d).row(0, named=True)
        assert s["p_open"] == 1.0  # first minute at the Sunday open
        assert bars["n_real_min"].sum() == 60  # the weekend hour before the open is ignored
        assert sess.filter(pl.col("session_date") == d - timedelta(days=3))["n_real_bars"].item() == 0


def test_input_normalisation_and_validation():
    sched = one_session("BTC", date(2024, 1, 2))
    m = minutes_frame([(utc(2024, 1, 2, 10, 4), 1.0, 2.0, True)])
    ref, _ = bars_from_minutes(m, sched, "BTC")
    naive_ms = m.with_columns(pl.col("ts").dt.replace_time_zone(None).dt.cast_time_unit("ms"))
    assert bars_from_minutes(naive_ms, sched, "BTC")[0].equals(ref)
    ny = m.with_columns(pl.col("ts").dt.convert_time_zone("America/New_York").dt.cast_time_unit("ns"))
    assert bars_from_minutes(ny, sched, "BTC")[0].equals(ref)
    with pytest.raises(ValueError, match="duplicate"):
        bars_from_minutes(pl.concat([m, m]), sched, "BTC")
    overlapping = pl.concat([sched, sched.with_columns(pl.col("session_date") + timedelta(days=1))])
    with pytest.raises(ValueError, match="overlapping"):
        bars_from_minutes(m, overlapping, "BTC")


# --------------------------------------------------------------------- randomized check vs a naive reference
def _reference(minutes: pl.DataFrame, sched: pl.DataFrame, asset: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Literal per-session / per-bar loop over SPEC §4.2 (slow; small inputs only)."""
    real = sorted((r["ts"], r["open"], r["close"]) for r in minutes.iter_rows(named=True) if r["is_real"])
    tss = [r[0] for r in real]
    brows, srows = [], []
    for s in sched.iter_rows(named=True):
        o, c, n = s["open_utc"], s["close_utc"], s["n_sched"]
        inside = real[bisect.bisect_left(tss, o - M1) : bisect.bisect_right(tss, c)]
        n_real = 0
        for k in range(1, n + 1):
            T = o + k * M5
            mem = [x for x in inside if T - M5 < x[0] + M1 <= T]
            brows.append((asset, s["session_date"], k, T, mem[-1][2] if mem else None, len(mem)))
            n_real += bool(mem)
        opens = [x[1] for x in inside if o <= x[0] < c]
        closes = [x[2] for x in inside if o < x[0] + M1 <= c]
        srows.append(
            (
                asset,
                s["session_date"],
                o,
                c,
                n,
                n_real,
                n_real / n,
                opens[0] if opens else None,
                closes[-1] if closes else None,
            )
        )
    return (
        pl.DataFrame(brows, schema=BARS_SCHEMA, orient="row"),
        pl.DataFrame(srows, schema=SESSIONS_SCHEMA, orient="row"),
    )


@pytest.mark.parametrize(
    "asset,start,end",
    [
        ("BTC", date(2026, 3, 7), date(2026, 3, 9)),
        ("EURUSD", date(2026, 3, 5), date(2026, 3, 10)),
        ("EURUSD", date(2024, 12, 23), date(2024, 12, 27)),
        ("SPX", date(2026, 10, 29), date(2026, 11, 3)),
        ("SPX", date(2025, 11, 26), date(2025, 12, 1)),
    ],
)
def test_matches_naive_reference(asset, start, end):
    rng = np.random.default_rng(C.seed())
    sched = session_schedule(asset, start, end)
    lo, hi = sched["open_utc"].min() - timedelta(hours=3), sched["close_utc"].max() + timedelta(hours=3)
    ts = pl.datetime_range(lo, hi, "1m", eager=True, time_unit="us")
    n = ts.len()
    px = 100.0 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    real = rng.random(n) < 0.6
    close = np.where(real, px, np.nan)
    close = pl.Series(close).fill_nan(None).forward_fill().fill_null(100.0).to_numpy()  # flat fills repeat
    m = pl.DataFrame({"ts": ts, "open": close * 0.999, "close": close, "is_real": real})
    m = m.filter(pl.Series(rng.random(n) < 0.8)).sample(fraction=1.0, shuffle=True, seed=1)  # gaps, unsorted
    got_b, got_s = bars_from_minutes(m, sched, asset)
    exp_b, exp_s = _reference(m, sched, asset)
    assert got_b.equals(exp_b)
    assert got_s.equals(exp_s)
    assert got_b["price"].null_count() > 0 and got_s["coverage"].min() < 1.0  # the test exercises nulls


# --------------------------------------------------------------------------------------- real sample files
_BI5 = np.dtype([("t", ">i4"), ("o", ">i4"), ("c", ">i4"), ("l", ">i4"), ("h", ">i4"), ("v", ">f4")])


def _dukascopy_minutes(instr: str, day: date, scale: float) -> pl.DataFrame:
    """Hand-decoded Dukascopy BID/ASK minute candles -> bronze-like mid minutes (SPEC §2.2)."""
    sides = {}
    for side in ("BID", "ASK"):
        rec = np.frombuffer(lzma.decompress((FIX / f"dukascopy_{instr}_{day}_{side}.bi5").read_bytes()), dtype=_BI5)
        sides[side] = pl.DataFrame({"t": rec["t"].astype(np.int64), "o": rec["o"], "c": rec["c"], "v": rec["v"]})
    j = sides["BID"].join(sides["ASK"], on="t", suffix="_a")
    midnight = datetime(day.year, day.month, day.day, tzinfo=UTC)
    return j.select(
        ts=pl.lit(midnight) + pl.duration(seconds=pl.col("t")),
        open=(pl.col("o") + pl.col("o_a")) / 2 / scale,
        close=(pl.col("c") + pl.col("c_a")) / 2 / scale,
        is_real=(pl.col("v") > 0) | (pl.col("v_a") > 0),
    ).with_columns(pl.col("ts").dt.cast_time_unit("us"))


def test_dukascopy_spx_fixture_78_rth_bars():
    d = date(2024, 1, 2)
    m = _dukascopy_minutes("USA500IDXUSD", d, 1e3)
    assert m.height == 1440 and m["is_real"].sum() > 390  # the CFD also trades outside RTH
    bars, sess = bars_from_minutes(m, one_session("SPX", d), "SPX")
    assert bars.height == 78 and bars["price"].null_count() == 0
    assert (bars["n_real_min"] == 5).all()
    assert bars["ts_end"].min() == utc(2024, 1, 2, 14, 35) and bars["ts_end"].max() == utc(2024, 1, 2, 21)
    s = sess.row(0, named=True)
    assert (s["n_sched"], s["n_real_bars"], s["coverage"]) == (78, 78, 1.0)
    first = m.filter(pl.col("ts") == utc(2024, 1, 2, 14, 30)).row(0, named=True)
    last = m.filter(pl.col("ts") == utc(2024, 1, 2, 20, 59)).row(0, named=True)
    assert s["p_open"] == first["open"] and s["p_close"] == last["close"]
    assert bars["price"][-1] == s["p_close"]
    assert 4000 < s["p_open"] < 5500 and abs(np.log(s["p_close"] / s["p_open"])) < 0.05


def test_dukascopy_eurusd_fixture_session_split():
    d = date(2024, 1, 2)
    m = _dukascopy_minutes("EURUSD", d, 1e5)
    sched = session_schedule("EURUSD", d, d + timedelta(days=1))  # (Jan 1 22:00, Jan 2 22:00], (.., Jan 3 22:00]
    bars, sess = bars_from_minutes(m, sched, "EURUSD")
    assert bars.height == 2 * 288
    # Every real minute of the UTC day lands in exactly one of the two sessions.
    assert bars["n_real_min"].sum() == m["is_real"].sum()
    b2 = bars.filter(pl.col("session_date") == d)
    assert b2.filter(pl.col("ts_end") <= utc(2024, 1, 2))["price"].null_count() == 24  # Jan 1 evening: no data
    b3 = bars.filter(pl.col("session_date") == d + timedelta(days=1))
    assert b3.filter(pl.col("ts_end") > utc(2024, 1, 3))["n_real_min"].sum() == 0
    assert b3.filter(pl.col("ts_end") <= utc(2024, 1, 3))["n_real_min"].sum() > 0  # 22:00-24:00 Jan 2
    s2 = sess.filter(pl.col("session_date") == d).row(0, named=True)
    assert s2["coverage"] <= 264 / 288 and 1.0 < s2["p_open"] < 1.2 and 1.0 < s2["p_close"] < 1.2


def _binance_minutes(day: str) -> pl.DataFrame:
    with zipfile.ZipFile(FIX / f"BTCUSDT-1m-{day}.zip") as z:
        raw = z.read(z.namelist()[0])
    df = pl.read_csv(
        io.BytesIO(raw),
        has_header=False,
        new_columns=["open_time", "open", "high", "low", "close", "volume", "close_time"],
        columns=list(range(7)),
    )
    unit = "us" if df["open_time"][0] > 10**14 else "ms"  # Binance switched to µs in 2025
    return df.select(
        ts=pl.from_epoch("open_time", time_unit=unit).dt.cast_time_unit("us").dt.replace_time_zone("UTC"),
        open=pl.col("open").cast(pl.Float64),
        close=pl.col("close").cast(pl.Float64),
        is_real=pl.col("volume") > 0,
    )


def test_binance_btc_fixtures_ms_and_us():
    m = pl.concat([_binance_minutes("2024-12-31"), _binance_minutes("2025-01-01")])
    assert m.height == 2880 and m["ts"].is_unique().all()
    bars, sess = bars_from_minutes(m, session_schedule("BTC", date(2024, 12, 31), date(2025, 1, 1)), "BTC")
    assert bars.height == 576 and sess["n_sched"].to_list() == [288, 288]
    assert bars["n_real_min"].sum() == m["is_real"].sum()
    assert sess["coverage"].min() > 0.99
    jan1_open = m.filter(pl.col("ts") == utc(2025, 1, 1)).row(0, named=True)["open"]
    dec31_close = m.filter(pl.col("ts") == utc(2024, 12, 31, 23, 59)).row(0, named=True)["close"]
    assert sess["p_open"][1] == jan1_open and sess["p_close"][0] == dec31_close
    assert abs(np.log(jan1_open / dec31_close)) < 1e-3  # continuous 24/7 market: ~zero gap


# ------------------------------------------------------------------------------------------------ build_bars
def _write_bronze(root: Path, asset: str, year: int, df: pl.DataFrame) -> None:
    p = root / "minute" / f"asset={asset}" / f"year={year}" / "part.parquet"
    p.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(p)


def _btc_bronze(bronze: Path, shift: float = 0.0) -> pl.DataFrame:
    """Two BTC days (2023-12-31, 2024-01-01) of SPEC §3 bronze minutes in two year partitions; every 7th is fake."""
    t0 = utc(2023, 12, 31)
    full = pl.DataFrame(
        {
            "ts": pl.datetime_range(t0, t0 + timedelta(days=2) - M1, "1m", eager=True, time_unit="us"),
        }
    ).with_columns(
        open=pl.int_range(pl.len()).cast(pl.Float64) + 100.0 + shift,
        high=pl.lit(1e9),
        low=pl.lit(0.0),
        close=pl.int_range(pl.len()).cast(pl.Float64) + 100.5 + shift,
        volume=pl.when(pl.int_range(pl.len()) % 7 == 0).then(0.0).otherwise(1.0),
        is_real=pl.int_range(pl.len()) % 7 != 0,
        bid_close=pl.lit(None, pl.Float64),
        ask_close=pl.lit(None, pl.Float64),
        spread=pl.lit(None, pl.Float64),
        n_trades=pl.lit(3, pl.Int64),
    )
    for y in (2023, 2024):
        _write_bronze(bronze, "BTC", y, full.filter(pl.col("ts").dt.year() == y))
    return full


BTC_RANGE = {"start": date(2023, 12, 31), "end": date(2024, 1, 1)}


def test_build_bars_writes_silver(tmp_path):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    full = _btc_bronze(bronze)
    bad = bronze / "minute" / "asset=BTC" / "year=2020" / "part.parquet"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b"not parquet")  # outside the requested range -> must not be read

    summary = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **BTC_RANGE)
    bp, sp = bars_path("BTC", silver), sessions_path("BTC", silver)
    assert bp == silver / "bars5m" / "asset=BTC" / "part.parquet" and bp.exists() and sp.exists()
    bars, sess = pl.read_parquet(bp), pl.read_parquet(sp)
    assert dict(bars.schema) == BARS_SCHEMA and dict(sess.schema) == SESSIONS_SCHEMA
    exp_b, exp_s = bars_from_minutes(full, session_schedule("BTC", date(2023, 12, 31), date(2024, 1, 1)), "BTC")
    assert bars.equals(exp_b) and sess.equals(exp_s)
    assert summary["n_sessions"] == 2 and summary["n_bars"] == 576
    assert summary["n_real_minutes"] == summary["n_real_minutes_used"] == int(full["is_real"].sum())
    assert summary["n_real_minutes_outside_sessions"] == 0 and summary["n_no_price"] == 0
    assert summary["bars_path"] == str(bp) and summary["first_session"] == date(2023, 12, 31)
    assert sess["p_open"].to_list() == [101.0, 100.0 + 1440]  # minute k=0 is a flat fill (k % 7 == 0)
    assert summary["skipped"] is False and summary["cfg_hash"] == C.cfg_hash()
    assert summary["n_dev_sessions"] == 2 and summary["n_holdout_sessions"] == 0
    assert not list(tmp_path.rglob("*.part"))


# ----------------------------------------------------------------- SPEC §11: no holdout statistics in the summary
_DIAG_KEYS = (
    "n_null_bars",
    "n_real_minutes",
    "n_real_minutes_used",
    "n_real_minutes_outside_sessions",
    "mean_coverage",
    "n_low_coverage",
    "n_no_price",
)


def _boundary_bronze(bronze: Path) -> pl.DataFrame:
    """Synthetic BTC minutes around the real holdout boundary: a clean dev day and two broken holdout days.

    Last dev day: every 7th minute is a flat fill (no null bar, coverage 1). First holdout day: real minutes only
    in its first hour (coverage 12/288); second holdout day: no minutes at all (no price).
    """
    h0 = C.holdout_start()
    t0 = datetime(h0.year, h0.month, h0.day, tzinfo=UTC) - timedelta(days=1)
    dev = pl.DataFrame({"ts": pl.datetime_range(t0, t0 + timedelta(days=1) - M1, "1m", eager=True, time_unit="us")})
    dev = dev.with_columns(is_real=pl.int_range(pl.len()) % 7 != 0)
    t1 = t0 + timedelta(days=1)
    hold = pl.DataFrame({"ts": pl.datetime_range(t1, t1 + timedelta(hours=1) - M1, "1m", eager=True, time_unit="us")})
    hold = hold.with_columns(is_real=pl.lit(True))
    full = pl.concat([dev, hold]).with_columns(
        open=pl.int_range(pl.len()).cast(pl.Float64) + 100.0,
        close=pl.int_range(pl.len()).cast(pl.Float64) + 100.5,
    )
    assert full["ts"].dt.year().n_unique() == 1
    _write_bronze(bronze, "BTC", t0.year, full)
    return full


def test_build_bars_summary_has_no_holdout_statistics(tmp_path, caplog):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    full = _boundary_bronze(bronze)
    h0 = C.holdout_start()
    rng = {"start": h0 - timedelta(days=1), "end": h0 + timedelta(days=1)}
    with caplog.at_level("INFO", logger="volrisk.bars"):
        summary = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)

    # Silver itself is not split (gold is): the holdout sessions are built, with their real defects ...
    sess = pl.read_parquet(sessions_path("BTC", silver))
    assert sess["session_date"].to_list() == [rng["start"], h0, rng["end"]]
    assert sess["coverage"].to_list()[1:] == [12 / 288, 0.0] and sess["p_open"].null_count() == 1
    # ... but the summary counts them only; every diagnostic is that of the dev session alone.
    dev_real = int(full.filter(pl.col("ts").dt.date() < h0)["is_real"].sum())
    assert summary["n_sessions"] == 3 and summary["n_bars"] == 3 * 288
    assert summary["n_dev_sessions"] == 1 and summary["n_holdout_sessions"] == 2
    assert summary["diagnostics_scope"] == f"session_date < {h0.isoformat()}"
    assert {k: summary[k] for k in _DIAG_KEYS} == {
        "n_null_bars": 0,
        "n_real_minutes": dev_real,
        "n_real_minutes_used": dev_real,
        "n_real_minutes_outside_sessions": 0,
        "mean_coverage": 1.0,
        "n_low_coverage": 0,
        "n_no_price": 0,
    }
    # The build marker stores the same dev-only summary; the log line shows no holdout statistic.
    rec = json.loads(marker_path("BTC", silver).read_text(encoding="utf-8"))
    assert {k: rec["summary"][k] for k in _DIAG_KEYS} == {k: summary[k] for k in _DIAG_KEYS}
    msg = [r.getMessage() for r in caplog.records if r.name == "volrisk.bars"][-1]
    assert "mean coverage 1.000, 0 below" in msg and "holdout: 2 sessions (row count only)" in msg
    assert build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)["skipped"] is True


def test_build_bars_summary_holdout_only_range(tmp_path):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _boundary_bronze(bronze)
    h0 = C.holdout_start()
    rng = {"start": h0, "end": h0 + timedelta(days=1)}
    summary = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)
    assert summary["n_dev_sessions"] == 0 and summary["n_holdout_sessions"] == 2
    assert {k: summary[k] for k in _DIAG_KEYS} == dict.fromkeys(_DIAG_KEYS, 0) | {"mean_coverage": None}
    again = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)
    assert again["skipped"] is True and again["mean_coverage"] is None


# -------------------------------------------------------------------------------- SPEC §12 skip / --force
def _no_recompute(*_a, **_k):
    raise AssertionError("bars were recomputed although the outputs are current")


def test_build_bars_skips_current_outputs_and_force_rebuilds(tmp_path, monkeypatch):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _btc_bronze(bronze)
    first = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **BTC_RANGE)
    bp, sp, mp = bars_path("BTC", silver), sessions_path("BTC", silver), marker_path("BTC", silver)
    assert mp == silver / "_build" / "bars_BTC.json" and mp.is_file()
    rec = json.loads(mp.read_text(encoding="utf-8"))
    assert rec["key"]["cfg_hash"] == C.cfg_hash()
    assert set(rec["key"]["bronze_sha256"]) == {"year=2023/part.parquet", "year=2024/part.parquet"}
    assert set(rec["outputs"]) == {"bars5m", "sessions"}
    blobs = bp.read_bytes(), sp.read_bytes()

    # Same config, schedule, bronze and code -> no recompute, same summary, outputs untouched.
    monkeypatch.setattr(bars_mod, "bars_from_minutes", _no_recompute)
    again = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **BTC_RANGE)
    assert again["skipped"] is True
    assert {k: v for k, v in again.items() if k != "skipped"} == {k: v for k, v in first.items() if k != "skipped"}
    assert isinstance(again["first_session"], date) and (bp.read_bytes(), sp.read_bytes()) == blobs

    # force=True always recomputes (the patched builder proves it is reached).
    with pytest.raises(AssertionError, match="recomputed"):
        build_bars("BTC", bronze_dir=bronze, out_dir=silver, force=True, **BTC_RANGE)
    monkeypatch.undo()
    forced = build_bars("BTC", bronze_dir=bronze, out_dir=silver, force=True, **BTC_RANGE)
    assert forced["skipped"] is False and (bp.read_bytes(), sp.read_bytes()) == blobs  # deterministic output


@pytest.mark.parametrize(
    "change", ["config", "bronze", "range", "bars_output", "sessions_output", "marker", "marker_summary"]
)
def test_build_bars_rebuilds_when_anything_changed(tmp_path, monkeypatch, change):
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    _btc_bronze(bronze)
    build_bars("BTC", bronze_dir=bronze, out_dir=silver, **BTC_RANGE)
    bp, sp, mp = bars_path("BTC", silver), sessions_path("BTC", silver), marker_path("BTC", silver)
    rng = dict(BTC_RANGE)
    if change == "config":
        monkeypatch.setattr(C, "cfg_hash", lambda: "0123456789abcdef")
    elif change == "bronze":
        _btc_bronze(bronze, shift=1.0)  # re-downloaded / rebuilt bronze with different prices
    elif change == "range":
        rng["end"] = date(2023, 12, 31)
    elif change == "bars_output":
        bp.unlink()
    elif change == "sessions_output":
        pl.read_parquet(sp).head(1).write_parquet(sp)  # truncated / overwritten by someone else
    elif change == "marker":
        mp.write_text("{not json", encoding="utf-8")
    else:
        rec = json.loads(mp.read_text(encoding="utf-8"))
        del rec["summary"]["first_session"]
        mp.write_text(json.dumps(rec), encoding="utf-8")
    summary = build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)
    assert summary["skipped"] is False
    sched = session_schedule("BTC", rng["start"], rng["end"])
    minutes = bars_mod.read_minutes("BTC", bronze, sched["open_utc"].min(), sched["close_utc"].max())
    exp_b, exp_s = bars_from_minutes(minutes, sched, "BTC")
    assert pl.read_parquet(bp).equals(exp_b) and pl.read_parquet(sp).equals(exp_s)
    if change == "bronze":
        assert pl.read_parquet(sp)["p_open"].to_list() == [102.0, 101.0 + 1440]
    rec = json.loads(mp.read_text(encoding="utf-8"))
    assert rec["key"]["cfg_hash"] == C.cfg_hash() and rec["key"]["end"] == rng["end"].isoformat()
    # ... and the fresh marker makes the next call a skip again.
    assert build_bars("BTC", bronze_dir=bronze, out_dir=silver, **rng)["skipped"] is True


def test_build_bars_missing_bronze(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_bars("ETH", bronze_dir=tmp_path, out_dir=tmp_path, start=date(2024, 1, 1), end=date(2024, 1, 2))


def test_default_paths_point_at_silver():
    assert bars_path("SPX") == C.SILVER / "bars5m" / "asset=SPX" / "part.parquet"
    assert sessions_path("SPX") == C.SILVER / "sessions" / "asset=SPX" / "part.parquet"
