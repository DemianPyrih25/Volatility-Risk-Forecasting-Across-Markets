"""Data-quality report (SPEC §10) on synthetic bronze/silver/gold data written to tmp_path."""

from __future__ import annotations

import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from volrisk import io
from volrisk import quality as Q

UTC = timezone.utc
TS = pl.Datetime("us", "UTC")
M1 = timedelta(minutes=1)
FIX = Path(__file__).parent / "fixtures"


# --------------------------------------------------------------------------------------------- synthetic layers
def utc(d: date, h: int = 0, m: int = 0, s: int = 0) -> datetime:
    return datetime(d.year, d.month, d.day, h, m, s, tzinfo=UTC)


def weekdays(start: date, end: date) -> list[date]:
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [d for d in days if d.weekday() < 5]


def rth_schedule(dates: list[date]):
    """SPX-like test calendar: 14:30-21:00 UTC (78 bars) on the given dates."""

    def fn(asset: str, start: date, end: date) -> pl.DataFrame:
        ds = [d for d in dates if start <= d <= end]
        return pl.DataFrame(
            {
                "session_date": ds,
                "open_utc": [utc(d, 14, 30) for d in ds],
                "close_utc": [utc(d, 21) for d in ds],
                "n_sched": [78] * len(ds),
            },
            schema={"session_date": pl.Date, "open_utc": TS, "close_utc": TS, "n_sched": pl.Int32},
        )

    return fn


def utc_day_schedule(asset: str, start: date, end: date) -> pl.DataFrame:
    """Crypto-like test calendar: UTC days, 288 bars."""
    ds = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    return pl.DataFrame(
        {
            "session_date": ds,
            "open_utc": [utc(d) for d in ds],
            "close_utc": [utc(d + timedelta(days=1)) for d in ds],
            "n_sched": [288] * len(ds),
        },
        schema={"session_date": pl.Date, "open_utc": TS, "close_utc": TS, "n_sched": pl.Int32},
    )


def minutes_frame(ts: list[datetime], close: np.ndarray, is_real=None, spread=None, open_=None) -> pl.DataFrame:
    n = len(ts)
    close = np.asarray(close, dtype=float)
    return pl.DataFrame(
        {
            "ts": ts,
            "open": np.r_[close[0], close[:-1]] if open_ is None else open_,
            "high": close,
            "low": close,
            "close": close,
            "volume": np.ones(n),
            "is_real": [True] * n if is_real is None else list(is_real),
            "spread": [None] * n if spread is None else list(spread),
        },
        schema={"ts": TS, "open": pl.Float64, "high": pl.Float64, "low": pl.Float64, "close": pl.Float64,
                "volume": pl.Float64, "is_real": pl.Boolean, "spread": pl.Float64},
    )


def write_bronze(bronze: Path, asset: str, df: pl.DataFrame) -> None:
    for (year,), part in df.group_by(pl.col("ts").dt.year(), maintain_order=True):
        p = bronze / "minute" / f"asset={asset}" / f"year={year}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        part.sort("ts").write_parquet(p)


def build_silver(silver: Path, asset: str, minutes: pl.DataFrame, sched: pl.DataFrame) -> pl.DataFrame:
    """Minimal SPEC §4.2 bars/sessions (independent of volrisk.bars) written under ``silver``."""
    aligned = pl.col("ts").dt.truncate("1m") == pl.col("ts")
    real = minutes.filter(pl.col("is_real") & aligned).unique("ts", keep="last").sort("ts")
    real = real.with_columns(tc=pl.col("ts") + M1)
    j = (
        real.join_asof(sched.sort("open_utc"), left_on="tc", right_on="open_utc", strategy="backward",
                       allow_exact_matches=False)
        .filter(pl.col("tc") <= pl.col("close_utc"))
        .with_columns(bar_idx=(((pl.col("tc") - pl.col("open_utc")).dt.total_minutes() + 4) // 5).cast(pl.Int32))
    )
    in_bar = j.group_by("session_date", "bar_idx").agg(price=pl.col("close").last(), n_real_min=pl.len())
    grid = (
        sched.with_columns(bar_idx=pl.int_ranges(1, pl.col("n_sched") + 1, dtype=pl.Int32))
        .explode("bar_idx", empty_as_null=True)
        .with_columns(ts_end=pl.col("open_utc") + pl.duration(minutes=5) * pl.col("bar_idx"))
    )
    bars = grid.join(in_bar, on=["session_date", "bar_idx"], how="left").with_columns(asset=pl.lit(asset))
    bars = bars.select("asset", "session_date", "bar_idx", "ts_end", "price",
                       pl.col("n_real_min").fill_null(0).cast(pl.Int32)).sort("session_date", "bar_idx")
    ends = j.group_by("session_date").agg(p_open=pl.col("open").first(), p_close=pl.col("close").last())
    nb = bars.group_by("session_date").agg(n_real_bars=pl.col("price").is_not_null().sum().cast(pl.Int32))
    sess = (
        sched.join(nb, on="session_date", how="left")
        .join(ends, on="session_date", how="left")
        .with_columns(asset=pl.lit(asset), coverage=pl.col("n_real_bars") / pl.col("n_sched"))
        .select("asset", "session_date", "open_utc", "close_utc", "n_sched", "n_real_bars", "coverage", "p_open",
                "p_close")
        .sort("session_date")
    )
    write_silver(silver, asset, sess, bars)
    return sess


def write_silver(silver: Path, asset: str, sessions: pl.DataFrame, bars: pl.DataFrame) -> None:
    for name, df in (("sessions", sessions), ("bars5m", bars)):
        p = silver / name / f"asset={asset}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)


def silver_from_returns(dates, returns: np.ndarray, open_hour=(14, 30), p0=100.0, coverage=None):
    """Silver sessions/bars with given per-bar percent returns (rows = sessions; NaN return = null bar)."""
    n_sess, n_bar = returns.shape
    s_rows, b_rows = [], []
    for k, d in enumerate(dates):
        o = utc(d, *open_hour)
        last = p0
        for i in range(n_bar):
            r = returns[k, i]
            if np.isnan(r):
                b_rows.append((d, i + 1, o + timedelta(minutes=5 * (i + 1)), None))
                continue
            p = last * np.exp(r / 100)
            last = p
            b_rows.append((d, i + 1, o + timedelta(minutes=5 * (i + 1)), p))
        n_real = int(np.isfinite(returns[k]).sum())
        cov = n_real / n_bar if coverage is None else coverage[k]
        s_rows.append((d, o, o + timedelta(minutes=5 * n_bar), n_bar, n_real, cov, p0, last))
    sessions = pl.DataFrame(
        s_rows, orient="row",
        schema={"session_date": pl.Date, "open_utc": TS, "close_utc": TS, "n_sched": pl.Int32,
                "n_real_bars": pl.Int32, "coverage": pl.Float64, "p_open": pl.Float64, "p_close": pl.Float64},
    )
    bars = pl.DataFrame(
        b_rows, orient="row",
        schema={"session_date": pl.Date, "bar_idx": pl.Int32, "ts_end": TS, "price": pl.Float64},
    ).with_columns(n_real_min=pl.lit(5, pl.Int32))
    return sessions, bars


# --------------------------------------------------------------------------------------------- coverage
SPX_DATES = [date(2024, 1, 30), date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 5)]
SPX_SCHED_DATES = sorted([*SPX_DATES, date(2024, 2, 2)])  # 2024-02-02 is scheduled but has no data at all


@pytest.fixture
def spx_tree(tmp_path: Path):
    rng = np.random.default_rng(1)
    frames = []
    for d in SPX_DATES:
        ts = [utc(d, 13) + i * M1 for i in range(9 * 60)]  # 13:00-21:59, RTH is 14:30-20:59 (minute opens)
        close = 5000 * np.exp(np.cumsum(rng.normal(0, 2e-4, len(ts))))
        df = minutes_frame(ts, close, spread=np.full(len(ts), 0.5))
        hm = pl.col("ts").dt.hour().cast(pl.Int32) * 60 + pl.col("ts").dt.minute()
        if d == date(2024, 1, 30):
            df = df.filter(~hm.is_between(13 * 60 + 10, 13 * 60 + 30))  # gap outside the session: ignored
            df = df.with_columns(spread=pl.when(hm.is_between(15 * 60, 15 * 60 + 7)).then(5.0).otherwise(0.5))
        if d == date(2024, 1, 31):
            drop = (hm.is_between(15 * 60, 15 * 60 + 9) | hm.is_between(16 * 60, 16 * 60 + 4)
                    | hm.is_between(17 * 60, 17 * 60 + 3))  # 10 / 5 / 4 missing -> intervals 11, 6, 5 min
            df = df.filter(~drop)
        if d == date(2024, 2, 1):
            df = df.with_columns(is_real=~hm.is_between(18 * 60, 20 * 60 + 49))  # flat fills: 34 null bars
            dup = df.filter(pl.col("ts") == utc(d, 15))
            odd = dup.with_columns(ts=pl.lit(utc(d, 15, 30, 30)).cast(TS))  # misaligned timestamp
            df = pl.concat([df, dup, odd])
        frames.append(df)
    minutes = pl.concat(frames)
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    write_bronze(bronze, "SPX", minutes)
    sched = rth_schedule(SPX_SCHED_DATES)
    silver_sched = sched("SPX", SPX_DATES[0], SPX_DATES[-1]).filter(pl.col("session_date") != date(2024, 2, 2))
    build_silver(silver, "SPX", minutes, silver_sched)
    return bronze, silver, sched


def test_monthly_coverage_counts(spx_tree):
    bronze, silver, sched = spx_tree
    m = Q.monthly_coverage("SPX", bronze, silver, schedule_fn=sched)
    assert m["month"].to_list() == [date(2024, 1, 1), date(2024, 2, 1)]
    jan, feb = m.row(0, named=True), m.row(1, named=True)

    assert jan["n_sessions"] == 2 and jan["sched_bars"] == 156 and jan["sched_min"] == 780
    assert jan["real_min"] == 390 + 390 - 19  # out-of-session minutes and gaps excluded
    assert jan["real_bars"] == 78 + 75  # 15:00-15:09 and 16:00-16:04 empty three bars
    assert jan["n_gaps_gt5"] == 2  # 11- and 6-minute intervals; the 5-minute one is not a gap
    assert jan["max_gap_min"] == 11
    assert jan["n_dup_bronze"] == 0 and jan["n_misaligned_bronze"] == 0
    assert jan["n_dropped"] == 1 and jan["n_flagged"] == 0  # 2024-01-30: first session, no previous close
    assert jan["n_pre_start"] == 0
    assert jan["spread_med"] == pytest.approx(0.5)
    assert jan["spread_p99"] > 0.5
    assert jan["spread_med_bp"] == pytest.approx(1e4 * 0.5 / 5000, rel=0.1)
    # no flag file and no gold table: raw flag counts and jump share are unknown, not zero
    assert all(jan[c] is None for c in Q.FLAG_COLUMNS)
    assert jan["n_gold_days"] is None and jan["jump_share"] is None and jan["jump_suspect"] is None

    assert feb["n_sessions"] == 3 and feb["sched_bars"] == 234
    assert feb["real_bars"] == 44 + 0 + 78
    assert feb["null_bar_share"] == pytest.approx(1 - 122 / 234)
    assert feb["real_min"] == 220 + 390  # duplicate de-duplicated, misaligned row not used
    assert feb["n_dup_bronze"] == 1 and feb["n_misaligned_bronze"] == 1  # bronze integrity check
    assert feb["n_gaps_gt5"] == 1 and feb["max_gap_min"] == 171  # 17:59 -> 20:50
    assert feb["n_missing"] == 1 and feb["n_low_cov"] == 1 and feb["n_dropped"] == 2


def test_monthly_coverage_without_schedule_uses_silver_only(spx_tree):
    bronze, silver, _ = spx_tree
    m = Q.monthly_coverage("SPX", bronze, silver)
    feb = m.row(1, named=True)
    assert feb["n_sessions"] == 2 and feb["n_missing"] == 0 and feb["n_dropped"] == 1


def test_monthly_coverage_jump_share_per_month(spx_tree):
    bronze, silver, sched = spx_tree
    daily = pd.DataFrame({
        "asset": ["SPX", "SPX", "BTC"],
        "session_date": pd.to_datetime(["2024-01-31", "2024-02-05", "2024-02-05"]),
        "j": [0.1, 0.0, 0.5],
    })
    m = Q.monthly_coverage("SPX", bronze, silver, schedule_fn=sched, daily=daily)
    jan, feb = m.row(0, named=True), m.row(1, named=True)
    assert (jan["n_gold_days"], jan["n_jump_days"], jan["jump_share"], jan["jump_suspect"]) == (1, 1, 1.0, True)
    assert (feb["n_gold_days"], feb["n_jump_days"], feb["jump_share"], feb["jump_suspect"]) == (1, 0, 0.0, False)


def test_yearly_coverage_aggregates(spx_tree):
    bronze, silver, sched = spx_tree
    y = Q.yearly_coverage(Q.monthly_coverage("SPX", bronze, silver, schedule_fn=sched))
    row = y.row(0, named=True)
    assert y.height == 1 and row["year"] == 2024
    assert row["n_sessions"] == 5 and row["n_gaps_gt5"] == 3 and row["n_dropped"] == 3
    assert row["null_bar_share"] == pytest.approx(1 - (78 + 75 + 44 + 78) / 390)
    assert row["n_dup_raw"] is None and row["n_misaligned_days"] is None  # unknown stays null, not 0
    assert row["n_dup_bronze"] == 1


def test_session_issues_crypto_flags_but_keeps(tmp_path: Path):
    ds = [date(2024, 5, 1), date(2024, 5, 2), date(2024, 5, 3), date(2024, 5, 5), date(2024, 5, 6)]  # 05-04 absent
    sess, bars = silver_from_returns(ds, np.zeros((5, 288)), open_hour=(0, 0),
                                     coverage=[1.0, 0.5, 0.0, 1.0, 0.3])
    sess = sess.with_columns(
        p_open=pl.when(pl.col("session_date") == ds[2]).then(None).otherwise(pl.col("p_open")),
        p_close=pl.when(pl.col("session_date") == ds[2]).then(None).otherwise(pl.col("p_close")),
    )
    write_silver(tmp_path, "BTC", sess, bars)
    out = Q.session_issues("BTC", tmp_path, schedule_fn=utc_day_schedule, until=date(2024, 5, 31))
    got = {r["session_date"]: (r["reason"], r["action"]) for r in out.iter_rows(named=True)}
    assert got == {
        ds[0]: ("no_prev_close", "dropped"),  # first session with prices: gold has no t-1 close for it
        ds[1]: ("low_coverage", "flagged"),  # crypto: never dropped for low coverage
        ds[2]: ("no_price", "dropped"),  # no real minute at all
        date(2024, 5, 4): ("missing", "dropped"),  # only gaps inside the silver range count as missing
        ds[3]: (None, None),
        ds[4]: ("low_coverage", "flagged"),
    }


def test_dev_only_cut(spx_tree):
    bronze, silver, sched = spx_tree
    m = Q.monthly_coverage("SPX", bronze, silver, schedule_fn=sched, until=date(2024, 1, 31))
    assert m["month"].to_list() == [date(2024, 1, 1)]
    assert Q.read_bronze("SPX", bronze, until=date(2024, 1, 30))["ts"].max() < utc(date(2024, 1, 31))


def test_holdout_dates_refused_while_sealed(spx_tree, tmp_path: Path, monkeypatch):
    bronze, silver, sched = spx_tree
    monkeypatch.setattr(io, "_UNLOCKED", False)  # sealed (restored after the test)
    holdout_day = date(2025, 10, 1)
    assert Q._until(None) == date(2025, 9, 30)
    for call in (
        lambda: Q._until(date(2026, 9, 30)),
        lambda: Q.read_bronze("SPX", bronze, until=holdout_day),
        lambda: Q.monthly_coverage("SPX", bronze, silver, schedule_fn=sched, until=holdout_day),
        lambda: Q.read_flags(tmp_path / "flags", until=holdout_day),
        lambda: Q.top_returns("SPX", silver, until=holdout_day),
        lambda: Q.jump_review("SPX", pd.DataFrame({"asset": [], "session_date": [], "j": []}), silver,
                              until=holdout_day),
    ):
        with pytest.raises(io.HoldoutSealedError):
            call()
    out = tmp_path / "rep" / "dq.md"
    with pytest.raises(io.HoldoutSealedError):
        Q.build_report(assets=("SPX",), out_md=out, fig_dir=tmp_path / "rep", table_dir=None, bronze_dir=bronze,
                       silver_dir=silver, implied_dir=tmp_path, daily=pd.DataFrame(), schedule_fn=sched,
                       until=date(2026, 9, 30))
    assert not out.exists()

    monkeypatch.setattr(io, "_UNLOCKED", True)  # what volrisk.holdout.unlock does after verifying the seal
    assert Q._until(date(2026, 9, 30)) == date(2026, 9, 30)


# --------------------------------------------------------------------------------------------- gold membership
def _probe_tree(silver: Path):
    """SPX: April 2024 sessions with full bars but only 40% real minutes (fails the SPEC §1 probe) and a huge
    return; May 2024 complete. BTC: five days incl. a low-coverage and a price-less session."""
    april, may = weekdays(date(2024, 4, 1), date(2024, 4, 30)), weekdays(date(2024, 5, 1), date(2024, 5, 31))
    rng = np.random.default_rng(8)
    ret = rng.normal(0, 0.02, (len(april) + len(may), 78))
    ret[3, 20] = 9.0  # pre-start spike: never modelled, must not reach the top-returns list
    ret[len(april) + 2, 30] = 1.5  # in-sample spike
    cov = [1.0] * (len(april) + len(may))
    cov[5] = 0.5  # one pre-start session with low coverage
    sess, bars = silver_from_returns(april + may, ret, coverage=cov)
    bars = bars.with_columns(n_real_min=pl.when(pl.col("session_date") < date(2024, 5, 1)).then(2).otherwise(5)
                             .cast(pl.Int32))
    write_silver(silver, "SPX", sess, bars.with_columns(asset=pl.lit("SPX")))

    ds = [date(2024, 5, 1) + timedelta(days=i) for i in range(5)]
    bs, bb = silver_from_returns(ds, rng.normal(0, 0.02, (5, 288)), open_hour=(0, 0),
                                 coverage=[1.0, 0.5, 1.0, 1.0, 1.0])
    no_px = pl.col("session_date") == ds[2]
    bs = bs.with_columns(p_open=pl.when(no_px).then(None).otherwise(pl.col("p_open")),
                         p_close=pl.when(no_px).then(None).otherwise(pl.col("p_close")))
    write_silver(silver, "BTC", bs, bb.with_columns(asset=pl.lit("BTC")))
    return april, may, ds


def test_gold_membership_follows_sample_start_and_first_close(tmp_path: Path):
    april, may, _ = _probe_tree(tmp_path)
    assert Q.sample_start("SPX", tmp_path) == date(2024, 5, 1)
    assert Q.sample_start("BTC", tmp_path) == date(2018, 1, 1)

    issues = Q.session_issues("SPX", tmp_path)
    got = dict(zip(issues["session_date"].to_list(), zip(issues["reason"].to_list(), issues["action"].to_list())))
    assert got[april[0]] == ("before_sample_start", "dropped")
    assert got[april[5]] == ("low_coverage", "dropped")  # the data problem is reported first
    assert all(got[d] == ("before_sample_start", "dropped") for d in april if d != april[5])
    assert all(got[d] == (None, None) for d in may)  # May 1 has its previous close from April

    top = Q.top_returns("SPX", tmp_path, n=1)
    assert top["session_date"][0] == may[2] and top["r"][0] == pytest.approx(1.5)
    assert Q.top_returns("SPX", tmp_path, n=1, valid_only=False)["r"][0] == pytest.approx(9.0)
    r = Q.intraday_returns("SPX", tmp_path)
    assert r["session_date"].min() == may[0] and r["session_date"].n_unique() == len(may)


def test_gold_membership_matches_measures_build_gold(tmp_path: Path):
    """The report's notion of 'enters the gold table' is the gold builder's (volrisk.measures)."""
    from volrisk import measures

    _probe_tree(tmp_path / "silver")
    measures.build_gold(("SPX", "BTC"), tmp_path / "silver", tmp_path / "gold", tmp_path / "holdout")
    gold = pl.read_parquet(tmp_path / "gold" / "daily.parquet")
    for a in ("SPX", "BTC"):
        issues = Q.session_issues(a, tmp_path / "silver")
        kept = issues.filter(pl.col("action").is_null() | (pl.col("action") == "flagged"))["session_date"]
        assert sorted(kept.to_list()) == sorted(gold.filter(pl.col("asset") == a)["session_date"].to_list()), a
        in_returns = Q.intraday_returns(a, tmp_path / "silver")["session_date"].unique().sort()
        assert in_returns.to_list() == sorted(kept.to_list()), a


# --------------------------------------------------------------------------------------------- ingestion flags
def _binance_raw_with_bad_rows(raw: Path) -> None:
    """The real 2024-12-31 BTCUSDT daily zip plus one duplicated line and one open time 14.789 s off the grid."""
    name = "BTCUSDT-1m-2024-12-31.zip"
    with zipfile.ZipFile(FIX / name) as zf:
        inner = zf.namelist()[0]
        lines = zf.read(inner).decode().splitlines()
    data = [ln for ln in lines if ln and ln[0].isdigit()]
    off = data[200].split(",")
    off[0] = str(int(off[0]) + 14_789)  # ms timestamps in 2024 files
    p = raw / "BTCUSDT" / "daily" / name
    p.parent.mkdir(parents=True)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(inner, "\n".join([*lines, data[100], ",".join(off)]) + "\n")


def test_raw_duplicates_and_moved_rows_reach_monthly_table(tmp_path: Path):
    from volrisk.data import binance as B

    day = date(2024, 12, 31)
    _binance_raw_with_bad_rows(tmp_path / "raw")
    bronze, silver = tmp_path / "bronze", tmp_path / "silver"
    s = B.build_bronze("BTC", tmp_path / "raw", bronze / "minute", start=day, end=day, verify=False)
    assert s["moved"] == 1 and s["duplicates"] == 2 and s["spacing_violations"] > 0  # ingester cleaned bronze
    minutes = pl.read_parquet(bronze / "minute" / "asset=BTC" / "year=2024" / "part.parquet")
    build_silver(silver, "BTC", minutes, utc_day_schedule("BTC", day, day))

    m = Q.monthly_coverage("BTC", bronze, silver, schedule_fn=utc_day_schedule, until=day)
    row = m.row(0, named=True)
    assert row["month"] == date(2024, 12, 1)
    assert row["n_dup_raw"] == s["duplicates"] and row["n_moved_raw"] == s["moved"]
    assert row["n_spacing_raw"] == s["spacing_violations"]
    assert row["n_dup_bronze"] == 0 and row["n_misaligned_bronze"] == 0  # bronze itself is clean
    assert row["n_misaligned_days"] is None  # Dukascopy-only column
    y = Q.yearly_coverage(m).row(0, named=True)
    assert y["n_dup_raw"] == 2 and y["n_moved_raw"] == 1


def test_flag_counts_dukascopy_days(tmp_path: Path):
    from volrisk.data import dukascopy as D

    rows = [
        ("EURUSD", date(2024, 1, 3), "misaligned"),
        ("EURUSD", date(2024, 1, 4), "misaligned"),
        ("EURUSD", date(2024, 1, 9), "missing_side"),
        ("EURUSD", date(2024, 2, 6), "continuity"),
        ("EURUSD", date(2024, 2, 7), "decode_error"),
        ("EURUSD", date(2025, 12, 1), "misaligned"),  # holdout: never counted
    ]
    df = pl.DataFrame([{"asset": a, "date": d, "flag": f, "detail": "x"} for a, d, f in rows],
                      schema={k: D.FLAG_SCHEMA[k] for k in ("asset", "date", "flag", "detail")})
    df.write_parquet(tmp_path / "dukascopy_EURUSD.parquet")
    fc = Q.flag_counts("EURUSD", tmp_path)
    assert fc["month"].to_list() == [date(2024, 1, 1), date(2024, 2, 1)]
    assert fc["n_misaligned_days"].to_list() == [2, 0]
    assert fc["n_missing_side_days"].to_list() == [1, 0]
    assert fc["n_continuity_days"].to_list() == [0, 1]
    assert fc["n_decode_error_days"].to_list() == [0, 1]
    assert fc["n_dup_raw"].null_count() == 2  # Binance-only column
    assert Q.flag_counts("BTC", tmp_path) is None  # no binance_BTC.parquet


# --------------------------------------------------------------------------------------------- gold-based
def test_jump_share_flags_suspect_years():
    d = pd.DataFrame(
        {
            "asset": ["BTC"] * 20,
            "session_date": pd.to_datetime([f"2023-01-{i + 1:02d}" for i in range(10)]
                                           + [f"2024-01-{i + 1:02d}" for i in range(10)]),
            "j": [0.1, 0.2, 0.3] + [0.0] * 7 + [0.5] + [0.0] * 9,
        }
    )
    js = Q.jump_share(d, by="year")
    assert js["year"].to_list() == [2023, 2024]
    assert js["share"].to_list() == pytest.approx([0.3, 0.1])
    assert js["suspect"].to_list() == [True, False]
    with pytest.raises(ValueError):
        Q.jump_share(d, by="week")


def test_jump_share_per_month_catches_a_bad_month():
    """A month of bad data is averaged away in the yearly share; SPEC §10 asks for asset × month."""
    days = [date(2023, 1, 1) + timedelta(days=i) for i in range(365)]
    rng = np.random.default_rng(4)
    j = np.where([d.month == 6 for d in days], 0.3, np.where(rng.random(365) < 0.05, 0.3, 0.0))
    d = pl.DataFrame({"asset": ["BTC"] * 365, "session_date": days, "j": j})
    assert not Q.jump_share(d, by="year")["suspect"][0]
    js = Q.jump_share(d)  # default: per asset × month
    assert js.height == 12 and js["month"][0] == date(2023, 1, 1)
    assert js.filter(pl.col("suspect"))["month"].to_list() == [date(2023, 6, 1)]
    assert js.filter(pl.col("month") == date(2023, 6, 1))["share"][0] == 1.0


def test_jump_review_separates_bad_prints_from_genuine_jumps(tmp_path: Path):
    """SPEC §10: a suspect jump-day share is reviewed, not filtered. March: every jump day is an isolated bad print
    (spike + reversal); April: genuine one-way jumps, one across null bars, and stale prices (zero returns)."""
    ds = weekdays(date(2024, 3, 1), date(2024, 4, 30))
    mar = [k for k, d in enumerate(ds) if d.month == 3][1:11]  # the first session has no previous close
    apr = [k for k, d in enumerate(ds) if d.month == 4]
    rng = np.random.default_rng(8)
    ret = rng.normal(0, 0.05, (len(ds), 78))
    for n, k in enumerate(mar):  # spike and reversal; on odd days the reversal is the larger return
        ret[k, 20], ret[k, 21] = 3.0, (-3.2 if n % 2 else -2.9)
    for k in apr[:10]:
        ret[k, 40] = 3.0
    ret[apr[0], 30], ret[apr[0], 31] = np.nan, 3.5  # bar 32 carries one return spanning bars 31-32
    ret[apr[0], 40] = 0.01
    ret[apr, 50:60] = 0.0  # ten stale bars per April session
    sess, bars = silver_from_returns(ds, ret)
    write_silver(tmp_path, "SPX", sess, bars.with_columns(asset=pl.lit("SPX")))
    jump_days = {ds[k] for k in mar + apr[:10]}
    gold = pl.DataFrame({"asset": ["SPX"] * (len(ds) - 1), "session_date": ds[1:]}).with_columns(
        j=pl.col("session_date").is_in(list(jump_days)).cast(pl.Float64) * 0.5,
        flag_partial=pl.col("session_date") == ds[mar[0]],
    )

    rev = Q.jump_review("SPX", gold, tmp_path, start=ds[0])
    assert rev["month"].to_list() == [date(2024, 3, 1), date(2024, 4, 1)]
    m, a = rev.row(0, named=True), rev.row(1, named=True)
    assert (m["n_jump"], m["n_reversed"], m["n_spanning"], m["n_partial"]) == (10, 10, 0, 1)
    assert (a["n_jump"], a["n_reversed"], a["n_spanning"], a["n_partial"]) == (10, 0, 1, 0)
    assert m["zero_ret_share"] == 0.0
    assert a["n_ret"] == 78 * len(apr) - 1 and a["n_zero"] == 10 * len(apr)
    assert a["zero_ret_share"] == pytest.approx(10 * len(apr) / (78 * len(apr) - 1))
    js = Q.jump_share(gold)  # same counts, share and flag as the SPEC §10 jump-day share
    assert rev.select("n_days", "n_jump", "share", "suspect").equals(
        js.select(pl.col("n_days").cast(pl.Int64), "n_jump", "share", "suspect"))
    assert rev["suspect"].to_list() == [True, True]

    no_flags = Q.jump_review("SPX", gold.drop("flag_partial"), tmp_path, start=ds[0])
    assert no_flags["n_partial"].null_count() == 2  # neither flag_partial nor coverage: unknown, not 0
    with_cov = gold.drop("flag_partial").with_columns(coverage=pl.lit(1.0))
    assert Q.jump_review("SPX", with_cov, tmp_path, start=ds[0])["n_partial"].to_list() == [0, 0]


def _write_fred(path: Path, dates, closes, blanks: dict[date, str]) -> None:
    lines = ["observation_date,SP500"]
    for d, c in zip(dates, closes):
        lines.append(f"{d.isoformat()},{blanks[d] if d in blanks else f'{c:.4f}'}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cfd_check_correlation_and_outlier(tmp_path: Path):
    rng = np.random.default_rng(7)
    dates = weekdays(date(2016, 9, 26), date(2017, 3, 31))
    fred_close = 2150 * np.exp(np.cumsum(rng.normal(0, 0.008, len(dates))))
    missing_fred, dropped_spx, outlier = date(2016, 11, 15), date(2017, 1, 18), date(2017, 2, 22)
    fred_csv = tmp_path / "fred_SP500_2026-10-01.csv"
    _write_fred(fred_csv, dates, fred_close, {missing_fred: ".", date(2016, 9, 28): ""})

    cfd_close = fred_close * np.exp(rng.normal(0, 3e-5, len(dates)))  # CFD vs cash index: tiny basis noise
    spx = pl.DataFrame({"session_date": dates, "p_close": cfd_close}).filter(pl.col("session_date") != dropped_spx)
    spx = spx.with_columns(r_cc=100 * (pl.col("p_close") / pl.col("p_close").shift(1)).log(), asset=pl.lit("SPX"))
    spx = spx.with_columns(r_cc=pl.when(pl.col("session_date") == outlier).then(pl.col("r_cc") + 0.30)
                           .otherwise(pl.col("r_cc")))

    res = Q.cfd_check(spx.to_pandas(), fred_csv)
    assert res["corr"] > 0.999
    assert [r["session_date"] for r in res["days_over_25bp"]] == [outlier]
    assert res["days_over_25bp"][0]["diff_bp"] == pytest.approx(30, abs=2)
    assert res["n_prev_mismatch"] == 2  # day after the missing FRED value and day after the dropped session
    n_common = sum(1 for d in dates if d >= Q.CFD_START and d not in (missing_fred, dropped_spx))
    assert res["n"] == n_common - 2
    assert res["mean_abs_diff_bp"] < 2


def test_read_fred_handles_missing_markers(tmp_path: Path):
    p = tmp_path / "f.csv"
    p.write_text("observation_date,SP500\n2024-01-02,4742.83\n2024-01-03,.\n2024-01-04,\n2024-01-05,4697.24\n")
    f = Q.read_fred_sp500(p)
    assert f["date"].to_list() == [date(2024, 1, 2), date(2024, 1, 5)]


def test_fred_fixture_parses():
    f = Q.read_fred_sp500(Path(__file__).parent / "fixtures" / "fred_sp500_sample.csv")
    assert f.height > 2000 and f["date"].min() == date(2016, 10, 3)
    assert f["close"].min() > 1000


# --------------------------------------------------------------------------------------------- 5-minute returns
def test_top_returns_order_spans_and_first_return(tmp_path: Path):
    ds = [date(2024, 5, 31), date(2024, 6, 3), date(2024, 6, 4), date(2024, 6, 5)]
    rng = np.random.default_rng(3)
    ret = rng.normal(0, 0.01, (4, 78))
    ret[0, 50] = 7.0  # first session: no previous close, not in gold
    ret[1, 9], ret[1, 10] = 2.0, -1.9  # bars 10, 11
    ret[2, 0] = -1.0  # first bar: return from p_open
    ret[2, 29:31] = np.nan  # bars 30-31 null ...
    ret[2, 31] = 1.5  # ... so bar 32 carries one return spanning 3 bars
    ret[3, 40] = 5.0  # but this session is invalid (coverage < 0.80)
    sess, bars = silver_from_returns(ds, ret, coverage=[1.0, 1.0, 76 / 78, 0.5])
    write_silver(tmp_path, "SPX", sess, bars.with_columns(asset=pl.lit("SPX")))

    top = Q.top_returns("SPX", tmp_path, n=4)
    assert top.height == 4
    assert top["r"].to_list() == pytest.approx([2.0, -1.9, 1.5, -1.0])
    assert top["bar_idx"].to_list() == [10, 11, 32, 1]
    assert top["span_bars"].to_list() == [1, 1, 3, 1]
    assert top["ts_prev"][3] == utc(ds[2], 14, 30)  # first return starts at the session open (p_open)
    assert top["ts_end"][2] == utc(ds[2], 14, 30) + timedelta(minutes=5 * 32)
    assert top["ts_prev"][2] == utc(ds[2], 14, 30) + timedelta(minutes=5 * 29)

    allr = Q.top_returns("SPX", tmp_path, n=2, valid_only=False)
    assert allr["r"].to_list() == pytest.approx([7.0, 5.0])


def _tz_tree(tmp_path: Path, shift_dst: int, seed: int = 11):
    ds = weekdays(date(2024, 1, 2), date(2024, 5, 31))
    i = np.arange(1, 79)

    def prof(x):  # opening spike at bar 1, closing bump at bar 78
        return 1 + 30 * np.exp(-np.abs(x - 1) / 1.5) + 5 * np.exp(-np.abs(78 - x) / 4)

    rng = np.random.default_rng(seed)
    ret = np.empty((len(ds), 78))
    for k, d in enumerate(ds):
        p = prof(i - shift_dst) if d >= date(2024, 3, 11) else prof(i)  # a clock bug shifts the DST profile
        ret[k] = rng.normal(0, 0.02, 78) * np.sqrt(p)
    sess, bars = silver_from_returns(ds, ret)
    # one half-day session (42 bars) with a huge spike must not enter the full-session profile
    hd = date(2024, 6, 3)
    hs, hb = silver_from_returns([hd], np.r_[np.full(4, 0.01), 50.0, np.full(37, 0.01)][None, :])
    write_silver(tmp_path, "SPX", pl.concat([sess, hs]), pl.concat([bars, hb]))
    n_dst = sum(d >= date(2024, 3, 11) for d in ds)
    return len(ds), n_dst


def test_tz_check_aligned_profiles(tmp_path: Path):
    n_sess, n_dst = _tz_tree(tmp_path, shift_dst=0)
    fig = tmp_path / "fig" / "dq_tz_SPX.png"
    tz = Q.tz_check("SPX", tmp_path, fig)
    assert tz["bar_idx"].to_list() == list(range(1, 79))
    # DST from 2024-03-10 in New York; the first session (no previous close) is not in gold
    assert tz["n_dst"][0] == n_dst and tz["n_std"][0] == n_sess - n_dst - 1
    assert tz["mean_r2_dst"].arg_max() == 0 and tz["mean_r2_std"].arg_max() == 0
    assert Q.profile_shift(tz) == 0
    assert fig.exists() and fig.stat().st_size > 1000


def test_tz_check_detects_shifted_spike(tmp_path: Path):
    _tz_tree(tmp_path, shift_dst=12)
    tz = Q.tz_check("SPX", tmp_path)
    assert int(tz["bar_idx"][tz["mean_r2_dst"].arg_max()]) == 13
    assert abs(Q.profile_shift(tz) + 12) <= 1  # a 1-hour shift, up to one bar of estimation noise
    al = Q.profile_alignment(tz)
    assert al["corr_best"] > 0.8 and al["corr_0"] < 0.3  # profiles only line up after the 1-hour shift


@pytest.mark.slow
@pytest.mark.parametrize("shift", [0, 6, 12])
def test_profile_shift_recovered_across_seeds(tmp_path: Path, shift: int):
    """Size and power of the DST alignment check: no false shift on a correct clock, and an injected shift
    is recovered to within one bar for every seed."""
    got = []
    for seed in range(15):
        d = tmp_path / str(seed)
        _tz_tree(d, shift_dst=shift, seed=seed)
        got.append(Q.profile_shift(Q.tz_check("SPX", d)))
    if shift == 0:
        assert got == [0] * 15
    else:
        assert all(abs(g + shift) <= 1 for g in got), got


# --------------------------------------------------------------------------------------------- signature
def _gbm_bronze(bronze: Path, asset: str, days: int, sigma: float, noise: float, seed: int) -> None:
    rng = np.random.default_rng(seed)
    n = days * 1440
    start = utc(date(2024, 3, 1))
    ts = [start + i * M1 for i in range(n)]
    eff = np.log(40000) + np.cumsum(rng.normal(0, sigma, n))
    obs = eff + rng.normal(0, noise, n) if noise else eff
    close = np.exp(obs)
    df = minutes_frame(ts, close)
    # an extra day with only 40% real minutes: excluded from the signature
    last_day = utc(date(2024, 3, 1) + timedelta(days=days))
    half = [last_day + i * M1 for i in range(1440) if i % 5 in (0, 1)]
    df = pl.concat([df, minutes_frame(half, np.exp(obs[-1]) * np.ones(len(half)))])
    write_bronze(bronze, asset, df)


def test_signature_flat_on_gbm(tmp_path: Path):
    _gbm_bronze(tmp_path, "BTC", days=20, sigma=1e-4, noise=0.0, seed=5)
    calls = []

    def sched(asset, start, end):
        calls.append((asset, start, end))
        return utc_day_schedule(asset, start, end)

    fig = tmp_path / "dq_signature_BTC.png"
    sig = Q.signature("BTC", tmp_path, sched, fig_path=fig, until=date(2024, 3, 31))
    assert calls and calls[0][0] == "BTC"
    assert sig["freq_min"].to_list() == list(Q.SIGNATURE_FREQS)
    assert set(sig["n_sessions"].to_list()) == {20}  # the 40%-coverage day is excluded
    rv = sig["mean_rv"].to_numpy()
    expected = 1440 * (100 * 1e-4) ** 2
    assert np.all(np.abs(rv / expected - 1) < 0.2)  # roughly flat at the true integrated variance
    assert sig.filter(pl.col("freq_min") == 5)["rel_5m"][0] == pytest.approx(1.0)
    assert fig.exists()


def test_signature_rises_with_microstructure_noise(tmp_path: Path):
    _gbm_bronze(tmp_path, "ETH", days=10, sigma=1e-4, noise=1e-4, seed=6)
    sig = Q.signature("ETH", tmp_path, utc_day_schedule, until=date(2024, 3, 31))
    rv = dict(zip(sig["freq_min"].to_list(), sig["mean_rv"].to_list()))
    assert rv[1] > 1.5 * rv[30]
    assert rv[1] > rv[2] > rv[5]


# --------------------------------------------------------------------------------------------- flags & report
def test_read_flags_absent_and_holdout_rows(tmp_path: Path):
    assert Q.read_flags(tmp_path / "nope").is_empty()
    d = tmp_path / "flags"
    d.mkdir()
    pl.DataFrame(
        {"asset": ["EURUSD", "EURUSD"], "date": [date(2015, 3, 2), date(2025, 12, 1)],
         "flag": ["continuity", "continuity"], "detail": ["jump 3%", "late"]}
    ).write_parquet(d / "dukascopy_EURUSD.parquet")
    pl.DataFrame({"asset": ["BTC"], "date": ["2018-02-08"], "flag": ["incident"]}).write_parquet(d / "binance.parquet")
    f = Q.read_flags(d)
    assert f.height == 2  # the holdout-dated row is not shown
    assert set(f["source_file"].to_list()) == {"dukascopy_EURUSD", "binance"}


@pytest.fixture
def report_tree(tmp_path: Path):
    bronze, silver, implied = tmp_path / "bronze", tmp_path / "silver", tmp_path / "raw" / "implied"
    rng = np.random.default_rng(21)

    # BTC: 10 UTC days of 1-minute GBM
    btc_days = [date(2024, 2, 1) + timedelta(days=i) for i in range(10)]
    ts = [utc(btc_days[0]) + i * M1 for i in range(len(btc_days) * 1440)]
    btc_min = minutes_frame(ts, 40000 * np.exp(np.cumsum(rng.normal(0, 1e-4, len(ts)))))
    write_bronze(bronze, "BTC", btc_min)
    btc_sess = build_silver(silver, "BTC", btc_min, utc_day_schedule("BTC", btc_days[0], btc_days[-1]))

    # SPX: 3 weeks of RTH minutes from 2016-10-03 (CFD-check start), with quotes
    spx_days = weekdays(date(2016, 9, 30), date(2016, 10, 21))
    sched_spx = rth_schedule(spx_days)
    rows = []
    for d in spx_days:
        rows += [utc(d, 14) + i * M1 for i in range(7 * 60 + 30)]  # 14:00-21:29 incl. non-RTH minutes
    spx_min = minutes_frame(rows, 2150 * np.exp(np.cumsum(rng.normal(0, 3e-4, len(rows)))),
                            spread=np.full(len(rows), 0.4))
    write_bronze(bronze, "SPX", spx_min)
    spx_sess = build_silver(silver, "SPX", spx_min, sched_spx("SPX", spx_days[0], spx_days[-1]))

    def daily_of(asset, sess, j_every):
        s = sess.sort("session_date").with_columns(
            asset=pl.lit(asset),
            r_cc=100 * (pl.col("p_close") / pl.col("p_close").shift(1)).log(),
            rv=pl.lit(1.0) + pl.int_range(pl.len()).cast(pl.Float64) / 10,
            j=pl.when(pl.int_range(pl.len()) % j_every == 0).then(0.2).otherwise(0.0),
            valid=pl.lit(True), flag_partial=pl.lit(False), M=pl.lit(78),
        )
        return s.select("asset", "session_date", "coverage", "valid", "flag_partial", "M", "p_close", "r_cc", "rv",
                        "j")

    gold = pl.concat([daily_of("BTC", btc_sess, 3), daily_of("SPX", spx_sess, 10)])
    spx_gold = gold.filter(pl.col("asset") == "SPX")
    implied.mkdir(parents=True)
    _write_fred(implied / "fred_SP500_2026-09-01.csv", spx_gold["session_date"].to_list(),
                spx_gold["p_close"].to_numpy(), {})
    (implied / "fred_SP500_2026-10-01.csv").write_text(  # newest snapshot wins
        (implied / "fred_SP500_2026-09-01.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    flags = bronze / "flags"
    flags.mkdir(parents=True)
    pl.DataFrame(
        {"asset": ["SPX", "SPX"], "date": [date(2016, 10, 5), date(2025, 12, 1)],
         "flag": ["continuity", "continuity"], "detail": ["first mid 2.4% off", "HOLDOUT-ROW"]}
    ).write_parquet(flags / "dukascopy_SPX.parquet")
    # 45 Binance flag rows (sorted before the Dukascopy file): the continuity flag must still be listed
    pl.DataFrame(
        {"asset": ["BTC"] * 45, "date": [btc_days[i % 10] for i in range(45)],
         "flag": [("moved", "duplicate", "spacing")[i % 3] for i in range(45)], "n_rows": [2] * 45,
         "detail": [f"row {i}" for i in range(45)]},
        schema={"asset": pl.Utf8, "date": pl.Date, "flag": pl.Utf8, "n_rows": pl.Int64, "detail": pl.Utf8},
    ).write_parquet(flags / "binance_BTC.parquet")

    def sched(asset, start, end):
        return utc_day_schedule(asset, start, end) if asset in ("BTC", "ETH") else sched_spx(asset, start, end)

    return tmp_path, bronze, silver, implied, gold.to_pandas(), sched


def test_build_report_writes_sections_and_figures(report_tree):
    root, bronze, silver, implied, gold, sched = report_tree
    out = Q.build_report(
        assets=("BTC", "SPX"),
        out_md=root / "reports" / "data_quality.md",
        fig_dir=root / "reports" / "figures",
        table_dir=root / "reports" / "tables",
        bronze_dir=bronze,
        silver_dir=silver,
        implied_dir=implied,
        daily=gold,
        schedule_fn=sched,
    )
    text = out.read_text(encoding="utf-8")
    assert text.startswith("# Data-quality report")
    for h in Q.REPORT_SECTIONS:
        assert h in text
    pos = [text.index(h) for h in Q.REPORT_SECTIONS]
    assert pos == sorted(pos)
    figs = root / "reports" / "figures"
    for name in ("dq_coverage.png", "dq_tz_BTC.png", "dq_tz_SPX.png", "dq_signature_BTC.png",
                 "dq_signature_SPX.png"):
        assert (figs / name).exists(), name
        assert f"figures/{name}" in text
    for a in ("BTC", "SPX"):
        assert (root / "reports" / "tables" / f"dq_monthly_{a}.csv").exists()
    assert "fred_SP500_2026-10-01.csv" in text
    assert "**1.00000**" in text  # identical synthetic SPX and FRED closes
    assert "not available" not in text

    # section 3: per asset x month (BTC Feb 2024: 4 of 10; SPX Sep 2016: its single session jumps)
    jumps = text[text.index(Q.REPORT_SECTIONS[2]):text.index(Q.REPORT_SECTIONS[3])]
    assert "Suspect asset-months: 2 of 3" in jumps
    assert "likely data problems" not in jumps and "manual review" in jumps  # a flag, not a finding (SPEC §10)
    assert "| BTC | 2024-02-01 | 10 | 4 | 40.0 | 0.00 | " in jumps  # review aids next to each suspect month
    assert "| SPX | 2016-09-01 | 1 | 1 | 100.0 | - | 0 | 0 | 0 |" in jumps  # first session: no returns in gold
    assert "| asset | months | n_months | n_days | n_jump | share_% | zero_ret_% | reversed_% |" in jumps
    assert "| BTC | suspect | 1 | 10 | 4 | 40.0 | 0.00 | " in jumps
    assert "| BTC | 2024 | 10 | 4 |" in jumps  # per-year summary kept

    # section 8: continuity flags listed on their own, other flags capped per file, n_rows summed
    flags = text[text.index(Q.REPORT_SECTIONS[7]):]
    assert "first mid 2.4% off" in flags and "HOLDOUT-ROW" not in text
    assert "| binance_BTC | BTC | duplicate | 15 | 30 |" in flags
    assert f"#### `binance_BTC` (first {Q.FLAG_ROWS_PER_FILE} by date of 45)" in flags
    assert flags.index("### Dukascopy continuity flags") < flags.index("### Other ingestion flags")
    assert "Listed: all of 1." in flags

    # section 1 + CSV: raw counts from the flag files, jump share per month
    btc = pl.read_csv(root / "reports" / "tables" / "dq_monthly_BTC.csv", try_parse_dates=True).row(0, named=True)
    assert (btc["n_dup_raw"], btc["n_moved_raw"], btc["n_spacing_raw"]) == (30, 30, 30)
    assert btc["jump_share"] == pytest.approx(0.4) and btc["jump_suspect"] is True
    assert btc["n_dropped"] == 1  # 2024-02-01: first session, no previous close
    spx = pl.read_csv(root / "reports" / "tables" / "dq_monthly_SPX.csv", try_parse_dates=True)
    assert spx["n_continuity_days"].to_list() == [0, 1] and spx["n_dup_raw"].null_count() == 2
    assert "Bronze integrity: 0 duplicate and 0 off-minute" in text
    assert "Modelling sample starts 2016-09-01 (SPX probe" in text


def test_build_report_tolerates_missing_inputs(tmp_path: Path):
    empty_gold = pl.DataFrame(schema={"asset": pl.Utf8, "session_date": pl.Date, "j": pl.Float64,
                                      "rv": pl.Float64, "r_cc": pl.Float64})
    out = Q.build_report(
        assets=("ETH", "EURUSD"),
        out_md=tmp_path / "dq.md",
        fig_dir=tmp_path / "figs",
        table_dir=None,
        bronze_dir=tmp_path / "bronze",
        silver_dir=tmp_path / "silver",
        implied_dir=tmp_path / "implied",
        daily=empty_gold,
        schedule_fn=utc_day_schedule,
    )
    text = out.read_text(encoding="utf-8")
    for h in Q.REPORT_SECTIONS:
        assert h in text
    assert "not available" in text
    assert "No flag rows found" in text
