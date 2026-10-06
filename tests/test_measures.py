"""Tests for realized measures and the gold daily table (SPEC §1 probe, §4.3, §5)."""

from __future__ import annotations

import math
import time
from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from scipy.special import gamma

from volrisk import config as C
from volrisk import measures as Ms

SEED = 20261002
CRIT = 3.090232306167813  # Phi^{-1}(0.999)


# ---------------------------------------------------------------------------------------------------------
# helpers


def _frames(
    asset: str,
    dates: list[date],
    prices: list[np.ndarray],
    p_open: np.ndarray,
    n_sched: list[int] | None = None,
    p_close: np.ndarray | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Silver-like (bars, sessions); NaN prices are null bars; p_close defaults to the last real bar."""
    lens = [len(p) for p in prices]
    n_sched = n_sched if n_sched is not None else lens
    flat = np.concatenate(prices) if prices else np.array([])
    bars = pl.DataFrame(
        {
            "asset": asset,
            "session_date": pl.Series(np.repeat(np.array(dates, dtype="datetime64[D]"), lens)).cast(pl.Date),
            "bar_idx": np.concatenate([np.arange(1, k + 1) for k in lens]).astype(np.int32),
            "price": pl.Series(flat, nan_to_null=True),
            "n_real_min": np.where(np.isnan(flat), 0, 5).astype(np.int32),
        }
    )
    n_real = np.array([int(np.sum(~np.isnan(p))) for p in prices])
    if p_close is None:
        p_close = np.array([p[~np.isnan(p)][-1] if np.any(~np.isnan(p)) else np.nan for p in prices])
    sessions = pl.DataFrame(
        {
            "asset": asset,
            "session_date": pl.Series(dates, dtype=pl.Date),
            "n_sched": pl.Series(n_sched, dtype=pl.Int32),
            "n_real_bars": pl.Series(n_real, dtype=pl.Int32),
            "coverage": n_real / np.asarray(n_sched, dtype=float),
            "p_open": pl.Series(np.asarray(p_open, dtype=float), nan_to_null=True),
            "p_close": pl.Series(np.asarray(p_close, dtype=float), nan_to_null=True),
        }
    )
    return bars, sessions


def _days(start: date, n: int, weekdays_only: bool = False) -> list[date]:
    out, d = [], start
    while len(out) < n:
        if not weekdays_only or d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _gbm(n_days: int, m: int, sigma_d: float, rng: np.random.Generator, jump_k: float = 0.0):
    """Continuous GBM (percent log returns, daily sd ``sigma_d`` %) cut into crypto sessions of m bars.

    With ``jump_k`` a single jump of ``jump_k`` 5-minute sds with random sign hits one random bar per day.
    """
    s5 = sigma_d / math.sqrt(m)
    r = rng.standard_normal((n_days, m)) * s5
    if jump_k:
        r[np.arange(n_days), rng.integers(0, m, n_days)] += rng.choice([-1.0, 1.0], n_days) * jump_k * s5
    logp = math.log(100.0) + np.concatenate([[0.0], np.cumsum(r.ravel())]) / 100.0
    p = np.exp(logp)
    p_open = p[: n_days * m : m]
    prices = [p[i * m + 1 : (i + 1) * m + 1] for i in range(n_days)]
    return p_open, prices


def _ref_measures(r: np.ndarray) -> dict:
    """Literal transcription of the SPEC §5.2 formulas (1-based sums), for M >= 5."""
    M = len(r)
    a = np.abs(r)
    mu43 = 2 ** (2 / 3) * gamma(7 / 6) / gamma(1 / 2)
    rv = sum(x * x for x in r)
    bv = (math.pi / 2) * (M / (M - 2)) * sum(a[i - 1] * a[i - 3] for i in range(3, M + 1))
    tq = (
        M
        * mu43**-3
        * (M / (M - 4))
        * sum((a[i - 1] * a[i - 3] * a[i - 5]) ** (4 / 3) for i in range(5, M + 1))
    )
    rq = (M / 3) * sum(x**4 for x in r)
    z = math.sqrt(M) * ((rv - bv) / rv) / math.sqrt((math.pi**2 / 4 + math.pi - 5) * max(1.0, tq / bv**2))
    j = (z > CRIT) * max(rv - bv, 0.0)
    return {
        "rv": rv,
        "bv": bv,
        "tq": tq,
        "rq": rq,
        "rs_pos": sum(x * x for x in r if x > 0),
        "rs_neg": sum(x * x for x in r if x < 0),
        "z_jump": z,
        "j": j,
        "c": rv - j,
        "M": M,
    }


# ---------------------------------------------------------------------------------------------------------
# §5.1–5.2 kernel


def test_constants():
    assert Ms.MU43 == pytest.approx(0.8308609, abs=1e-6)
    assert Ms.THETA == pytest.approx(0.6089938, abs=1e-6)
    assert Ms.jump_critical_value() == pytest.approx(CRIT, abs=1e-12)
    assert Ms.min_coverage() == 0.80


def test_intraday_returns_path_starts_at_p_open_and_spans_null_bars():
    prices = np.array([101.0, np.nan, np.nan, 103.0, 102.0, np.nan, 104.0])
    r = Ms.intraday_returns(prices, 100.0)
    path = np.array([100.0, 101.0, 103.0, 102.0, 104.0])
    np.testing.assert_allclose(r, 100 * np.diff(np.log(path)), rtol=0, atol=1e-13)
    assert r.size == 4  # one return spans the two null bars
    # None (object arrays) is treated like a null bar too
    r2 = Ms.intraday_returns(np.array([101.0, None, 103.0], dtype=object), 100.0)
    np.testing.assert_allclose(r2, 100 * np.diff(np.log([100.0, 101.0, 103.0])), atol=1e-13)
    assert Ms.intraday_returns(np.array([np.nan, np.nan]), 100.0).size == 0


@pytest.mark.parametrize("jump", [False, True])
def test_session_measures_match_spec_formulas(jump):
    rng = np.random.default_rng(SEED)
    r = rng.standard_normal(57) * 0.1
    if jump:
        r[20] += 3.0
    got = Ms.session_measures(r)
    ref = _ref_measures(r)
    assert list(got) == list(Ms.MEASURE_KEYS)
    assert got["M"] == 57
    for k in Ms.MEASURE_KEYS:
        assert got[k] == pytest.approx(ref[k], rel=1e-12, abs=1e-15), k
    assert (got["z_jump"] > CRIT) == jump
    if jump:
        assert got["j"] == pytest.approx(got["rv"] - got["bv"]) and got["c"] == pytest.approx(got["bv"])
    else:
        assert got["j"] == 0.0 and got["c"] == got["rv"]
    assert got["rs_pos"] + got["rs_neg"] == pytest.approx(got["rv"], rel=1e-14)


@pytest.mark.parametrize("m", [0, 1, 2, 3, 4])
def test_session_measures_fewer_than_five_returns(m):
    r = np.array([0.3, -0.2, 0.5, -0.1])[:m]
    got = Ms.session_measures(r)
    assert got["M"] == m
    assert got["rv"] == pytest.approx(float(np.sum(r**2)))
    assert got["bv"] == got["rv"]
    assert got["rq"] == pytest.approx(m / 3 * float(np.sum(r**4)))
    assert got["tq"] == got["rq"]
    assert got["z_jump"] == 0.0 and got["j"] == 0.0 and got["c"] == got["rv"]


def test_session_measures_flat_path_is_finite():
    got = Ms.session_measures(np.zeros(20))
    assert all(math.isfinite(v) for v in got.values())
    assert got["z_jump"] == 0.0 and got["j"] == 0.0
    # a single non-zero return: bv = tq = 0, pure jump (max(1, 0/0) taken as 1)
    one = Ms.session_measures(np.r_[np.zeros(10), 1.0, np.zeros(10)])
    assert one["bv"] == 0.0 and one["tq"] == 0.0
    assert one["z_jump"] == pytest.approx(math.sqrt(21) / math.sqrt(Ms.THETA))
    assert one["j"] == pytest.approx(1.0) and one["c"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------
# statistical behaviour on GBM (vectorised path)


def _gold_gbm(n_days: int, seed: int, jump_k: float = 0.0, sigma_d: float = 1.5) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    p_open, prices = _gbm(n_days, 288, sigma_d, rng, jump_k)
    bars, sessions = _frames("BTC", _days(date(2015, 1, 1), n_days), prices, p_open)
    return Ms.daily_measures(bars, sessions, "BTC")


def test_gbm_rv_bv_unbiased_and_jump_size():
    sigma_d = 1.5
    g = _gold_gbm(2000, SEED, sigma_d=sigma_d)
    assert g.height == 1999  # first kept session dropped (no previous close)
    assert (g["M"] == 288).all() and g["valid"].all() and not g["flag_partial"].any()
    true_var = sigma_d**2
    assert abs(g["rv"].mean() / true_var - 1) < 0.02
    assert abs(g["bv"].mean() / true_var - 1) < 0.03
    assert abs(g["tq"].mean() / true_var**2 - 1) < 0.05
    assert abs(g["rq"].mean() / true_var**2 - 1) < 0.05
    # crypto sessions are contiguous: gap ~ 0 and tv ~ rv
    np.testing.assert_allclose(g["gap"].to_numpy(), 0.0, atol=1e-10)
    false_rate = float((g["z_jump"] > CRIT).mean())
    assert 0.0 <= false_rate <= 0.005
    assert ((g["j"] > 0) == (g["z_jump"] > CRIT)).all()


def test_injected_jump_detected():
    # 15 five-minute sds: theoretical power ~99.7% at M = 288
    g15 = _gold_gbm(2000, SEED + 1, jump_k=15.0)
    assert float((g15["z_jump"] > CRIT).mean()) >= 0.99
    assert float((g15["j"] > 0).mean()) >= 0.99
    # 10 five-minute sds: BNS power is only ~80% at M = 288 (E[z] ~ 4.1)
    g10 = _gold_gbm(2000, SEED + 2, jump_k=10.0)
    assert float((g10["z_jump"] > CRIT).mean()) >= 0.70


def test_jump_statistic_centred_under_gaussian_null_but_shifted_by_fat_tails():
    """The kernel centres z_jump at ~0 under a Gaussian no-jump null (so the dev 5-minute median z of
    1.3-2.1 for BTC/ETH/EURUSD is a data property, not a kernel bias), while iid fat-tailed 5-minute
    returns without any discrete jump (realized kurtosis ~5 vs 3) push z far above 0 and flag > 25% of days
    -- the SPEC §10 'suspect' level -- with the pre-registered SPEC §5.2 test.
    """
    n, m = 2000, 288
    rng = np.random.default_rng(SEED + 6)
    gid = np.repeat(np.arange(n), m)
    gauss = Ms._grouped_measures(rng.standard_normal(n * m) * 0.1, gid, n)
    assert abs(float(np.median(gauss["z_jump"]))) < 0.15
    assert float((gauss["z_jump"] > CRIT).mean()) <= 0.005
    assert float(np.median(3 * gauss["rq"] / gauss["rv"] ** 2)) < 3.3
    fat = Ms._grouped_measures(rng.standard_t(5, n * m) * 0.1 / math.sqrt(5 / 3), gid, n)
    assert float(np.median(3 * fat["rq"] / fat["rv"] ** 2)) > 4.0
    assert float(np.median(fat["z_jump"])) > 2.0
    assert float((fat["j"] > 0).mean()) > 0.25


@pytest.mark.slow
def test_jump_test_size_and_power_large_sample():
    g = _gold_gbm(20000, SEED + 3)
    rate = float((g["z_jump"] > CRIT).mean())
    assert 0.0002 <= rate <= 0.004, rate  # nominal 0.1%; ~0.18% finite-sample at M = 288
    assert abs(g["rv"].mean() / 2.25 - 1) < 0.005
    assert abs(g["bv"].mean() / 2.25 - 1) < 0.01
    p10 = float((_gold_gbm(5000, SEED + 4, jump_k=10.0)["z_jump"] > CRIT).mean())
    assert 0.74 <= p10 <= 0.86, p10
    p20 = float((_gold_gbm(5000, SEED + 5, jump_k=20.0)["z_jump"] > CRIT).mean())
    assert p20 >= 0.999


# ---------------------------------------------------------------------------------------------------------
# identities and vectorised == scalar


def _random_sessions(asset: str, n: int, m: int, seed: int, null_share: float = 0.1):
    """Sessions with overnight gaps and randomly null bars (kept above the 80% coverage threshold)."""
    rng = np.random.default_rng(seed)
    p_open, prices, last = [], [], 100.0
    for _ in range(n):
        po = last * math.exp(rng.normal(0, 0.005))
        path = po * np.exp(np.cumsum(rng.normal(0, 0.001, m)))
        path[rng.random(m) < null_share] = np.nan
        path[-1] = po * math.exp(rng.normal(0, 0.01))  # last bar always real
        p_open.append(po)
        prices.append(path)
        last = path[-1]
    return np.array(p_open), prices


@pytest.mark.parametrize("asset,m", [("SPX", 78), ("EURUSD", 288)])
def test_identity_r_cc_gap_and_target(asset, m):
    n = 60
    p_open, prices = _random_sessions(asset, n, m, SEED + 7, null_share=0.05)
    dates = _days(date(2020, 1, 6), n, weekdays_only=True)
    bars, sessions = _frames(asset, dates, prices, p_open)
    g = Ms.daily_measures(bars, sessions, asset)
    assert g.columns == Ms.GOLD_COLUMNS
    assert dict(g.schema) == Ms.GOLD_SCHEMA
    assert g.height == n - 1 and g["session_date"].to_list() == dates[1:]
    for i, row in enumerate(g.iter_rows(named=True), start=1):
        r = Ms.intraday_returns(prices[i], p_open[i])
        p_close_prev = sessions["p_close"][i - 1]
        assert row["gap"] == pytest.approx(100 * math.log(p_open[i] / p_close_prev), abs=1e-11)
        assert row["r_cc"] == pytest.approx(100 * math.log(row["p_close"] / p_close_prev), abs=1e-11)
        assert abs(row["r_cc"] - (row["gap"] + r.sum())) < 1e-10
        assert row["tv"] == row["gap"] ** 2 + row["rv"]
        ref = Ms.session_measures(r)
        for k in Ms.MEASURE_KEYS:
            assert row[k] == pytest.approx(ref[k], rel=1e-10, abs=1e-14), k


def test_null_bars_spanned_by_one_return_in_daily_table():
    prices = [
        np.array([100.0, 100.5, 100.2, 100.4, 100.1, 100.3]),
        np.array([100.6, np.nan, np.nan, 101.0, 100.8, np.nan, 100.9, 101.2, 101.1, 101.3]),
    ]
    bars, sessions = _frames("SPX", [date(2020, 1, 6), date(2020, 1, 7)], prices, np.array([100.0, 100.4]))
    sessions = sessions.with_columns(coverage=pl.lit(1.0))  # isolate the path logic from validity
    g = Ms.daily_measures(bars, sessions, "SPX")
    assert g.height == 1
    path = np.array([100.4, 100.6, 101.0, 100.8, 100.9, 101.2, 101.1, 101.3])
    r = 100 * np.diff(np.log(path))
    assert g["M"][0] == 7 and g["n_real_bars"][0] == 7
    assert g["rv"][0] == pytest.approx(float(np.sum(r**2)), rel=1e-12)
    assert g["bv"][0] == pytest.approx(_ref_measures(r)["bv"], rel=1e-12)


# ---------------------------------------------------------------------------------------------------------
# validity rules (§4.3, §5.3)


def _full(rng, po, m=288):
    return po * np.exp(np.cumsum(rng.normal(0, 0.001, m)))


def test_crypto_partial_session_rules():
    rng = np.random.default_rng(SEED + 11)
    dates = _days(date(2021, 3, 1), 8)
    p_open = np.array([100.0, 101.0, 102.0, 103.0, 104.0, np.nan, 105.0, 106.0])
    prices = [_full(rng, po) if np.isfinite(po) else np.full(288, np.nan) for po in p_open]
    prices[2][::2] = np.nan  # coverage 0.50 -> partial, own measures, inherited rq
    prices[3][:115] = np.nan  # coverage 0.60 -> partial, rq inherited through s2 from s1
    prices[6][3:] = np.nan  # M = 3 -> partial with M < 5 fallbacks
    bars, sessions = _frames("BTC", dates, prices, p_open)
    g = Ms.daily_measures(bars, sessions, "BTC")
    # s0 dropped (first kept, no previous close), s5 dropped (no real minute)
    assert g["session_date"].to_list() == [dates[i] for i in (1, 2, 3, 4, 6, 7)]
    assert g["flag_partial"].to_list() == [False, True, True, False, True, False]
    # §4.3: crypto validity is not a coverage rule — every kept crypto session is valid, partial is a flag
    assert g["valid"].all()
    t = Ms._session_table(bars, sessions, "BTC")
    assert t["valid"].to_list() == [True] * 5 + [False] + [True] * 2  # only the session without prices
    own = {i: Ms.session_measures(Ms.intraday_returns(prices[i], p_open[i])) for i in (1, 2, 3, 4, 6, 7)}
    rows = dict(zip((1, 2, 3, 4, 6, 7), g.iter_rows(named=True)))
    # t-1 of the session after a partial one is that partial session (continuous calendar)
    assert rows[3]["gap"] == pytest.approx(100 * math.log(p_open[3] / rows[2]["p_close"]), abs=1e-12)
    assert rows[2]["coverage"] == pytest.approx(0.5) and rows[2]["M"] == 144
    for i in (1, 4, 7):
        assert rows[i]["rq"] == pytest.approx(own[i]["rq"], rel=1e-12)
    assert rows[2]["rq"] == pytest.approx(own[1]["rq"], rel=1e-12)
    assert rows[3]["rq"] == pytest.approx(own[1]["rq"], rel=1e-12)
    assert rows[6]["rq"] == pytest.approx(own[4]["rq"], rel=1e-12)
    for k in ("rv", "bv", "tq", "z_jump", "j"):  # partial sessions keep their own other measures
        assert rows[2][k] == pytest.approx(own[2][k], rel=1e-12, abs=1e-15), k
    s6 = rows[6]
    assert s6["M"] == 3
    assert s6["bv"] == s6["rv"] and s6["tq"] == pytest.approx(own[6]["rq"])  # tq := (own) rq
    assert s6["z_jump"] == 0.0 and s6["j"] == 0.0 and s6["c"] == s6["rv"]
    # gap / r_cc against the previous KEPT session (s4), skipping the dropped s5
    pc4 = rows[4]["p_close"]
    assert s6["gap"] == pytest.approx(100 * math.log(105.0 / pc4), abs=1e-12)
    assert s6["r_cc"] == pytest.approx(100 * math.log(s6["p_close"] / pc4), abs=1e-12)
    assert s6["tv"] == s6["gap"] ** 2 + s6["rv"]


@pytest.mark.parametrize("asset", ["SPX", "EURUSD"])
def test_fx_spx_invalid_sessions_dropped(asset):
    rng = np.random.default_rng(SEED + 13)
    dates = _days(date(2021, 3, 1), 8, weekdays_only=True)
    m = 78 if asset == "SPX" else 288
    p_open = np.array([100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    prices = [_full(rng, po, m) for po in p_open]
    n_sched = [m] * 8
    p_close = np.array([p[-1] for p in prices])
    prices[2][: int(0.3 * m)] = np.nan  # coverage 0.70 -> invalid
    p_close[2] = prices[2][-1]
    p_close[3] = np.nan  # missing p_close -> invalid
    prices[5] = prices[5][:42]  # half-day: 42 of 42 bars -> valid
    n_sched[5] = 42
    p_close[5] = prices[5][-1]
    prices[6] = prices[6][:4]  # coverage 1.0 but M = 4 -> invalid
    n_sched[6] = 4
    p_close[6] = prices[6][-1]
    bars, sessions = _frames(asset, dates, prices, p_open, n_sched=n_sched, p_close=p_close)
    g = Ms.daily_measures(bars, sessions, asset)
    assert g["session_date"].to_list() == [dates[i] for i in (1, 4, 5, 7)]
    assert g["valid"].all() and not g["flag_partial"].any()
    assert (g["coverage"] >= 0.8).all() and (g["M"] >= 5).all()
    rows = g.iter_rows(named=True)
    r1, r4, r5, r7 = rows
    assert r4["gap"] == pytest.approx(100 * math.log(p_open[4] / p_close[1]), abs=1e-12)
    assert r4["r_cc"] == pytest.approx(100 * math.log(p_close[4] / p_close[1]), abs=1e-12)
    assert r5["n_sched"] == 42 and r5["M"] == 42
    assert r7["gap"] == pytest.approx(100 * math.log(p_open[7] / p_close[5]), abs=1e-12)
    # reasons recorded on the full session table
    t = Ms._session_table(bars, sessions, asset)
    assert t["reason"].to_list() == [None, None, "low_coverage", "no_prices", None, None, "few_returns", None]


def test_daily_measures_filters_by_asset_and_is_fast():
    rng = np.random.default_rng(SEED + 17)
    n = 3200
    p_open, prices = _gbm(n, 288, 3.0, rng)
    prices = [p.copy() for p in prices]
    for p in prices:  # sprinkle null bars
        p[rng.random(288) < 0.05] = np.nan
    bars, sessions = _frames("BTC", _days(date(2017, 6, 1), n), prices, p_open)
    other_b, other_s = _frames("ETH", [date(2017, 6, 1)], [prices[0]], p_open[:1])
    bars, sessions = pl.concat([bars, other_b]), pl.concat([sessions, other_s])
    t0 = time.perf_counter()
    g = Ms.daily_measures(bars, sessions, "BTC")
    elapsed = time.perf_counter() - t0
    assert g.height == n - 1 and (g["asset"] == "BTC").all()
    assert elapsed < 10.0, elapsed
    k = 1234
    ref = Ms.session_measures(Ms.intraday_returns(prices[k], p_open[k]))
    row = g.row(k - 1, named=True)
    for key in Ms.MEASURE_KEYS:
        assert row[key] == pytest.approx(ref[key], rel=1e-10, abs=1e-14), key


# ---------------------------------------------------------------------------------------------------------
# SPX start probe (§1)


FULL_RTH = 371  # of 78 * 5 = 390 RTH minutes: 371 / 390 = 0.951 is complete, 370 / 390 = 0.949 is not


def _probe_frames(real_min: dict[date, int], n_sched: dict[date, int] | None = None):
    """SPX (sessions, bars) whose bars hold ``real_min[d]`` real minutes in total, spread evenly.

    With at least ``n_sched`` real minutes every bar has a price, so bar coverage is 1.0 whatever the
    minute share — the probe must look at minutes, not bars.
    """
    n_sched = n_sched or {}
    s_rows, b_rows = [], []
    for d, k in real_min.items():
        n = n_sched.get(d, 78)
        per = np.full(n, k // n)
        per[: k % n] += 1
        s_rows.append((d, n, int((per > 0).sum()), float((per > 0).mean())))
        b_rows += [(d, i + 1, 100.0 if per[i] else None, int(per[i])) for i in range(n)]
    s_schema = {"session_date": pl.Date, "n_sched": pl.Int32, "n_real_bars": pl.Int32, "coverage": pl.Float64}
    sessions = pl.DataFrame(s_rows, schema=s_schema, orient="row")
    bars = pl.DataFrame(
        b_rows,
        schema={"session_date": pl.Date, "bar_idx": pl.Int32, "price": pl.Float64, "n_real_min": pl.Int32},
        orient="row",
    )
    return sessions, bars


def test_rth_minute_share():
    d = [date(2013, 3, 4), date(2013, 3, 5), date(2013, 3, 6), date(2013, 3, 7)]
    sessions, bars = _probe_frames({d[0]: FULL_RTH, d[1]: FULL_RTH - 1, d[2]: 200, d[3]: 78}, {d[2]: 42})
    bars = bars.filter(pl.col("session_date") != d[3])  # a session without bar rows
    sh = Ms.rth_minute_share(bars, sessions)
    np.testing.assert_allclose(sh["minute_share"].to_numpy(), [371 / 390, 370 / 390, 200 / 210, 0.0])
    assert sh["coverage"].to_list() == [1.0, 1.0, 1.0, 1.0]  # bar coverage cannot tell them apart


def test_spx_model_start_probe():
    rng = np.random.default_rng(SEED + 19)
    # month -> number of incomplete sessions among 20; incomplete = 312..370 real minutes (all bars real)
    plan = {1: 20, 2: 10, 3: 2, 4: 1, 5: 0, 6: 5}
    real = {}
    for mth, bad in plan.items():
        k = np.r_[np.full(20 - bad, FULL_RTH), rng.integers(312, FULL_RTH, bad)]
        real |= {date(2012, mth, d): int(x) for d, x in zip(range(1, 21), rng.permutation(k))}
    sessions, bars = _probe_frames(real)
    assert (sessions["coverage"] == 1.0).all()
    assert Ms.spx_model_start(sessions, bars) == date(2012, 4, 1)  # 19/20 = 95% complete (boundary inclusive)
    early = pl.col("session_date") < date(2012, 4, 1)
    with pytest.raises(ValueError):
        Ms.spx_model_start(sessions.filter(early), bars.filter(early))


def test_spx_probe_counts_real_minutes_not_bars():
    # March 2013: every bar holds exactly 1 real minute -> bar coverage 1.0, real RTH minute share 0.2
    days = _days(date(2013, 3, 1), 21, weekdays_only=True)
    sessions, bars = _probe_frames({d: 78 for d in days})
    assert (sessions["coverage"] == 1.0).all()
    with pytest.raises(ValueError):
        Ms.spx_model_start(sessions, bars)
    # a scheduled session with no bars at all counts as incomplete (share 0) in the month's denominator
    april = _days(date(2013, 4, 1), 20, weekdays_only=True)
    s2, b2 = _probe_frames({d: FULL_RTH for d in april})
    assert Ms.spx_model_start(s2, b2) == date(2013, 4, 1)
    no_bars = b2.filter(~pl.col("session_date").is_in(april[:2]))  # 18/20 complete = 90%
    with pytest.raises(ValueError):
        Ms.spx_model_start(s2, no_bars)


def test_sample_start_probe_ignores_holdout_and_pre_start_months():
    h0 = C.holdout_start()
    dev_aug = _days(date(2025, 8, 1), 21, weekdays_only=True)
    dev_sep = _days(date(2025, 9, 1), 21, weekdays_only=True)
    hold = _days(h0, 44, weekdays_only=True)
    pre = _days(date(2011, 12, 1), 21, weekdays_only=True)  # before the SPX config start (2012-01-01)
    assert all(d < h0 for d in dev_sep) and all(d >= h0 for d in hold)

    # only holdout months qualify -> no dev start
    s, b = _probe_frames({**{d: 300 for d in dev_aug + dev_sep}, **{d: 390 for d in hold}})
    with pytest.raises(ValueError):
        Ms.sample_start("SPX", s, b)
    # a qualifying month before the config start is ignored as well
    s, b = _probe_frames({**{d: 390 for d in pre}, **{d: 300 for d in dev_aug + dev_sep}})
    with pytest.raises(ValueError):
        Ms.sample_start("SPX", s, b)

    # mixed: September qualifies; complete or broken holdout months do not move the start
    for hold_min in (390, 0):
        s, b = _probe_frames(
            {**{d: 390 for d in pre}, **{d: 300 for d in dev_aug}, **{d: 390 for d in dev_sep},
             **{d: hold_min for d in hold}}
        )
        assert Ms.sample_start("SPX", s, b) == date(2025, 9, 1)
    # other clocks: the config start, whatever the data
    assert Ms.sample_start("EURUSD", s, b) == C.asset("EURUSD").start
    assert Ms.sample_start("BTC", s, b) == C.asset("BTC").start


# ---------------------------------------------------------------------------------------------------------
# build_gold: sample starts and dev / holdout split


def _write_silver(silver, asset, bars, sessions):
    for kind, df in (("bars5m", bars), ("sessions", sessions)):
        p = silver / kind / f"asset={asset}" / "part.parquet"
        p.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(p)


def test_build_gold_split_and_starts(tmp_path):
    h0, end = C.holdout_start(), C.data_end()
    rng = np.random.default_rng(SEED + 23)
    silver, gold_dir, hold_dir = tmp_path / "silver", tmp_path / "gold", tmp_path / "holdout"

    # BTC: daily around the split, plus sessions straddling the end of data
    btc_dates = [*_days(h0 - timedelta(days=40), 80), end, end + timedelta(days=1)]
    btc_dates.insert(0, C.asset("BTC").start - timedelta(days=2))  # before the sample start
    p_open, prices = _gbm(len(btc_dates), 288, 3.0, rng)
    _write_silver(silver, "BTC", *_frames("BTC", btc_dates, prices, p_open))

    # EURUSD: weekdays, one low-coverage session in dev
    fx_dates = _days(h0 - timedelta(days=30), 40, weekdays_only=True)
    p_open, prices = _random_sessions("EURUSD", len(fx_dates), 288, SEED + 29, null_share=0.0)
    prices[5][:100] = np.nan
    _write_silver(silver, "EURUSD", *_frames("EURUSD", fx_dates, prices, p_open))

    # SPX: July sessions have a price in every bar (coverage 1.0, valid) but only 4 of 5 real minutes per
    # bar (80% of RTH minutes) -> not complete, so the probe starts in August
    spx_dates = _days(date(2025, 7, 1), 90, weekdays_only=True)
    p_open, prices = _random_sessions("SPX", len(spx_dates), 78, SEED + 31, null_share=0.0)
    spx_bars, spx_sessions = _frames("SPX", spx_dates, prices, p_open)
    july = pl.col("session_date").dt.month() == 7
    spx_bars = spx_bars.with_columns(n_real_min=pl.when(july).then(4).otherwise("n_real_min").cast(pl.Int32))
    assert (spx_sessions["coverage"] == 1.0).all()
    _write_silver(silver, "SPX", spx_bars, spx_sessions)

    summary = Ms.build_gold(("BTC", "EURUSD", "SPX"), silver, gold_dir, hold_dir)
    dev = pl.read_parquet(gold_dir / "daily.parquet")
    hold = pl.read_parquet(hold_dir / "daily.parquet")
    assert dev.columns == Ms.GOLD_COLUMNS and hold.columns == Ms.GOLD_COLUMNS
    assert dev["session_date"].max() < h0 and hold["session_date"].min() >= h0
    assert hold["session_date"].max() <= end
    assert set(dev["asset"]) == set(hold["asset"]) == {"BTC", "EURUSD", "SPX"}
    assert not list((tmp_path / "gold").glob("*.part")) and not list((tmp_path / "holdout").glob("*.part"))

    # the first in-sample session is kept: its t-1 is the previous valid session in silver (July 31)
    assert summary["SPX"]["start"] == date(2025, 8, 1)
    spx = dev.filter(pl.col("asset") == "SPX")
    assert spx["session_date"].min() == date(2025, 8, 1)
    jul31 = spx_sessions.filter(pl.col("session_date") == date(2025, 7, 31)).row(0, named=True)
    first = spx.row(0, named=True)
    assert first["gap"] == pytest.approx(100 * math.log(first["p_open"] / jul31["p_close"]), abs=1e-12)
    assert summary["BTC"]["start"] == C.asset("BTC").start
    btc_all = pl.concat([dev, hold]).filter(pl.col("asset") == "BTC").sort("session_date")
    assert btc_all["session_date"].min() == btc_dates[1]  # pre-start session only serves as t-1
    assert btc_all["session_date"].max() == end
    assert btc_all.height == 81  # all in-sample sessions up to the end of data
    for a in ("BTC", "EURUSD", "SPX"):
        assert summary[a]["dev_rows"] == dev.filter(pl.col("asset") == a).height
        assert summary[a]["holdout_rows"] == hold.filter(pl.col("asset") == a).height
    assert summary["EURUSD"]["dev_dropped"] == {"low_coverage": 1}
    assert set(summary["EURUSD"]) == {"start", "dev_rows", "holdout_rows", "dev_dropped", "dev_flag_partial"}

    # the first holdout row's gap is measured against the last dev close (computed before the split)
    first_hold = hold.filter(pl.col("asset") == "EURUSD").row(0, named=True)
    last_dev = dev.filter(pl.col("asset") == "EURUSD").row(-1, named=True)
    assert first_hold["gap"] == pytest.approx(100 * math.log(first_hold["p_open"] / last_dev["p_close"]))


def _crypto_silver(silver, asset, seed, scale=1.0):
    h0 = C.holdout_start()
    rng = np.random.default_rng(seed)
    dates = _days(h0 - timedelta(days=20), 40)
    p_open, prices = _gbm(len(dates), 288, 3.0 * scale, rng)
    _write_silver(silver, asset, *_frames(asset, dates, prices, p_open))


def test_build_gold_subset_rebuild_keeps_other_assets(tmp_path, monkeypatch):
    h0 = C.holdout_start()
    silver, gold_dir, hold_dir = tmp_path / "silver", tmp_path / "gold", tmp_path / "holdout"
    dev_p, hold_p = gold_dir / "daily.parquet", hold_dir / "daily.parquet"
    _crypto_silver(silver, "BTC", SEED + 41)
    _crypto_silver(silver, "ETH", SEED + 43)
    Ms.build_gold(("BTC", "ETH"), silver, gold_dir, hold_dir)
    dev0, hold0 = pl.read_parquet(dev_p), pl.read_parquet(hold_p)
    assert set(dev0["asset"]) == set(hold0["asset"]) == {"BTC", "ETH"}

    # rebuild BTC alone from changed silver: ETH rows survive unchanged in both files, BTC rows are replaced
    _crypto_silver(silver, "BTC", SEED + 47, scale=2.0)
    Ms.build_gold(("BTC", "BTC"), silver, gold_dir, hold_dir)
    dev1, hold1 = pl.read_parquet(dev_p), pl.read_parquet(hold_p)
    for old, new, is_dev in ((dev0, dev1, True), (hold0, hold1, False)):
        assert new.columns == Ms.GOLD_COLUMNS and dict(new.schema) == Ms.GOLD_SCHEMA
        assert ((new["session_date"] < h0) == is_dev).all()
        assert new.equals(new.sort("asset", "session_date"))
        assert not new.select("asset", "session_date").is_duplicated().any()
        assert new.filter(pl.col("asset") == "ETH").equals(old.filter(pl.col("asset") == "ETH"))
        b_old, b_new = old.filter(pl.col("asset") == "BTC"), new.filter(pl.col("asset") == "BTC")
        assert b_new["session_date"].equals(b_old["session_date"])
        assert not np.allclose(b_new["rv"].to_numpy(), b_old["rv"].to_numpy())

    # the same BTC rows as a fresh build from the new silver
    Ms.build_gold(("BTC",), silver, tmp_path / "g2", tmp_path / "h2")
    assert pl.read_parquet(tmp_path / "g2" / "daily.parquet").equals(dev1.filter(pl.col("asset") == "BTC"))

    # a rebuild of every configured asset replaces both files without reading them
    monkeypatch.setattr(C, "ASSETS", ("BTC", "ETH"))
    dev_p.write_bytes(b"not parquet")
    hold_p.write_bytes(b"not parquet")
    Ms.build_gold(("ETH", "BTC"), silver, gold_dir, hold_dir)
    assert pl.read_parquet(dev_p).equals(dev1) and pl.read_parquet(hold_p).equals(hold1)
